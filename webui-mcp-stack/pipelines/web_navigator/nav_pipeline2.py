"""
ReAct Web Navigator — Plan → Execute → Observe → Respond
Playwright MCP + OpenWebUI, with plan-first strategy and ReAct fallback.

- Planner: creates a JSON action plan
- Executor: runs steps (tolerant payloads + XY fallback)
- Observer: screenshot + URL + TITLE + VISIBLE_TEXT (truncated)
- Writer: human-friendly final answer; includes screenshot links
- ReAct fallback when plan fails/stalls (think → act → observe loop)
- Skips OpenWebUI "housekeeping" prompts (title/tags/follow-ups)

Valves:
- model:            vision-capable model id (e.g. gpt-4o-mini, deepseek-vl-1.5)
- openai_api_base:  OpenAI-compatible base (auto-detects /v1 vs /api/v1)
- openai_api_key:   API key (if your gateway requires it)

Playwright MCP is assumed at MCP_BASE; screenshots are saved under OUT_DIR and
served publicly via PUBLIC_BASE for clickable links.
"""

import os
import re
import json
import base64
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Union, Generator, Iterator, Any, Tuple

import httpx

# Try to expose valves in OpenWebUI Admin
try:
    from pydantic import BaseModel
except Exception:
    class BaseModel:
        def __init__(self, **kwargs): pass

# ---- MCP + public web server ----
MCP_BASE      = "http://91.99.79.208:3880/mcp_playwright"
PUBLIC_BASE   = "http://91.99.79.208:3888"
OUT_DIR       = "/tmp/playwright-output"

# ---- Controls ----
TIMEOUT       = 120.0
MAX_PLAN_STEPS = 10
MAX_REACT_STEPS = 16
SCREEN_DELAY  = 0.6
STUCK_WINDOW  = 3

# ============================= LLM SYSTEMS =============================

PLANNER_SYS = """You are the Planner in a multi-agent web automation team.
Given a user GOAL, produce a clear, bounded JSON-only PLAN of at most 10 steps,
using ONLY these actions:

- navigate { "url": string, "wait_until"?: "load"|"domcontentloaded"|"networkidle" }
- click { "selector": string }
- type { "selector": string, "text": string }
- press_key { "key": string }
- hover { "selector": string }
- wait { "ms": number }
- screenshot { "fullPage"?: boolean }
- done { "success": boolean, "message": string }

Prefer robust text selectors: button:has-text("Akzeptieren"), a:has-text("Login"),
text="Alle akzeptieren", etc. If a cookie banner likely appears, include a click
step for it before proceeding. If the GOAL includes a URL/domain, begin with navigate.

Return ONLY valid JSON:

{ "plan": [ { "op": "...", ... }, ... ] }
"""

REACT_SYS = """You are the ReAct Agent controlling Playwright MCP via REST.
You see the current OBSERVATION (URL, TITLE, partial VISIBLE_TEXT, and an image).
Produce the NEXT ACTION as JSON:

{ "thought": "why this next step", "action": { "op": "<tool>", ... } }

Allowed tools:
- navigate { "url": string, "wait_until"?: "load"|"domcontentloaded"|"networkidle" }
- click { "selector": string }
- type { "selector": string, "text": string }
- press_key { "key": string }
- hover { "selector": string }
- wait { "ms": number }
- screenshot { "fullPage"?: boolean }
- done { "success": boolean, "message": string }

Keep thoughts short. Use text-based selectors and handle cookie banners.
Return ONLY JSON.
"""

WRITER_SYS = """You are the Writer. Using the GOAL and the latest OBSERVATION
(URL, TITLE, partial VISIBLE_TEXT) plus the navigation TRACE, decide if the GOAL
is satisfied, and summarize clearly for a human.

Return ONLY JSON:
{
  "success": true|false,
  "message": "human-friendly summary/conclusion",
  "tips"?: "if not done, short hint for next step"
}
"""

# ============================ UTILITIES ============================

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def _now_png() -> str:
    return f"page-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.png"

def _join(base: str, tail: str) -> str:
    return f"{base.rstrip('/')}/{tail.lstrip('/')}"

def _looks_housekeeping(text: str) -> bool:
    if not text: return False
    pats = [
        r"concise, 3-5 word title",
        r"Generate 1-3 broad tags",
        r"Suggest 3-5 relevant follow-up questions",
        r"Your entire response must consist solely of the JSON object",
    ]
    return any(re.search(p, text, re.I) for p in pats)

# ============================ PIPELINE ============================

class Pipeline:
    class Valves(BaseModel):
        model: str = "gpt-4o-mini"
        openai_api_base: str = ""   # e.g. https://api.openai.com  OR your OpenWebUI proxy
        openai_api_key: str = ""    # leave empty if not required

    def __init__(self):
        self.name = "Web Navigator — Plan→Act→Observe→Respond (MCP)"
        self.description = "Plans first; executes with robust actions; observes; writes final answer; ReAct fallback."
        self.version = "2.1.0"
        self.author = "You"
        self.valves = self.Valves()
        os.makedirs(OUT_DIR, exist_ok=True)

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        goal = self._extract_user_goal(messages) or user_message or ""
        if not goal.strip():
            yield self._status("❌ No goal provided.", done=True); return

        if _looks_housekeeping(goal):
            yield self._status("⤴️ Skipping OpenWebUI housekeeping request.", done=True); return

        # ---- Resolve OpenAI-compatible backend ----
        chat_api_base, chat_api_key = self._extract_credentials(messages)
        openai_base = (self.valves.openai_api_base or os.getenv("OPENAI_API_BASE", "") or chat_api_base).strip().rstrip("/")
        openai_key  = (self.valves.openai_api_key  or os.getenv("OPENAI_API_KEY", "")  or chat_api_key).strip()
        chosen_model = (os.getenv("FORCE_MODEL") or self.valves.model or model_id or os.getenv("PIPELINE_DEFAULT_MODEL","")).strip()

        if not openai_base:
            yield self._status("❓ Set 'openai_api_base' valve or OPENAI_API_BASE env.", done=True); return
        if not chosen_model:
            yield self._status("❓ Set 'model' valve (vision-capable).", done=True); return

        headers = {"Content-Type": "application/json"}
        if openai_key:
            headers["Authorization"] = f"Bearer {openai_key}"

        # auto-detect /, /v1, /api, /api/v1 for chat URL
        with httpx.Client(timeout=TIMEOUT, headers=headers) as probe:
            models_url, chat_url, prefix = self._detect_paths(probe, openai_base)

        yield self._status(f"🔧 Using model: {chosen_model} @ {openai_base}{prefix or ''}")
        yield self._status(f"🎯 Goal: {goal}")

        # ---- Start session ----
        with httpx.Client(timeout=TIMEOUT) as http, httpx.Client(timeout=TIMEOUT, headers=headers) as llm:
            trace: List[Dict[str, Any]] = []
            screenshot_urls: List[str] = []

            # Optional initial navigate if goal contains a URL/domain
            start_url = self._extract_url(goal)
            if start_url:
                yield self._status(f"🧭 Planner→Executor: navigate to {start_url}")
                if not self._do_navigate(http, start_url):
                    yield self._status("❌ navigate failed.", done=True); return
                time.sleep(SCREEN_DELAY)
                b64, pub = self._take_screenshot(http)
                if b64 or pub:
                    screenshot_urls.append(pub) if pub else None
                    yield self._image_event(b64, pub)
                    if pub: yield self._status(f"🖼️ Screenshot: {pub}")
                url, title, txt = self._observe_text(http)
                trace.append({"op":"navigate","url":start_url,"obs":{"url":url,"title":title}})
                yield self._observation_event(url, title, txt)

            # ---- PLANNER: make a plan
            plan = self._plan(llm, chat_url, chosen_model, goal)
            if not plan:
                yield self._status("⚠️ Planner produced no plan; switching to ReAct.", done=False)
                return from_generator(self._react_loop(http, llm, chat_url, chosen_model, goal, trace, screenshot_urls, MAX_REACT_STEPS))

            yield self._status(f"📝 Planner: {len(plan)} step(s) created. Executing…")

            # ---- EXECUTOR: run plan
            for idx, step in enumerate(plan[:MAX_PLAN_STEPS], 1):
                op = str(step.get("op","")).lower().strip()
                yield self._status(f"⚙️ Executor [{idx}/{len(plan)}]: {op or 'unknown'}")

                ok = self._exec_op(http, step, yield_status=lambda s: (yield self._status(s)))
                if not ok:
                    yield self._status(f"⚠️ Step {idx} failed → falling back to ReAct.", done=False)
                    return from_generator(self._react_loop(http, llm, chat_url, chosen_model, goal, trace, screenshot_urls, MAX_REACT_STEPS))

                time.sleep(SCREEN_DELAY)

                # OBSERVE each step
                b64, pub = self._take_screenshot(http)
                if b64 or pub:
                    screenshot_urls.append(pub) if pub else None
                    yield self._image_event(b64, pub)
                    if pub: yield self._status(f"🖼️ Screenshot: {pub}")
                url, title, txt = self._observe_text(http)
                trace.append({"op": op, "args": {k:v for k,v in step.items() if k!='op'}, "obs":{"url":url,"title":title}})
                yield self._observation_event(url, title, txt)

                # if planner signals "done", stop early
                if op == "done":
                    break

            # ---- WRITER: judge success & respond
            b64, pub = self._take_screenshot(http)
            if b64 or pub:
                screenshot_urls.append(pub) if pub else None
                yield self._image_event(b64, pub)
                if pub: yield self._status(f"🖼️ Final Screenshot: {pub}")
            url, title, txt = self._observe_text(http)

            judgement = self._writer(llm, chat_url, chosen_model, goal, trace, url, title, txt)
            if not isinstance(judgement, dict):
                yield self._status("⚠️ Writer failed; switching to ReAct.", done=False)
                return from_generator(self._react_loop(http, llm, chat_url, chosen_model, goal, trace, screenshot_urls, MAX_REACT_STEPS))

            success = bool(judgement.get("success", False))
            message = judgement.get("message") or ("Done." if success else "Not done.")
            tips    = judgement.get("tips")

            # Final human-facing message
            lines = [("✅" if success else "⚠️") + " " + message]
            if tips and not success:
                lines.append(f"💡 Next: {tips}")
            if screenshot_urls:
                lines.append("\nScreenshots:")
                for u in screenshot_urls:
                    if u: lines.append(f"- {u}")

            yield "\n".join(lines)

    # ========================== Planning / ReAct / Writing ==========================

    def _plan(self, llm: httpx.Client, chat_url: str, model: str, goal: str) -> Optional[List[Dict[str, Any]]]:
        data = {
            "model": model,
            "messages": [
                {"role":"system","content":PLANNER_SYS},
                {"role":"user","content":f"GOAL:\n{goal}\nReturn ONLY the JSON per spec."}
            ],
            "temperature": 0.2,
            "stream": False,
        }
        try:
            r = llm.post(chat_url, json=data)
            r.raise_for_status()
            obj = self._extract_json(r)
            if isinstance(obj, dict) and isinstance(obj.get("plan"), list):
                return obj["plan"]
        except httpx.HTTPStatusError as e:
            pass
        except Exception:
            pass
        return None

    def _react_loop(self, http: httpx.Client, llm: httpx.Client, chat_url: str, model: str,
                    goal: str, trace: List[dict], screenshot_urls: List[str], max_steps: int):
        yield self._status("♻️ ReAct fallback engaged.")
        last_b64 = None; url=None; title=None; txt=None
        # initial observation (if needed)
        b64, pub = self._take_screenshot(http)
        if b64 or pub:
            screenshot_urls.append(pub) if pub else None
            yield self._image_event(b64, pub)
            if pub: yield self._status(f"🖼️ Screenshot: {pub}")
        url, title, txt = self._observe_text(http)

        obs_fps = []
        for step in range(1, max_steps+1):
            msgs = self._react_messages(goal, trace, last_b64 or b64, url, title, txt)
            try:
                r = llm.post(chat_url, json={"model": model, "messages": msgs, "temperature": 0.2, "stream": False})
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                yield self._status(f"❌ LLM HTTP {e.response.status_code}: {(e.response.text or '')[:300]}", done=True); return
            except Exception as e:
                yield self._status(f"❌ LLM error: {e}", done=True); return

            action_json = self._extract_json(r)
            if not isinstance(action_json, dict):
                yield self._status("❌ ReAct produced invalid JSON.", done=True); return

            thought = str(action_json.get("thought","")).strip()
            action  = action_json.get("action", {})
            if thought: yield self._status(f"🧠 ReAct {step}: {thought}")
            if not isinstance(action, dict) or "op" not in action:
                yield self._status("❌ ReAct action missing 'op'.", done=True); return

            op = str(action.get("op","")).lower().strip()
            if op == "done":
                msg = action.get("message") or "Task complete."
                yield self._status(f"✅ {msg}", done=True); return

            ok = self._exec_op(http, action, yield_status=lambda s: (yield self._status(s)))
            if not ok:
                yield self._status("⚠️ ReAct step failed; stopping.", done=True); return

            time.sleep(SCREEN_DELAY)
            b64, pub = self._take_screenshot(http)
            if b64 or pub:
                screenshot_urls.append(pub) if pub else None
                yield self._image_event(b64, pub)
                if pub: yield self._status(f"🖼️ Screenshot: {pub}")
            url, title, txt = self._observe_text(http)
            trace.append({"op":op,"args":{k:v for k,v in action.items() if k!='op'},"obs":{"url":url,"title":title}})

            # stuck detection
            self._remember_obs(obs_fps, url, title, txt)
            if len(obs_fps) >= STUCK_WINDOW and len(set(obs_fps[-STUCK_WINDOW:])) == 1:
                yield self._status("🤔 Stuck (page unchanged). Please give a hint (selector/button text).", done=True); return

        yield self._status("⚠️ Reached max steps in ReAct loop.", done=True)

    def _writer(self, llm: httpx.Client, chat_url: str, model: str,
                goal: str, trace: List[dict], url: Optional[str], title: Optional[str], txt: Optional[str]) -> Optional[Dict[str, Any]]:
        obs_lines = []
        if url: obs_lines.append(f"URL: {url}")
        if title: obs_lines.append(f"TITLE: {title}")
        if txt:
            obs_lines.append("VISIBLE_TEXT (truncated):")
            obs_lines.append(txt[:6000])
        messages = [
            {"role":"system","content":WRITER_SYS},
            {"role":"user","content":f"GOAL:\n{goal}"},
            {"role":"user","content":"TRACE:\n"+json.dumps(trace, ensure_ascii=False)},
            {"role":"user","content":"OBSERVATION:\n"+("\n".join(obs_lines) if obs_lines else "(none)")},
        ]
        try:
            r = llm.post(chat_url, json={"model": model, "messages": messages, "temperature": 0.0, "stream": False})
            r.raise_for_status()
            obj = self._extract_json(r)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # ========================== Action execution ==========================

    def _exec_op(self, http: httpx.Client, step: Dict[str, Any], yield_status) -> bool:
        op = str(step.get("op","")).lower().strip()
        if op == "navigate":
            url = step.get("url"); wait_until = step.get("wait_until","load")
            if not isinstance(url, str) or not url:
                yield_status("❌ navigate requires 'url'"); return False
            yield_status(f"🛠️ navigate → {url} (wait_until={wait_until})")
            return self._do_navigate(http, url, wait_until)

        if op == "click":
            sel = step.get("selector")
            if not isinstance(sel, str) or not sel:
                yield_status("❌ click requires 'selector'"); return False
            yield_status(f"🛠️ click → {sel}")
            return self._click_selector(http, sel)

        if op == "hover":
            sel = step.get("selector")
            if not isinstance(sel, str) or not sel:
                yield_status("❌ hover requires 'selector'"); return False
            yield_status(f"🛠️ hover → {sel}")
            return self._hover_selector(http, sel)

        if op == "type":
            sel = step.get("selector"); txt = step.get("text")
            if not isinstance(sel, str) or not isinstance(txt, str):
                yield_status("❌ type requires 'selector' and 'text'"); return False
            yield_status(f"🛠️ type → {sel} = {txt}")
            return self._type_text(http, sel, txt)

        if op == "press_key":
            key = step.get("key")
            if not isinstance(key, str) or not key:
                yield_status("❌ press_key requires 'key'"); return False
            yield_status(f"🛠️ press_key → {key}")
            r = self._try_post(http, "browser_press_key", {"key": key})
            return self._ok(r)

        if op == "wait":
            ms = step.get("ms", 1000)
            try: ms = int(ms)
            except Exception: ms = 1000
            yield_status(f"🛠️ wait → {ms}ms")
            r = self._try_post(http, "browser_wait_for", {"ms": ms})
            if not self._ok(r): time.sleep(ms/1000.0)
            return True

        if op == "screenshot":
            # No-op here; we always take screenshots in OBSERVE section anyway
            yield_status("🛠️ screenshot (captured in OBSERVE)")
            return True

        if op == "done":
            # Let caller stop the loop
            yield_status("🛠️ done")
            return True

        yield_status(f"ℹ️ Unknown op '{op}'")
        return False

    # ---- tolerant actions + fallbacks ----

    def _try_post(self, http: httpx.Client, endpoint: str, payload: dict) -> httpx.Response:
        return http.post(_join(MCP_BASE, endpoint), json=payload)

    def _click_selector(self, http: httpx.Client, selector: str) -> bool:
        r = self._try_post(http, "browser_click", {"selector": selector})
        if self._ok(r): return True

        r = self._try_post(http, "browser_click", {"element": {"selector": selector}})
        if self._ok(r): return True

        for ref_payload in ({"ref": {"selector": selector}}, {"ref": selector}):
            r = self._try_post(http, "browser_click", ref_payload)
            if self._ok(r): return True

        coords = self._css_center_xy(http, selector)
        if coords:
            x, y = coords
            self._try_post(http, "browser_mouse_move_xy", {"x": x, "y": y})
            r = self._try_post(http, "browser_mouse_click_xy", {"x": x, "y": y})
            return self._ok(r)
        return False

    def _hover_selector(self, http: httpx.Client, selector: str) -> bool:
        r = self._try_post(http, "browser_hover", {"selector": selector})
        if self._ok(r): return True

        r = self._try_post(http, "browser_hover", {"element": {"selector": selector}})
        if self._ok(r): return True

        for ref_payload in ({"ref": {"selector": selector}}, {"ref": selector}):
            r = self._try_post(http, "browser_hover", ref_payload)
            if self._ok(r): return True

        coords = self._css_center_xy(http, selector)
        if coords:
            x, y = coords
            r = self._try_post(http, "browser_mouse_move_xy", {"x": x, "y": y})
            return self._ok(r)
        return False

    def _type_text(self, http: httpx.Client, selector: str, text: str) -> bool:
        r = self._try_post(http, "browser_type", {"selector": selector, "text": text})
        if self._ok(r): return True

        r = self._try_post(http, "browser_type", {"element": {"selector": selector}, "text": text})
        if self._ok(r): return True

        r = self._try_post(http, "browser_fill_form", {"fields": [{"selector": selector, "value": text}]})
        if self._ok(r): return True

        if self._focus_via_js(http, selector):
            for ch in text:
                rr = self._try_post(http, "browser_type", {"text": ch})
                if not self._ok(rr): break
            else:
                return True
        return False

    def _css_center_xy(self, http: httpx.Client, selector: str) -> Optional[tuple]:
        js = f"""(() => {{
          const el = document.querySelector({json.dumps(selector)});
          if(!el) return null;
          const r = el.getBoundingClientRect();
          return {{ "x": r.left + r.width/2, "y": r.top + r.height/2 }};
        }})()"""
        for p in ({"expression": js}, {"script": js}, {"code": js}, {"js": js}):
            try:
                r = self._try_post(http, "browser_evaluate", p)
                if self._ok(r):
                    data = self._safe_json(r)
                    if isinstance(data, str):
                        try: obj = json.loads(data)
                        except Exception: obj = None
                    else:
                        obj = data
                    if isinstance(obj, dict) and "x" in obj and "y" in obj:
                        return (float(obj["x"]), float(obj["y"]))
            except Exception:
                pass
        return None

    def _focus_via_js(self, http: httpx.Client, selector: str) -> bool:
        js = f"""(() => {{
          const el = document.querySelector({json.dumps(selector)});
          if(!el) return false;
          el.focus();
          return document.activeElement === el;
        }})()"""
        for p in ({"expression": js}, {"script": js}, {"code": js}, {"js": js}):
            try:
                r = self._try_post(http, "browser_evaluate", p)
                if self._ok(r):
                    data = self._safe_json(r)
                    if data in (True, "true", "True", 1, "1"):
                        return True
            except Exception:
                pass
        return False

    # ========================== Observe (Screenshot + Text) ==========================

    def _observe_text(self, http: httpx.Client) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        js = "JSON.stringify({url: location.href, title: document.title, text: document.body ? document.body.innerText : ''})"
        for p in ({"expression": js}, {"script": js}, {"code": js}, {"js": js}):
            try:
                r = self._try_post(http, "browser_evaluate", p)
                if self._ok(r):
                    obj = self._safe_json(r)
                    if isinstance(obj, str):
                        try: parsed = json.loads(obj)
                        except Exception: parsed = {}
                        return parsed.get("url"), parsed.get("title"), parsed.get("text")
                    if isinstance(obj, dict):
                        return obj.get("url"), obj.get("title"), obj.get("text")
            except Exception:
                pass
        # fallback to snapshot HTML
        try:
            r = self._try_post(http, "browser_snapshot", {})
            if self._ok(r):
                obj = self._safe_json(r)
                html = None
                if isinstance(obj, dict):
                    html = obj.get("html") or obj.get("content") or obj.get("data")
                elif isinstance(obj, str) and "<html" in obj.lower():
                    html = obj
                if not html and ("<html" in (r.text or "").lower()):
                    html = r.text
                if html:
                    return None, None, self._html_to_text(html)
        except Exception:
            pass
        return None, None, None

    def _html_to_text(self, html: str) -> str:
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ")
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ")
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()[:12000]

    def _take_screenshot(self, http: httpx.Client, full: bool = True, path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Base64-first for reliable inline rendering, then write to disk for a public URL.
        Also handles servers that reply with '### Result ... saved it as ...'.
        """
        if not path:
            path = os.path.join(OUT_DIR, _now_png())

        # base64-first
        r2 = self._try_post(http, "browser_take_screenshot", {"fullPage": full, "return": "base64"})
        if self._ok(r2):
            obj = self._safe_json(r2); b64 = None
            if isinstance(obj, dict):
                b64 = obj.get("data") or obj.get("base64")
            elif isinstance(obj, str) and len(obj) > 64 and "### Result" not in obj:
                b64 = obj
            if isinstance(b64, str) and b64.startswith("data:image"):
                try: b64 = b64.split(",", 1)[1]
                except Exception: pass
            if b64:
                try:
                    with open(path, "wb") as f: f.write(base64.b64decode(b64))
                    public = _join(PUBLIC_BASE, os.path.basename(path))
                except Exception:
                    public = None
                return b64, public

        # file-save fallback
        r = self._try_post(http, "browser_take_screenshot", {"fullPage": full, "filename": path})
        if self._ok(r) and os.path.exists(path):
            with open(path, "rb") as f: b64 = _b64(f.read())
            return b64, _join(PUBLIC_BASE, os.path.basename(path))

        # parse "### Result ... saved it as /tmp/..."
        body_text = ""
        try: body_text = r.text or ""
        except Exception: pass
        m = re.search(r"saved it as\s+(/[^'\s]+\.png)", body_text, flags=re.IGNORECASE)
        if m:
            server_path = m.group(1)
            for candidate in (server_path, os.path.join(OUT_DIR, os.path.basename(server_path))):
                if os.path.exists(candidate):
                    with open(candidate, "rb") as f: b64 = _b64(f.read())
                    return b64, _join(PUBLIC_BASE, os.path.basename(candidate))

        return None, None

    # ========================== LLM helpers & misc ==========================

    def _react_messages(self, goal: str, trace: List[dict],
                        image_b64: Optional[str], url: Optional[str], title: Optional[str], text: Optional[str]):
        obs_lines = []
        if url: obs_lines.append(f"URL: {url}")
        if title: obs_lines.append(f"TITLE: {title}")
        if text:
            obs_lines.append("VISIBLE_TEXT (truncated):")
            obs_lines.append(text[:6000])

        msgs: List[Dict[str, Any]] = [{"role":"system","content":REACT_SYS}]
        msgs.append({"role":"user","content":f"GOAL:\n{goal}\nReturn ONLY the JSON per spec."})
        if trace:
            msgs.append({"role":"user","content":"RECENT_STEPS:\n"+json.dumps(trace, ensure_ascii=False)})

        if image_b64:
            content: List[Dict[str, Any]] = []
            if obs_lines:
                content.append({"type":"text","text":"OBSERVATION:\n" + "\n".join(obs_lines)})
            content.append({"type":"image_url","image_url":{"url": f"data:image/png;base64,{image_b64}"}})
            msgs.append({"role":"user","content":content})
        elif obs_lines:
            msgs.append({"role":"user","content":"OBSERVATION:\n" + "\n".join(obs_lines)})

        return msgs

    def _detect_paths(self, client: httpx.Client, base: str) -> Tuple[Optional[str], str, str]:
        prefixes = ["", "/v1", "/api", "/api/v1"]
        for pref in prefixes:
            url = _join(base, _join(pref, "models"))
            try:
                r = client.get(url)
                if 200 <= r.status_code < 300:
                    return url, _join(base, _join(pref, "chat/completions")), pref
            except Exception:
                pass
        return None, _join(base, "v1/chat/completions"), "/v1"

    def _fetch_models(self, models_url: str, headers: Dict[str,str]) -> Optional[List[Dict[str, Any]]]:
        try:
            r = httpx.get(models_url, headers=headers, timeout=TIMEOUT)
            if 200 <= r.status_code < 300:
                data = r.json()
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    return data["data"]
        except Exception:
            pass
        return None

    def _extract_json(self, llm_resp: httpx.Response) -> Optional[Dict[str, Any]]:
        try:
            data = llm_resp.json()
            msg = data["choices"][0]["message"]["content"]
        except Exception:
            return None
        # extract first JSON object
        try:
            start = msg.index("{")
            depth = 0; end = start
            for i, ch in enumerate(msg[start:], start=start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i+1
                        break
            return json.loads(msg[start:end])
        except Exception:
            try: return json.loads(msg)
            except Exception: return None

    def _extract_user_goal(self, messages: List[dict]) -> Optional[str]:
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str): return c
                if isinstance(c, list):
                    texts = [b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"]
                    txt = "\n".join(t for t in texts if t); 
                    if txt: return txt
        return None

    def _extract_credentials(self, messages: List[dict]) -> Tuple[str, str]:
        base = ""; key = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                text = c if isinstance(c, str) else "\n".join([b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"]) if isinstance(c, list) else None
                if not text: continue
                # JSON blob
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        base = obj.get("openai_api_base", base)
                        key  = obj.get("openai_api_key", key)
                except Exception:
                    pass
                # KEY=VALUE fallbacks
                m1 = re.search(r"OPENAI_API_BASE\s*=\s*([^\s]+)", text or "")
                m2 = re.search(r"OPENAI_API_KEY\s*=\s*([^\s]+)", text or "")
                if m1: base = m1.group(1)
                if m2: key  = m2.group(1)
        return base or "", key or ""

    def _do_navigate(self, http: httpx.Client, url: str, wait_until: str = "load") -> bool:
        r = self._try_post(http, "browser_navigate", {"url": url, "wait_until": wait_until})
        return self._ok(r)

    def _safe_json(self, resp: httpx.Response) -> Any:
        try: return resp.json()
        except Exception: return None

    def _extract_url(self, text: str) -> Optional[str]:
        m = re.search(r"(https?://[^\s]+)", text)
        if m: return m.group(1).strip()
        m = re.search(r"([a-zA-Z0-9.-]+\.[a-z]{2,})", text)
        if m:
            d = m.group(1)
            if not d.startswith("http"): d = "https://" + d
            return d
        return None

    def _remember_obs(self, fp_list: List[str], url: Optional[str], title: Optional[str], text: Optional[str]):
        h = hashlib.sha256()
        h.update((url or "").encode()); h.update((title or "").encode()); h.update((text or "").encode())
        fp_list.append(h.hexdigest())
        if len(fp_list) > STUCK_WINDOW: fp_list[:] = fp_list[-STUCK_WINDOW:]

    def _ok(self, resp: httpx.Response) -> bool:
        return 200 <= resp.status_code < 300

    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}

    def _observation_event(self, url: Optional[str], title: Optional[str], text: Optional[str]) -> Dict:
        parts = []
        if url: parts.append(f"URL: {url}")
        if title: parts.append(f"TITLE: {title}")
        if text: parts.append(f"TEXT: {text[:500]}{'…' if text and len(text) > 500 else ''}")
        return self._status("👀 Observation:\n" + ("\n".join(parts) if parts else "(none)"))

    def _image_event(self, b64: Optional[str], url: Optional[str]) -> Dict:
        data: Dict[str, Any] = {"mime_type": "image/png"}
        if b64: data["base64"] = b64
        if url: data["path"] = url   # clickable in OpenWebUI
        return {"event": {"type": "image", "data": data}}

# Small helper to allow returning a generator from inside pipe
def from_generator(gen):
    for x in gen:
        yield x
