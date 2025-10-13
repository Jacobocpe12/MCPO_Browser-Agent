"""
ReAct Web Navigator — Playwright MCP + OpenWebUI (valves for model + API base/key)

- Directly calls: http://91.99.79.208:3880/mcp_playwright (no /v1/run)
- ReAct loop: THINK -> ACT -> OBSERVE (screenshot + URL + TITLE + VISIBLE_TEXT)
- Streams status + screenshots (inline base64 + public URL) to OpenWebUI
- Knobs (valves): model, openai_api_base, openai_api_key
- Model selection priority: FORCE_MODEL env > valves.model > chat model_id > PIPELINE_DEFAULT_MODEL env
- API creds priority: valves > env > hardcoded > user message
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

# Try to import pydantic for valves (OpenWebUI pipelines include it)
try:
    from pydantic import BaseModel
except Exception:
    class BaseModel:  # fallback no-op if pydantic missing
        def __init__(self, **kwargs): pass

MCP_BASE      = "http://91.99.79.208:3880/mcp_playwright"
PUBLIC_BASE   = "http://91.99.79.208:3888"
OUT_DIR       = "/tmp/playwright-output"
TIMEOUT       = 120.0
MAX_STEPS     = 16
SCREEN_DELAY  = 0.5     # small UI settle delay
STUCK_WINDOW  = 3       # if last 3 observations identical -> ask user

# Optional hardcoded OpenAI-compatible creds (last-resort fallback)
HARDCODE_API_BASE = ""  # e.g. "https://api.openai.com"
HARDCODE_API_KEY  = ""  # e.g. "sk-..."

SYSTEM_PROMPT = """You are a web-browsing agent controlling Playwright MCP via REST.
Follow a strict ReAct loop: THINK -> ACT -> OBSERVE (from latest screenshot and page text) until the user's goal is satisfied.

TOOLS (choose exactly one per step):
- navigate { "url": string, "wait_until": "load"|"domcontentloaded"|"networkidle" }
- click { "selector": string }          // prefer text selectors: text="Akzeptieren", button:has-text("Kartenviewer")
- type { "selector": string, "text": string }
- press_key { "key": string }           // e.g. "Enter", "Escape"
- hover { "selector": string }
- wait { "ms": number }                 // milliseconds
- screenshot { "fullPage": boolean, "filename"?: string }
- done { "success": boolean, "message": string }

RETURN FORMAT (valid JSON only, no extra prose):
{
  "thought": "1-2 sentences explaining your next move",
  "action": { "op": "<one of the tools above>", "...": "..." }
}

Guidelines:
- Always include a short 'thought'.
- Use text-based selectors first; try cookie consent buttons if they block content (e.g., "Alle akzeptieren", "Akzeptieren", "Accept all").
- After navigation or meaningful interaction, take a screenshot within 1-2 steps to re-orient.
- OBSERVATION provided to you includes: current URL, page TITLE, and visible TEXT (truncated), plus a screenshot.
- When the goal is satisfied, return { "op": "done", "success": true, "message": "..." }.
"""

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def _now_png() -> str:
    return f"page-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.png"


class Pipeline:
    class Valves(BaseModel):
        model: str = "gpt-4o-mini"                # vision-capable model id
        openai_api_base: str = ""                 # e.g. https://api.openai.com
        openai_api_key: str = ""                  # api key (server-side)

    def __init__(self):
        self.name = "ReAct Web Navigator (Playwright MCP)"
        self.description = "LLM-driven ReAct loop with valves for model + API base/key and full OBSERVE."
        self.version = "1.4.0"
        self.author = "You"

        # Exposed knobs (editable in Admin → Pipelines)
        self.valves = self.Valves()

        os.makedirs(OUT_DIR, exist_ok=True)

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        goal = self._extract_user_goal(messages) or user_message or ""
        if not goal.strip():
            yield self._status("❌ No goal provided.", done=True)
            return

        # ----- Resolve API base/key: valves -> env -> hardcoded -> from user message -----
        chat_api_base, chat_api_key = self._extract_credentials(messages)
        openai_base = (
            (self.valves.openai_api_base or "").rstrip("/")
            or os.getenv("OPENAI_API_BASE", "").rstrip("/")
            or HARDCODE_API_BASE.rstrip("/")
            or chat_api_base.rstrip("/")
        )
        openai_key = (
            self.valves.openai_api_key
            or os.getenv("OPENAI_API_KEY", "")
            or HARDCODE_API_KEY
            or chat_api_key
        )

        if not openai_base or not openai_key:
            yield self._status(
                "❓ Missing API credentials. Set them in the pipeline valves, env vars, or paste in chat:\n"
                "OPENAI_API_BASE=https://<host>  OPENAI_API_KEY=sk-...\n"
                'or JSON: {"openai_api_base":"...","openai_api_key":"..."}',
                done=True,
            )
            return

        # ----- Resolve model explicitly: FORCE_MODEL > valves.model > chat model_id > PIPELINE_DEFAULT_MODEL -----
        chosen_model = (
            os.getenv("FORCE_MODEL")
            or (self.valves.model or "").strip()
            or (model_id or "").strip()
            or os.getenv("PIPELINE_DEFAULT_MODEL", "")
        ).strip()
        if not chosen_model:
            yield self._status(
                "❓ No model selected. Set FORCE_MODEL or the 'model' valve, "
                "or pick a vision-capable model in the chat.",
                done=True,
            )
            return

        yield self._status(f"🔧 Using model: {chosen_model} @ {openai_base}")
        yield self._status(f"🎯 Goal: {goal}")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}

        with httpx.Client(timeout=TIMEOUT) as http, httpx.Client(base_url=openai_base, headers=headers, timeout=TIMEOUT) as llm:
            start_url = self._extract_url(goal)
            trace: List[Dict[str, Any]] = []
            last_image_b64: Optional[str] = None
            last_text_snippet: Optional[str] = None
            current_url: Optional[str] = None
            current_title: Optional[str] = None
            obs_fps: List[str] = []

            # Optional initial navigate
            if start_url:
                yield self._status(f"🛠️ Boot: navigate → {start_url}")
                if not self._do_navigate(http, start_url):
                    yield self._status("❌ Initial navigate failed.", done=True); return
                time.sleep(SCREEN_DELAY)
                last_image_b64, public_url = self._take_screenshot(http)
                if last_image_b64 or public_url:
                    yield self._image_event(last_image_b64, public_url)
                current_url, current_title, last_text_snippet = self._observe_text(http)
                obs_ev = self._observation_event(current_url, current_title, last_text_snippet)
                if obs_ev: yield obs_ev
                trace.append({"thought": "navigated to start url", "action": {"op": "navigate", "url": start_url}})
                self._remember_obs(obs_fps, current_url, current_title, last_text_snippet)

            for step in range(1, MAX_STEPS + 1):
                # ---- LLM decides next action (ReAct) ----
                msgs = self._build_messages(
                    goal, trace, last_image_b64, current_url, current_title, last_text_snippet
                )
                try:
                    resp = llm.post("/v1/chat/completions", json={
                        "model": chosen_model,
                        "messages": msgs,
                        "temperature": 0.2,
                        "stream": False,
                    })
                    resp.raise_for_status()
                except Exception as e:
                    yield self._status(f"❌ LLM error: {e}", done=True); return

                action_json = self._extract_json(resp)
                if not isinstance(action_json, dict):
                    yield self._status("❌ LLM returned invalid JSON.", done=True); return

                thought = str(action_json.get("thought", "")).strip()
                action = action_json.get("action", {})
                if thought:
                    yield self._status(f"🧠 Step {step}: {thought}")
                if not isinstance(action, dict) or "op" not in action:
                    yield self._status("❌ LLM action missing 'op'.", done=True); return

                op = str(action.get("op")).lower().strip()

                # ---- Execute tools ----
                if op == "done":
                    success = bool(action.get("success", True))
                    msg = action.get("message") or "Task complete."
                    yield self._status(("✅ " if success else "⚠️ ") + msg, done=True)
                    return

                elif op == "navigate":
                    url = action.get("url"); wait_until = action.get("wait_until", "load")
                    if not isinstance(url, str) or not url:
                        yield self._status("❌ navigate requires 'url'", done=True); return
                    yield self._status(f"🛠️ Action: navigate → {url} (wait_until={wait_until})")
                    if not self._do_navigate(http, url, wait_until):
                        yield self._status("❌ navigate failed.", done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "click":
                    selector = action.get("selector")
                    if not isinstance(selector, str) or not selector:
                        yield self._status("❌ click requires 'selector'", done=True); return
                    yield self._status(f"🛠️ Action: click → {selector}")
                    r = http.post(f"{MCP_BASE}/browser_click", json={"selector": selector})
                    if not self._ok(r): yield self._status(self._http_error("click", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "type":
                    selector = action.get("selector"); textval = action.get("text")
                    if not isinstance(selector, str) or not isinstance(textval, str):
                        yield self._status("❌ type requires 'selector' and 'text'", done=True); return
                    yield self._status(f"🛠️ Action: type → {selector} = {textval}")
                    r = http.post(f"{MCP_BASE}/browser_type", json={"selector": selector, "text": textval})
                    if not self._ok(r): yield self._status(self._http_error("type", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "press_key":
                    key = action.get("key")
                    if not isinstance(key, str) or not key:
                        yield self._status("❌ press_key requires 'key'", done=True); return
                    yield self._status(f"🛠️ Action: press_key → {key}")
                    r = http.post(f"{MCP_BASE}/browser_press_key", json={"key": key})
                    if not self._ok(r): yield self._status(self._http_error("press_key", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "hover":
                    selector = action.get("selector")
                    if not isinstance(selector, str) or not selector:
                        yield self._status("❌ hover requires 'selector'", done=True); return
                    yield self._status(f"🛠️ Action: hover → {selector}")
                    r = http.post(f"{MCP_BASE}/browser_hover", json={"selector": selector})
                    if not self._ok(r): yield self._status(self._http_error("hover", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "wait":
                    ms = action.get("ms", 1000)
                    try: ms = int(ms)
                    except Exception: ms = 1000
                    yield self._status(f"🛠️ Action: wait → {ms}ms")
                    r = http.post(f"{MCP_BASE}/browser_wait_for", json={"ms": ms})
                    if not self._ok(r): time.sleep(ms/1000.0)

                elif op == "screenshot":
                    # We'll still perform OBSERVE below
                    pass

                else:
                    yield self._status(f"ℹ️ Unknown op '{op}', stopping.", done=True)
                    return

                # ---- OBSERVE after each step: screenshot + URL/TITLE/TEXT ----
                last_image_b64, public_url = self._take_screenshot(http)
                if last_image_b64 or public_url:
                    yield self._image_event(last_image_b64, public_url)
                current_url, current_title, last_text_snippet = self._observe_text(http)
                obs_ev = self._observation_event(current_url, current_title, last_text_snippet)
                if obs_ev: yield obs_ev

                # Append to trace (keep small)
                trace.append({"thought": thought, "action": action})
                trace = trace[-6:]

                # Stuck detection
                self._remember_obs(obs_fps, current_url, current_title, last_text_snippet)
                if len(obs_fps) >= STUCK_WINDOW and len(set(obs_fps[-STUCK_WINDOW:])) == 1:
                    yield self._status(
                        "🤔 I might be stuck (page state not changing). "
                        "Please reply with a hint—e.g., a CSS selector, a button text, or exact steps.",
                        done=True,
                    )
                    return

            yield self._status("⚠️ Reached max steps without 'done'. Please provide more guidance.", done=True)

    # ---------- Helpers: user goal & creds ----------

    def _extract_user_goal(self, messages: List[dict]) -> Optional[str]:
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    texts = [b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"]
                    txt = "\n".join(t for t in texts if t)
                    if txt: return txt
        return None

    def _extract_credentials(self, messages: List[dict]) -> Tuple[str, str]:
        base = ""; key = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                text = None
                if isinstance(c, str):
                    text = c
                elif isinstance(c, list):
                    parts = [b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"]
                    text = "\n".join(p for p in parts if p)
                if not text: continue
                # JSON block
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        base = obj.get("openai_api_base", base)
                        key  = obj.get("openai_api_key", key)
                except Exception:
                    pass
                # KEY=VALUE patterns
                m1 = re.search(r"OPENAI_API_BASE\s*=\s*([^\s]+)", text)
                m2 = re.search(r"OPENAI_API_KEY\s*=\s*([^\s]+)", text)
                if m1: base = m1.group(1)
                if m2: key  = m2.group(1)
        return base or "", key or ""

    # ---------- LLM messages & parsing ----------

    def _build_messages(
        self, goal: str, trace: List[dict],
        image_b64: Optional[str],
        url: Optional[str], title: Optional[str], text_snippet: Optional[str],
    ) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        obs_lines = []
        if url:   obs_lines.append(f"URL: {url}")
        if title: obs_lines.append(f"TITLE: {title}")
        if text_snippet:
            obs_lines.append("VISIBLE_TEXT (truncated):")
            obs_lines.append(text_snippet[:6000])

        msgs.append({"role": "user", "content": f"GOAL:\n{goal}\nReturn ONLY the JSON per spec."})
        if trace:
            msgs.append({"role": "user", "content": "RECENT_STEPS:\n" + json.dumps(trace, ensure_ascii=False)})

        if image_b64:
            content: List[Dict[str, Any]] = []
            if obs_lines:
                content.append({"type": "text", "text": "OBSERVATION:\n" + "\n".join(obs_lines)})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
            msgs.append({"role": "user", "content": content})
        elif obs_lines:
            msgs.append({"role": "user", "content": "OBSERVATION:\n" + "\n".join(obs_lines)})

        return msgs

    def _extract_json(self, llm_resp: httpx.Response) -> Optional[Dict[str, Any]]:
        try:
            data = llm_resp.json()
            msg = data["choices"][0]["message"]["content"]
        except Exception:
            return None
        # Extract first JSON object in the content
        try:
            start = msg.index("{")
            depth = 0; end = start
            for i, ch in enumerate(msg[start:], start=start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: end = i+1; break
            return json.loads(msg[start:end])
        except Exception:
            try: return json.loads(msg)
            except Exception: return None

    # ---------- OBSERVE helpers ----------

    def _observe_text(self, http: httpx.Client) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (current_url, title, text_snippet). Try /browser_evaluate; fallback to /browser_snapshot."""
        js = "JSON.stringify({url: location.href, title: document.title, text: document.body ? document.body.innerText : ''})"
        payloads = [
            {"expression": js},
            {"script": js},
            {"code": js},
            {"js": js},
        ]
        for p in payloads:
            try:
                r = http.post(f"{MCP_BASE}/browser_evaluate", json=p)
                if self._ok(r):
                    obj = self._safe_json(r)
                    if isinstance(obj, str):
                        try: parsed = json.loads(obj)
                        except Exception: parsed = {}
                        url  = parsed.get("url"); tit = parsed.get("title"); text = parsed.get("text")
                    elif isinstance(obj, dict):
                        url  = obj.get("url"); tit = obj.get("title"); text = obj.get("text")
                    else:
                        url = tit = text = None
                    if isinstance(text, str) and text.strip():
                        return url, tit, text
                    if isinstance(url, str) or isinstance(tit, str):
                        return url, tit, None
            except Exception:
                pass

        # Fallback: /browser_snapshot (HTML)
        try:
            r = http.post(f"{MCP_BASE}/browser_snapshot", json={})
            if self._ok(r):
                obj = self._safe_json(r)
                if isinstance(obj, dict):
                    html = obj.get("html") or obj.get("content") or obj.get("data")
                    if isinstance(html, str) and html:
                        return None, None, self._html_to_text(html)
                elif isinstance(obj, str) and "<html" in obj.lower():
                    return None, None, self._html_to_text(obj)
                body = r.text or ""
                if "<html" in body.lower():
                    return None, None, self._html_to_text(body)
        except Exception:
            pass
        return None, None, None

    def _html_to_text(self, html: str) -> str:
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()[:12000]

    # ---------- Screenshot helpers ----------

    def _take_screenshot(self, http: httpx.Client, full: bool = True, path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        if not path:
            path = os.path.join(OUT_DIR, _now_png())
        r = http.post(f"{MCP_BASE}/browser_take_screenshot", json={"fullPage": full, "filename": path})
        if self._ok(r) and os.path.exists(path):
            public = f"{PUBLIC_BASE}/{os.path.basename(path)}"
            with open(path, "rb") as f:
                b64 = _b64(f.read())
            return b64, public
        # Fallback to base64 return
        r2 = http.post(f"{MCP_BASE}/browser_take_screenshot", json={"fullPage": full, "return": "base64"})
        if self._ok(r2):
            obj = self._safe_json(r2); b64 = None
            if isinstance(obj, dict):
                b64 = obj.get("data") or obj.get("base64")
            elif isinstance(obj, str) and len(obj) > 64:
                b64 = obj
            if isinstance(b64, str) and b64.startswith("data:image"):
                try: b64 = b64.split(",", 1)[1]
                except Exception: pass
            if b64:
                try:
                    with open(path, "wb") as f: f.write(base64.b64decode(b64))
                    public = f"{PUBLIC_BASE}/{os.path.basename(path)}"
                    return b64, public
                except Exception:
                    return b64, None
        return None, None

    # ---------- Events ----------

    def _observation_event(self, url: Optional[str], title: Optional[str], text: Optional[str]) -> Optional[Dict]:
        parts = []
        if url:   parts.append(f"URL: {url}")
        if title: parts.append(f"TITLE: {title}")
        if text:  parts.append(f"TEXT: {text[:500]}{'…' if text and len(text) > 500 else ''}")
        if not parts:
            return None
        return self._status("👀 Observation:\n" + "\n".join(parts))

    def _image_event(self, b64: Optional[str], url: Optional[str]) -> Dict:
        data: Dict[str, Any] = {"mime_type": "image/png"}
        if b64: data["base64"] = b64           # render inline
        if url: data["path"] = url             # clickable download link
        return {"event": {"type": "image", "data": data}}

    # ---------- MCP helpers & small utils ----------

    def _do_navigate(self, http: httpx.Client, url: str, wait_until: str = "load") -> bool:
        r = http.post(f"{MCP_BASE}/browser_navigate", json={"url": url, "wait_until": wait_until})
        return self._ok(r)

    def _remember_obs(self, fp_list: List[str], url: Optional[str], title: Optional[str], text: Optional[str]):
        h = hashlib.sha256()
        h.update((url or "").encode()); h.update((title or "").encode()); h.update((text or "").encode())
        fp_list.append(h.hexdigest())
        if len(fp_list) > STUCK_WINDOW: fp_list[:] = fp_list[-STUCK_WINDOW:]

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

    def _ok(self, resp: httpx.Response) -> bool:
        return 200 <= resp.status_code < 300

    def _http_error(self, label: str, resp: httpx.Response) -> str:
        body = ""
        try: body = resp.text[:400]
        except Exception: pass
        return f"{label}: HTTP {resp.status_code} {body}"

    # ---- status event ----
    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}
