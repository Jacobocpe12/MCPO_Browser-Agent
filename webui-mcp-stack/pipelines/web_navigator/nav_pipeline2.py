"""
ReAct Web Navigator — Playwright MCP + OpenWebUI
(OpenAI-compatible autodiscovery for /models & /chat/completions)

- Valves: model, openai_api_base, openai_api_key
- Auto-detects path prefix: "", "/v1", "/api", "/api/v1"
- Verifies model via /models when available
- ReAct loop: THINK -> ACT -> OBSERVE (screenshot + URL + TITLE + VISIBLE_TEXT)
- Inline image events (base64 + public URL)
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

try:
    from pydantic import BaseModel
except Exception:
    class BaseModel:
        def __init__(self, **kwargs): pass

# ---- MCP + public web server ----
MCP_BASE      = "http://91.99.79.208:3880/mcp_playwright"
PUBLIC_BASE   = "http://91.99.79.208:3888"
OUT_DIR       = "/tmp/playwright-output"

TIMEOUT       = 120.0
MAX_STEPS     = 16
SCREEN_DELAY  = 0.5
STUCK_WINDOW  = 3

SYSTEM_PROMPT = """You are a web-browsing agent controlling Playwright MCP via REST.
Follow a strict ReAct loop: THINK -> ACT -> OBSERVE (from latest screenshot and page text) until the user's goal is satisfied.

TOOLS (choose exactly one per step):
- navigate { "url": string, "wait_until": "load"|"domcontentloaded"|"networkidle" }
- click { "selector": string }
- type { "selector": string, "text": string }
- press_key { "key": string }
- hover { "selector": string }
- wait { "ms": number }
- screenshot { "fullPage": boolean, "filename"?: string }
- done { "success": boolean, "message": string }

RETURN FORMAT (valid JSON only):
{ "thought": "...", "action": { "op": "<tool>", "...": "..." } }

Guidelines:
- Include a short 'thought'.
- Handle cookie banners (e.g., “Akzeptieren”, “Alle akzeptieren”, “Accept all”).
- After significant actions, take a screenshot to re-orient.
- OBSERVATION includes: URL, TITLE, VISIBLE_TEXT (truncated), and a screenshot.
- Return op=done when goal is satisfied.
"""

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")

def _now_png() -> str:
    return f"page-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.png"

def _join(base: str, tail: str) -> str:
    return f"{base.rstrip('/')}/{tail.lstrip('/')}"

class Pipeline:
    class Valves(BaseModel):
        model: str = "gpt-4o-mini"    # set your vision model here
        openai_api_base: str = ""     # e.g. https://api.openai.com  (or your OpenWebUI proxy)
        openai_api_key: str = ""      # leave empty if your gateway doesn't require it

    def __init__(self):
        self.name = "ReAct Web Navigator (Playwright MCP, autodetect)"
        self.description = "ReAct loop with OBSERVE; autodetect OpenAI-compatible path; valves for model/base/key."
        self.version = "1.5.0"
        self.author = "You"
        self.valves = self.Valves()
        os.makedirs(OUT_DIR, exist_ok=True)

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        goal = self._extract_user_goal(messages) or user_message or ""
        if not goal.strip():
            yield self._status("❌ No goal provided.", done=True); return

        # Resolve creds: valves -> env -> user msg
        chat_api_base, chat_api_key = self._extract_credentials(messages)
        openai_base = (self.valves.openai_api_base or os.getenv("OPENAI_API_BASE", "") or chat_api_base).strip().rstrip("/")
        openai_key  = (self.valves.openai_api_key  or os.getenv("OPENAI_API_KEY", "")  or chat_api_key).strip()

        if not openai_base:
            yield self._status("❓ Set 'openai_api_base' in valves or env (OPENAI_API_BASE).", done=True); return

        # Resolve model: FORCE_MODEL > valve.model > chat model_id > PIPELINE_DEFAULT_MODEL
        chosen_model = (os.getenv("FORCE_MODEL") or self.valves.model or (model_id or "") or os.getenv("PIPELINE_DEFAULT_MODEL", "")).strip()
        if not chosen_model:
            yield self._status("❓ No model selected. Set valve 'model' or FORCE_MODEL env.", done=True); return

        headers = {"Content-Type": "application/json"}
        if openai_key:
            headers["Authorization"] = f"Bearer {openai_key}"

        # --- Detect correct API prefix (/, /v1, /api, /api/v1) using /models probe ---
        with httpx.Client(timeout=TIMEOUT, headers=headers) as probe:
            models_url, chat_url, prefix = self._detect_paths(probe, openai_base)
        yield self._status(f"🔧 Using model: {chosen_model} @ {openai_base}{prefix or ''}")

        # Optional model validation
        available = self._fetch_models(models_url, headers) if models_url else None
        if isinstance(available, list) and available:
            ids = {m.get("id") for m in available if isinstance(m, dict)}
            if chosen_model not in ids:
                hint = (", ".join(sorted(list(ids))[:8]) + (" ..." if len(ids) > 8 else "")) if ids else "none reported"
                yield self._status(f"⚠️ Model '{chosen_model}' not found on backend. Available: {hint}")

        yield self._status(f"🎯 Goal: {goal}")

        with httpx.Client(timeout=TIMEOUT) as http, httpx.Client(timeout=TIMEOUT, headers=headers) as llm:
            start_url = self._extract_url(goal)
            trace: List[Dict[str, Any]] = []
            last_image_b64 = None
            current_url = None; current_title = None; last_text_snippet = None
            obs_fps: List[str] = []

            # Optional initial navigate
            if start_url:
                yield self._status(f"🛠️ Boot: navigate → {start_url}")
                if not self._do_navigate(http, start_url):
                    yield self._status("❌ Initial navigate failed.", done=True); return
                time.sleep(SCREEN_DELAY)
                b64, pub = self._take_screenshot(http)
                if b64 or pub: yield self._image_event(b64, pub)
                last_image_b64 = b64
                current_url, current_title, last_text_snippet = self._observe_text(http)
                obs = self._observation_event(current_url, current_title, last_text_snippet)
                if obs: yield obs
                trace.append({"thought": "navigated to start url", "action": {"op": "navigate", "url": start_url}})
                self._remember_obs(obs_fps, current_url, current_title, last_text_snippet)

            for step in range(1, MAX_STEPS + 1):
                # LLM ReAct decision
                msgs = self._build_messages(goal, trace, last_image_b64, current_url, current_title, last_text_snippet)
                try:
                    resp = llm.post(chat_url, json={
                        "model": chosen_model,
                        "messages": msgs,
                        "temperature": 0.2,
                        "stream": False,
                    })
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = (e.response.text or "")[:700]
                    yield self._status(f"❌ LLM HTTP {e.response.status_code}: {body}", done=True); return
                except Exception as e:
                    yield self._status(f"❌ LLM error: {e}", done=True); return

                action_json = self._extract_json(resp)
                if not isinstance(action_json, dict):
                    yield self._status("❌ LLM returned invalid JSON.", done=True); return

                thought = str(action_json.get("thought", "")).strip()
                action  = action_json.get("action", {})
                if thought: yield self._status(f"🧠 Step {step}: {thought}")
                if not isinstance(action, dict) or "op" not in action:
                    yield self._status("❌ LLM action missing 'op'.", done=True); return

                op = str(action.get("op")).lower().strip()

                # Execute
                if op == "done":
                    success = bool(action.get("success", True))
                    msg = action.get("message") or "Task complete."
                    yield self._status(("✅ " if success else "⚠️ ") + msg, done=True); return

                elif op == "navigate":
                    url = action.get("url"); wait_until = action.get("wait_until", "load")
                    if not isinstance(url, str) or not url:
                        yield self._status("❌ navigate requires 'url'", done=True); return
                    yield self._status(f"🛠️ Action: navigate → {url} (wait_until={wait_until})")
                    if not self._do_navigate(http, url, wait_until):
                        yield self._status("❌ navigate failed.", done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "click":
                    sel = action.get("selector")
                    if not isinstance(sel, str) or not sel:
                        yield self._status("❌ click requires 'selector'", done=True); return
                    yield self._status(f"🛠️ Action: click → {sel}")
                    r = http.post(_join(MCP_BASE, "browser_click"), json={"selector": sel})
                    if not self._ok(r): yield self._status(self._http_error("click", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "type":
                    sel = action.get("selector"); txt = action.get("text")
                    if not isinstance(sel, str) or not isinstance(txt, str):
                        yield self._status("❌ type requires 'selector' and 'text'", done=True); return
                    yield self._status(f"🛠️ Action: type → {sel} = {txt}")
                    r = http.post(_join(MCP_BASE, "browser_type"), json={"selector": sel, "text": txt})
                    if not self._ok(r): yield self._status(self._http_error("type", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "press_key":
                    key = action.get("key")
                    if not isinstance(key, str) or not key:
                        yield self._status("❌ press_key requires 'key'", done=True); return
                    yield self._status(f"🛠️ Action: press_key → {key}")
                    r = http.post(_join(MCP_BASE, "browser_press_key"), json={"key": key})
                    if not self._ok(r): yield self._status(self._http_error("press_key", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "hover":
                    sel = action.get("selector")
                    if not isinstance(sel, str) or not sel:
                        yield self._status("❌ hover requires 'selector'", done=True); return
                    yield self._status(f"🛠️ Action: hover → {sel}")
                    r = http.post(_join(MCP_BASE, "browser_hover"), json={"selector": sel})
                    if not self._ok(r): yield self._status(self._http_error("hover", r), done=True); return
                    time.sleep(SCREEN_DELAY)

                elif op == "wait":
                    ms = action.get("ms", 1000)
                    try: ms = int(ms)
                    except Exception: ms = 1000
                    yield self._status(f"🛠️ Action: wait → {ms}ms")
                    r = http.post(_join(MCP_BASE, "browser_wait_for"), json={"ms": ms})
                    if not self._ok(r): time.sleep(ms/1000.0)

                elif op == "screenshot":
                    pass  # OBSERVE below handles it

                else:
                    yield self._status(f"ℹ️ Unknown op '{op}', stopping.", done=True); return

                # OBSERVE
                b64, pub = self._take_screenshot(http)
                if b64 or pub: yield self._image_event(b64, pub)
                last_image_b64 = b64

                current_url, current_title, last_text_snippet = self._observe_text(http)
                obs = self._observation_event(current_url, current_title, last_text_snippet)
                if obs: yield obs

                trace.append({"thought": thought, "action": action})
                trace = trace[-6:]

                self._remember_obs(obs_fps, current_url, current_title, last_text_snippet)
                if len(obs_fps) >= STUCK_WINDOW and len(set(obs_fps[-STUCK_WINDOW:])) == 1:
                    yield self._status(
                        "🤔 I might be stuck (page state not changing). "
                        "Please provide a hint (CSS selector, button text, or exact step).",
                        done=True,
                    ); return

            yield self._status("⚠️ Reached max steps without 'done'. Please provide more guidance.", done=True)

    # ----- autodetect OpenAI-compatible endpoints -----
    def _detect_paths(self, client: httpx.Client, base: str) -> Tuple[Optional[str], str, str]:
        """
        Returns (models_url, chat_url, prefix_used)
        Tries '', '/v1', '/api', '/api/v1' for /models. If none work, assume '/v1'.
        """
        prefixes = ["", "/v1", "/api", "/api/v1"]
        for pref in prefixes:
            url = _join(base, _join(pref, "models"))
            try:
                r = client.get(url)
                if 200 <= r.status_code < 300:
                    return url, _join(base, _join(pref, "chat/completions")), pref
            except Exception:
                pass
        # Some servers don't expose /models; fall back to /v1
        return None, _join(base, "v1/chat/completions"), "/v1"

    def _fetch_models(self, models_url: str, headers: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
        try:
            r = httpx.get(models_url, headers=headers, timeout=TIMEOUT)
            if 200 <= r.status_code < 300:
                data = r.json()
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    return data["data"]
        except Exception:
            pass
        return None

    # ----- goal & creds -----
    def _extract_user_goal(self, messages: List[dict]) -> Optional[str]:
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str): return c
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
                text = c if isinstance(c, str) else "\n".join([b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"]) if isinstance(c, list) else None
                if not text: continue
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        base = obj.get("openai_api_base", base)
                        key  = obj.get("openai_api_key", key)
                except Exception:
                    pass
                m1 = re.search(r"OPENAI_API_BASE\s*=\s*([^\s]+)", text or "")
                m2 = re.search(r"OPENAI_API_KEY\s*=\s*([^\s]+)", text or "")
                if m1: base = m1.group(1)
                if m2: key  = m2.group(1)
        return base or "", key or ""

    # ----- messages & parsing -----
    def _build_messages(self, goal: str, trace: List[dict], image_b64: Optional[str],
                        url: Optional[str], title: Optional[str], text_snippet: Optional[str]) -> List[Dict[str, Any]]:
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

    # ----- OBSERVE -----
    def _observe_text(self, http: httpx.Client) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        js = "JSON.stringify({url: location.href, title: document.title, text: document.body ? document.body.innerText : ''})"
        for p in ({"expression": js}, {"script": js}, {"code": js}, {"js": js}):
            try:
                r = http.post(_join(MCP_BASE, "browser_evaluate"), json=p)
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
            r = http.post(_join(MCP_BASE, "browser_snapshot"), json={})
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

    # ----- Screenshots (robust) -----
    def _take_screenshot(self, http: httpx.Client, full: bool = True, path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        if not path:
            path = os.path.join(OUT_DIR, _now_png())

        r = http.post(_join(MCP_BASE, "browser_take_screenshot"), json={"fullPage": full, "filename": path})
        if self._ok(r) and os.path.exists(path):
            public = _join(PUBLIC_BASE, os.path.basename(path))
            with open(path, "rb") as f: b64 = _b64(f.read())
            return b64, public

        body_text = ""
        try: body_text = r.text or ""
        except Exception: pass
        m = re.search(r"saved it as\s+(/[^'\s]+\.png)", body_text, flags=re.IGNORECASE)
        if m:
            server_path = m.group(1)
            for candidate in (server_path, os.path.join(OUT_DIR, os.path.basename(server_path))):
                if os.path.exists(candidate):
                    public = _join(PUBLIC_BASE, os.path.basename(candidate))
                    with open(candidate, "rb") as f: b64 = _b64(f.read())
                    return b64, public

        r2 = http.post(_join(MCP_BASE, "browser_take_screenshot"), json={"fullPage": full, "return": "base64"})
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
                    return b64, public
                except Exception:
                    return b64, None

        return None, None

    # ----- events & utils -----
    def _observation_event(self, url: Optional[str], title: Optional[str], text: Optional[str]) -> Optional[Dict]:
        parts = []
        if url: parts.append(f"URL: {url}")
        if title: parts.append(f"TITLE: {title}")
        if text: parts.append(f"TEXT: {text[:500]}{'…' if text and len(text) > 500 else ''}")
        if not parts: return None
        return self._status("👀 Observation:\n" + "\n".join(parts))

    def _image_event(self, b64: Optional[str], url: Optional[str]) -> Dict:
        data: Dict[str, Any] = {"mime_type": "image/png"}
        if b64: data["base64"] = b64
        if url: data["path"] = url
        return {"event": {"type": "image", "data": data}}

    def _do_navigate(self, http: httpx.Client, url: str, wait_until: str = "load") -> bool:
        r = http.post(_join(MCP_BASE, "browser_navigate"), json={"url": url, "wait_until": wait_until})
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

    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}
