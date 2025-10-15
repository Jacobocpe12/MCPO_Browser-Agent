"""
title: Browser MOA (Planner + Visual + Verifier)
author: You
version: 0.3.0
description: Multi-agent browser pipeline with Planner + Visual + Verifier, runtime-editable settings, rich logs, model fallbacks, and URL bootstrap.
"""

import os, time, json, asyncio, httpx, yaml, re, traceback, logging, threading, pathlib
from typing import Optional, Dict, Any, List, Tuple, AsyncGenerator
from urllib.parse import urlparse

# =============================== UI Event helpers =================================

def ev_status(desc: str, done: bool=False) -> Dict[str, Any]:
    return {"event": {"type": "status", "data": {"description": desc, "done": done}}}

def ev_msg_md(md: str) -> Dict[str, Any]:
    return {"event": {"type": "message", "data": {"role": "assistant", "content": md, "content_type": "text/markdown"}}}

def ev_log_block(lines: List[str]) -> Dict[str, Any]:
    text = "\n".join(lines) if lines else "(no logs)"
    return ev_msg_md("**Debug log**\n\n```log\n" + text + "\n```")

# ================================ Defaults / Schema ================================

# 🔧 Your baked-in defaults (editable at runtime)
DEFAULTS = {
    "openai_base":      "https://ollama.gpu.lfi.rwth-aachen.de/api",
    "openai_api_key":   "sk-b260ccfd2e994dec9e0575124073ddc8",
    "model_visual":     "gpt-4o-mini",
    "model_planner":    "azure.gpt-4o-sweden",  # ← you asked for this
    "model_verifier":   "gpt-4o-mini",
    "model_summary":    "gpt-4o-mini",
    "playwright_base":  "http://91.99.79.208:3880/mcp_playwright",
    "screenshot_base":  "http://91.99.79.208:3888",
    "img_dir":          "/tmp/playwright-output",
    "max_tool_retries": 5,
    "step_limit":       15,
    "log_level":        "DEBUG",
    "debug":            1,

    # Optional: model fallbacks to try if the server returns "model not found"
    "model_fallbacks":  ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"],

    # Optional: force a first navigation if goal contains a URL/domain
    "bootstrap_if_url": True,
}

PERSIST_PATHS = [
    os.path.expanduser("~/.openwebui/browser_moa_pipeline.json"),
    os.path.join(os.getcwd(), "browser_moa_pipeline.json"),
]

# Chat completions paths to try (server quirks differ)
CANDIDATE_CHAT_URLS = [
    "/v1/chat/completions",
    "/chat/completions",
    "/api/v1/chat/completions",
    "/api/chat/completions",
]

SYS_VERIFIER = """You are the gatekeeper for a web-browsing agent.
Given a user's message, decide if it requires using the BROWSER PIPELINE.

If the task involves visiting or operating real websites, clicking, scrolling, or visual inspection, return `use_pipeline: true`.

Also, define a structured goal if possible.

Return JSON in this exact format:

{
 "use_pipeline": true|false,
 "goal": "<normalized concise goal or null>",
 "intent": "<search|form_fill|scraping|navigation|analysis|unknown>",
 "targets": ["optional list of domains or keywords"]
}

Never include extra text or comments."""
SYS_PLANNER = """You are the Planner. You receive: GOAL and OBSERVATION.
Available tools: browser_install, browser_navigate, browser_wait_for, browser_snapshot,
browser_click, browser_type, browser_press_key, browser_drag, browser_hover, browser_select_option,
browser_take_screenshot, browser_evaluate, browser_resize, browser_navigate_back, browser_close,
browser_pdf_save, browser_console_messages, browser_network_requests, browser_mouse_move_xy,
browser_mouse_click_xy, browser_mouse_drag_xy, browser_file_upload, browser_fill_form.
Respond ONLY JSON for the next action:
{"op":"navigate","url":"..."}
{"op":"click","ref":"e12"}        # or {"selector":"..."}
{"op":"type","ref":"e7","text":"...","submit":true}
{"op":"press","key":"Enter"}
{"op":"wait","ms":1200}
{"op":"screenshot"}
{"op":"done","reason":"..."}
Reason in your head, output only JSON."""
SYS_VISUAL = """You analyze a screenshot URL plus a YAML snapshot. Return strict JSON:
{
 "view":"map|list|form|article|unknown",
 "center":"<text or null>",
 "zoom": "<int or null>",
 "notable_elements":[{"ref":"e12","role":"button","text":"Login"}],
 "obstacles":[ "cookie_banner" ],
 "next_click_hint": {"ref":"e12"}
}
No prose. Use nulls when unknown."""
SYS_SUMMARY = """Write a concise, user-friendly summary of what the browsing agent did,
what it found, and the final outcome. Include key URLs if helpful. Keep it short and clear."""

# ============================== Utilities / Helpers ================================

def _redact(s: Optional[str]) -> str:
    if not s: return ""
    if len(s) <= 8: return "****"
    return s[:4] + "…" + s[-4:]

def _canon_base(url: str) -> str:
    url = (url or "").strip()
    return url.rstrip("/")

def _safe_snip(s: str, n: int = 300) -> str:
    try:
        return (s or "")[:n].replace("\n", " ")
    except Exception:
        return "(unprintable)"

def _strip_api_suffix(base: str) -> str:
    # If someone sets base like ".../api", also try the parent base
    if base.endswith("/api"): return base[:-4]  # drop "/api"
    return base

def _extract_first_url_or_domain(text: str) -> Optional[str]:
    if not text: return None
    # Quick URL or domain finder
    url_match = re.search(r"(https?://[^\s]+)", text, flags=re.I)
    if url_match: return url_match.group(1)
    dom_match = re.search(r"\b([a-z0-9\-]+\.[a-z]{2,})(/[^\s]*)?\b", text, flags=re.I)
    if dom_match:
        dom = dom_match.group(0)
        # common cases: google.com (no scheme)
        if not dom.lower().startswith("http"):
            return "https://" + dom
        return dom
    return None

# =============================== In-UI Log Buffer =================================

class UILogger:
    def __init__(self, flush_every: int = 10, to_stdout: bool = False, level: str = "INFO"):
        self._lines: List[str] = []
        self._idx_flushed = 0
        self._lock = threading.Lock()
        self._flush_every = max(flush_every, 1)
        self._to_stdout = to_stdout
        self.level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(level=self.level)

    def add(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self._lines.append(line)
            if self._to_stdout:
                print(line, flush=True)

    def add_kv(self, key: str, val: Any):
        try:
            s = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
        except Exception:
            s = str(val)
        self.add(f"{key} = {s}")

    def need_flush(self) -> bool:
        with self._lock:
            return len(self._lines) - self._idx_flushed >= self._flush_every

    def flush_events(self, force: bool=False) -> List[Dict[str, Any]]:
        with self._lock:
            if not force and not self.need_flush():
                return []
            chunk = self._lines[self._idx_flushed:]
            self._idx_flushed = len(self._lines)
        return [ev_log_block(chunk)] if chunk else []

    def all_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [ev_log_block(self._lines[:])]

# =============================== Config Management =================================

class ConfigManager:
    def __init__(self, logger: UILogger):
        self._cfg = DEFAULTS.copy()
        self.log = logger
        self._load_persisted()

    @property
    def cfg(self) -> Dict[str, Any]:
        return self._cfg

    def _load_persisted(self):
        for p in PERSIST_PATHS:
            try:
                if os.path.isfile(p):
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._cfg.update({k: data[k] for k in self._cfg.keys() if k in data})
                        self.log.add(f"loaded persisted config from {p}")
                        break
            except Exception as e:
                self.log.add(f"persisted config load error from {p}: {e}")

    def save(self):
        last_err = None
        for p in PERSIST_PATHS:
            try:
                pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(self._cfg, f, indent=2)
                self.log.add(f"config saved to {p}")
                return
            except Exception as e:
                last_err = e
                self.log.add(f"persist write error on {p}: {e}")
        if last_err:
            raise last_err

    def merge_ui(self, body: Dict[str, Any], user_text: str):
        # Merge from multiple possible body shapes
        def _dig(*keys):
            c = body or {}
            for k in keys:
                c = c.get(k, {}) if isinstance(c, dict) else {}
            return c if isinstance(c, dict) else {}

        candidates = [
            _dig("pipeline", "config"),
            body.get("config", {}),
            body.get("kwargs", {}),
            body.get("settings", {}),
            body.get("params", {}),
            body.get("variables", {}),
        ]
        merged_from_body = {}
        for d in candidates:
            for k, v in (d or {}).items():
                if k in self._cfg and v not in (None, ""):
                    merged_from_body[k] = v
        if merged_from_body:
            self._cfg.update(self._coerce_types(merged_from_body))
            self.log.add_kv("config.merge_from_ui_body", self._redacted_subset(merged_from_body))

        # Chat-side: !set key=value
        if user_text and "!set" in user_text:
            for line in user_text.splitlines():
                if line.strip().startswith("!set "):
                    try:
                        kv = line.strip()[5:]
                        k, v = kv.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k in self._cfg:
                            self._cfg[k] = self._coerce_type(k, v)
                            self.log.add(f"config.set {k} ← {self._redact_val(k, v)}")
                    except Exception as e:
                        self.log.add(f"set parse error: {e}")

        # Chat-side: !config {json}
        if user_text and "!config" in user_text:
            try:
                i = user_text.index("!config")
                rest = user_text[i+7:].strip()
                if rest.startswith("{"):
                    j = rest.rfind("}")
                    if j > 0:
                        obj = json.loads(rest[:j+1])
                        changed = {k: obj[k] for k in obj.keys() if k in self._cfg}
                        if changed:
                            self._cfg.update(self._coerce_types(changed))
                            self.log.add_kv("config.merge_from_chat_json", self._redacted_subset(changed))
            except Exception as e:
                self.log.add(f"!config parse error: {e}")

        # Canonicalize
        self._cfg["openai_base"] = _canon_base(self._cfg.get("openai_base", ""))
        self._cfg["playwright_base"] = _canon_base(self._cfg.get("playwright_base", ""))
        self._cfg["screenshot_base"] = _canon_base(self._cfg.get("screenshot_base", ""))

    def _coerce_types(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            out[k] = self._coerce_type(k, v)
        return out

    def _coerce_type(self, k: str, v: Any) -> Any:
        if k in ("max_tool_retries", "step_limit", "debug"):
            try: return int(v)
            except: return v
        if k == "model_fallbacks" and isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    def _redacted_subset(self, d: Dict[str, Any]) -> Dict[str, Any]:
        red = dict(d)
        if "openai_api_key" in red:
            red["openai_api_key"] = _redact(str(red["openai_api_key"]))
        return red

    def _redact_val(self, k: str, v: Any) -> str:
        return _redact(v) if "key" in k else str(v)

    def pretty(self) -> str:
        show = dict(self._cfg)
        show["openai_api_key"] = _redact(show.get("openai_api_key"))
        return json.dumps(show, indent=2)

# ================================ Browser Tool ===================================

class UILogger:
    def __init__(self, flush_every: int = 10, to_stdout: bool = False, level: str = "INFO"):
        self._lines: List[str] = []
        self._idx_flushed = 0
        self._lock = threading.Lock()
        self._flush_every = max(flush_every, 1)
        self._to_stdout = to_stdout
        self.level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(level=self.level)

    def add(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self._lines.append(line)
            if self._to_stdout:
                print(line, flush=True)

    def add_kv(self, key: str, val: Any):
        try:
            s = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
        except Exception:
            s = str(val)
        self.add(f"{key} = {s}")

    def need_flush(self) -> bool:
        with self._lock:
            return len(self._lines) - self._idx_flushed >= self._flush_every

    def flush_events(self, force: bool=False) -> List[Dict[str, Any]]:
        with self._lock:
            if not force and not self.need_flush():
                return []
            chunk = self._lines[self._idx_flushed:]
            self._idx_flushed = len(self._lines)
        return [ev_log_block(chunk)] if chunk else []

    def all_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [ev_log_block(self._lines[:])]

class BrowserTool:
    def __init__(self, base: str, img_dir: str, screenshot_base: str, retries: int, logger: UILogger):
        self.base = base.rstrip("/")
        self.img_dir = img_dir
        self.screenshot_base = screenshot_base.rstrip("/")
        self.retries = retries
        self.log = logger

    def _url(self, ep: str) -> str:
        return f"{self.base}/{ep.lstrip('/')}"

    def _retry(self):
        for i in range(self.retries):
            yield i, min(2 ** i * 0.2, 2.0)

    def _post(self, ep: str, json_body: Optional[Dict]=None) -> httpx.Response:
        url = self._url(ep)
        body = json_body or {}
        safe_body = body if "text" not in body else {**body, "text": "(redacted)"}
        self.log.add_kv(f"HTTP POST {url}", safe_body)
        with httpx.Client(timeout=httpx.Timeout(60, read=60, connect=30)) as http:
            r = http.post(url, json=body)
            self.log.add(f"→ {url} status={r.status_code}")
            if r.text:
                self.log.add(f"← body_snippet={_safe_snip(r.text)}")
            return r

    def install(self) -> Tuple[bool, str]:
        for i, sleep_s in self._retry():
            try:
                self.log.add(f"install attempt {i+1}")
                r = self._post("/browser_install", {})
                if r.status_code < 400:
                    return True, "installed"
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"install error: {e}")
                time.sleep(sleep_s)
        return False, "install_failed"

    def navigate(self, url: str) -> Tuple[bool, str]:
        payload = {"url": url}
        for i, sleep_s in self._retry():
            try:
                self.log.add(f"navigate attempt {i+1} → {url}")
                r = self._post("/browser_navigate", payload)
                if r.status_code < 400: return True, "navigated"
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"navigate error: {e}")
                time.sleep(sleep_s)
        return False, f"navigate_failed:{url}"

    def wait_for(self, selector_or_ref: str=None, timeout_ms: int=3000) -> Tuple[bool,str]:
        payload = {}
        if selector_or_ref:
            if selector_or_ref.startswith("e"): payload["ref"] = selector_or_ref
            else: payload["selector"] = selector_or_ref
        payload["timeout"] = timeout_ms
        for i, sleep_s in self._retry():
            try:
                self.log.add_kv("wait_for", payload)
                r = self._post("/browser_wait_for", payload)
                if r.status_code < 400: return True, "ready"
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"wait_for error: {e}")
                time.sleep(sleep_s)
        return False, "wait_failed"

    def snapshot(self) -> Tuple[bool, str]:
        for i, sleep_s in self._retry():
            try:
                self.log.add(f"snapshot attempt {i+1}")
                r = self._post("/browser_snapshot", {})
                if r.status_code < 400:
                    data = r.json()
                    result = data.get("result", data)
                    if isinstance(result, str) and (result.startswith("{") or result.startswith("[")):
                        try:
                            result = yaml.dump(json.loads(result), sort_keys=False, allow_unicode=True)
                        except Exception as e:
                            self.log.add(f"snapshot yaml dump error: {e}")
                    return True, result if isinstance(result, str) else str(result)
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"snapshot error: {e}")
                time.sleep(sleep_s)
        return False, "snapshot_failed"

    def click(self, ref: Optional[str]=None, selector: Optional[str]=None) -> Tuple[bool,str]:
        payload = {}
        if ref: payload["ref"] = ref
        elif selector: payload["selector"] = selector
        else: return False, "click_missing_target"
        for i, sleep_s in self._retry():
            try:
                self.log.add_kv("click", payload)
                r = self._post("/browser_click", payload)
                if r.status_code < 400: return True, "clicked"
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"click error: {e}")
                time.sleep(sleep_s)
        return False, "click_failed"

    def type(self, text: str, ref: Optional[str]=None, selector: Optional[str]=None, submit: bool=False) -> Tuple[bool,str]:
        payload = {"text": text, "submit": bool(submit)}
        if ref: payload["ref"] = ref
        elif selector: payload["selector"] = selector
        else: return False, "type_missing_target"
        for i, sleep_s in self._retry():
            try:
                safe_payload = {**payload, "text": "(redacted)"}
                self.log.add_kv("type", safe_payload)
                r = self._post("/browser_type", payload)
                if r.status_code < 400: return True, "typed"
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"type error: {e}")
                time.sleep(sleep_s)
        return False, "type_failed"

    def press(self, key: str="Enter") -> Tuple[bool,str]:
        for i, sleep_s in self._retry():
            try:
                self.log.add(f"press key={key} attempt {i+1}")
                r = self._post("/browser_press_key", {"key": key})
                if r.status_code < 400: return True, "pressed"
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"press error: {e}")
                time.sleep(sleep_s)
        return False, "press_failed"

    def screenshot(self, descriptive: str) -> Tuple[bool, str]:
        fn = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", descriptive) + ".png"
        path = f"{self.img_dir}/{fn}"
        payload = {"fullPage": True, "filename": path}
        for i, sleep_s in self._retry():
            try:
                self.log.add_kv("screenshot", payload)
                r = self._post("/browser_take_screenshot", payload)
                if r.status_code < 400:
                    url = f"{self.screenshot_base}/{fn}?ts={int(time.time()*1000)}"
                    self.log.add(f"screenshot saved → {url}")
                    return True, url
                time.sleep(sleep_s)
            except Exception as e:
                self.log.add(f"screenshot error: {e}")
                time.sleep(sleep_s)
        return False, "screenshot_failed"

# ================================ LLM Wrapper =====================================

def _candidate_urls_for_base(base: str) -> List[str]:
    base = _canon_base(base)
    strip_base = _strip_api_suffix(base)
    urls = [base + path for path in CANDIDATE_CHAT_URLS]
    if strip_base != base:
        # also try without trailing /api
        urls += [strip_base + path for path in CANDIDATE_CHAT_URLS]
    # de-duplicate while preserving order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

async def chat_complete(messages: List[Dict], model: str, base: str, api_key: str,
                        logger: UILogger, expect_json: bool=False,
                        fallbacks: Optional[List[str]] = None) -> str:
    """
    Tries multiple endpoint paths and (optionally) model fallbacks if the server complains
    about an unknown model.
    """
    payload_base = {"messages": messages, "stream": False}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    candidates = _candidate_urls_for_base(base)
    try_models = [model] + [m for m in (fallbacks or []) if m != model]

    logger.add_kv("chat_complete.request", {
        "models": try_models, "candidates": candidates, "msg_count": len(messages)
    })

    async with httpx.AsyncClient(timeout=httpx.Timeout(90, read=90, connect=30)) as http:
        last_error = None
        for model_name in try_models:
            payload = dict(payload_base, model=model_name)
            for url in candidates:
                try:
                    logger.add(f"POST {url} (model={model_name})")
                    r = await http.post(url, json=payload, headers=headers)
                    logger.add(f"→ status={r.status_code}")
                    txt = r.text or ""
                    if r.status_code >= 400:
                        logger.add(f"← err_snip={_safe_snip(txt)}")
                        # If model is not available, try the next model
                        if "model" in txt.lower() and "not" in txt.lower():
                            logger.add("suspect model-not-found; trying next model if available")
                            break  # try next model_name
                        continue  # try next url path with same model
                    data = r.json()
                    out = data["choices"][0]["message"]["content"]
                    logger.add(f"chat_complete.ok snippet={_safe_snip(out)}")
                    if expect_json:
                        s, e = out.find("{"), out.rfind("}")
                        if s != -1 and e != -1 and e > s:
                            j = out[s:e+1]
                            logger.add(f"chat_complete.json_snip={_safe_snip(j)}")
                            return j
                        logger.add("chat_complete.expect_json but no braces found; returning raw content")
                    return out
                except Exception as e:
                    last_error = e
                    logger.add(f"chat_complete error on {url} (model={model_name}): {e}")
                    continue
        raise RuntimeError(f"chat_complete failed: last_error={last_error}")

# ================================== Pipeline ======================================

class Pipeline:
    def __init__(self):
        self.log = UILogger(flush_every=5, to_stdout=False, level=DEFAULTS["log_level"])
        self.cfgm = ConfigManager(self.log)
        self.session: Dict[str, Any] = {
            "goal": None,
            "last_snapshot": None,
            "step_log": [],
            "visual_last": None
        }
        self.browser: Optional[BrowserTool] = None

    async def pipe(self, user_message: str, model_id: Optional[str], messages: List[dict], body: dict) -> AsyncGenerator[Dict[str, Any], None]:
        yield ev_status("🚀 Browser MOA pipeline starting…", done=False)
        self.cfgm.merge_ui(body or {}, user_message or "")
        cfg = self.cfgm.cfg

        # (re)init logger with current level
        self.log = UILogger(flush_every=5, to_stdout=False, level=cfg.get("log_level", "INFO"))
        self.cfgm.log = self.log

        self.browser = BrowserTool(
            base=cfg["playwright_base"],
            img_dir=cfg["img_dir"],
            screenshot_base=cfg["screenshot_base"],
            retries=int(cfg["max_tool_retries"]),
            logger=self.log,
        )

        self.log.add_kv("config.active", {
            "openai_base": cfg["openai_base"],
            "openai_api_key": _redact(cfg["openai_api_key"]),
            "models": {
                "planner": cfg["model_planner"],
                "verifier": cfg["model_verifier"],
                "visual": cfg["model_visual"],
                "summary": cfg["model_summary"],
            },
            "playwright_base": cfg["playwright_base"],
            "screenshot_base": cfg["screenshot_base"],
        })
        for ev in self.log.flush_events(force=True): yield ev

        # Quick commands
        if user_message and user_message.strip().lower() in ("!save", "!persist"):
            try:
                self.cfgm.save()
                yield ev_msg_md("✅ **Settings saved.**")
            except Exception as e:
                yield ev_msg_md(f"❌ **Save failed:** `{e}`")
            for ev in self.log.flush_events(force=True): yield ev
            yield ev_status("✅ Finished.", done=True)
            return

        if user_message and user_message.strip().lower() in ("!show", "!config"):
            yield ev_msg_md("**Current settings**\n\n```json\n" + self.cfgm.pretty() + "\n```")
            for ev in self.log.flush_events(force=True): yield ev
            yield ev_status("✅ Finished.", done=True)
            return

        try:
            # Connectivity probe for chat backend
            try:
                _ = await chat_complete(
                    messages=[{"role": "user", "content": "ping"}],
                    model=cfg["model_verifier"],
                    base=cfg["openai_base"],
                    api_key=cfg["openai_api_key"],
                    logger=self.log,
                    expect_json=False,
                    fallbacks=cfg.get("model_fallbacks", []),
                )
                self.log.add("chat backend probe ok")
            except Exception as e:
                self.log.add(f"chat backend probe failed: {e}")
                yield ev_msg_md(
                    "❌ **Chat backend not reachable or misconfigured.**\n\n"
                    f"- Base: `{cfg['openai_base']}`\n"
                    f"- Key: `{_redact(cfg['openai_api_key'])}`\n\n"
                    "Fix with `!set openai_base=...` and `!set openai_api_key=...`, then `!save` (optional) and rerun."
                )
                for ev in self.log.all_events(): yield ev
                yield ev_status("✅ Finished.", done=True)
                return

            # 1) Gate: should we run the pipeline?
            raw_goal = self._extract_goal(messages) or (user_message or "")
            self.log.add_kv("raw_goal", raw_goal)
            yield ev_status("📨 Reading messages…", done=False)
            for ev in self.log.flush_events(): yield ev

            if not raw_goal.strip():
                self.log.add("empty_goal")
                for ev in self.log.flush_events(force=True): yield ev
                yield ev_status("❌ Empty goal received — stopping.", done=True)
                return

            # Verifier
            try:
                decision_json = await chat_complete(
                    [{"role": "system", "content": SYS_VERIFIER},
                     {"role": "user", "content": raw_goal}],
                    cfg["model_verifier"], base=cfg["openai_base"], api_key=cfg["openai_api_key"],
                    logger=self.log, expect_json=True, fallbacks=cfg.get("model_fallbacks", [])
                )
                decision = json.loads(decision_json)
                self.log.add_kv("verifier.decision", decision)
            except Exception as e:
                self.log.add(f"verifier error: {e} — defaulting to use_pipeline=True")
                decision = {"use_pipeline": True, "goal": raw_goal, "intent": "unknown", "targets": []}

            use_pipeline = bool(decision.get("use_pipeline", False))
            goal        = decision.get("goal") or raw_goal

            yield ev_msg_md(f"🤖 **Verifier Decision:**\n\n```json\n{json.dumps(decision, indent=2)}\n```")
            for ev in self.log.flush_events(): yield ev

            if not use_pipeline:
                yield ev_status("💬 Verifier: pipeline not required.", done=False)
                yield ev_msg_md("This query doesn't require browser actions.")
                for ev in self.log.flush_events(force=True): yield ev
                yield ev_status("✅ Finished.", done=True)
                return

            self.session["goal"] = goal
            yield ev_status(f"🧭 Pipeline activated for goal: {goal}", done=False)

            # 2) Install/start browser
            yield ev_status("🧩 Installing/starting browser…", done=False)
            ok, msg = await asyncio.to_thread(self.browser.install)
            self.log.add_kv("install.result", {"ok": ok, "msg": msg})
            yield ev_status(f"🧩 {msg}", done=False)
            for ev in self.log.flush_events(): yield ev
            if not ok:
                yield ev_msg_md("❌ Browser install failed. Check MCP Playwright server address and availability.")
                for ev in self.log.all_events(): yield ev
                yield ev_status("✅ Finished.", done=True)
                return

            # 3) **Bootstrap navigation** if goal contains URL/domain
            if cfg.get("bootstrap_if_url", True):
                url_guess = _extract_first_url_or_domain(goal)
                if url_guess:
                    self.log.add(f"bootstrap.navigate {url_guess}")
                    yield ev_status(f"🌐 Navigating (bootstrap) → {url_guess}", done=False)
                    nav_ok, nav_msg = await asyncio.to_thread(self.browser.navigate, url_guess)
                    self.log.add_kv("bootstrap.navigate.result", {"ok": nav_ok, "msg": nav_msg})
                    if not nav_ok:
                        yield ev_msg_md(f"⚠️ Bootstrap navigate failed: `{nav_msg}`")
                    # snapshot/screenshot after bootstrap
                    snap_ok, snapshot = await asyncio.to_thread(self.browser.snapshot)
                    if snap_ok:
                        self.session["last_snapshot"] = snapshot
                        yield ev_msg_md("📖 **Snapshot (YAML, clipped)**\n\n```\n" + self._clip(snapshot, 1200) + "\n```")
                    shot_ok, url_or_err = await asyncio.to_thread(self.browser.screenshot, f"bootstrap_{int(time.time())}")
                    if shot_ok:
                        yield ev_msg_md(f"📸 **View (bootstrap):**\n\n![frame]({url_or_err})")
                    for ev in self.log.flush_events(): yield ev

            # 4) Planner loop
            for step in range(1, int(cfg["step_limit"]) + 1):
                self.log.add(f"loop.step {step} begin")
                obs_text = self._observation_text()
                self.log.add_kv("planner.obs_snip", obs_text[:400])

                try:
                    plan_json = await chat_complete(
                        [{"role":"system","content":SYS_PLANNER},
                         {"role":"user","content":f"GOAL:\n{goal}\n\nOBSERVATION:\n{obs_text}"}],
                        cfg["model_planner"], base=cfg["openai_base"], api_key=cfg["openai_api_key"],
                        logger=self.log, expect_json=True, fallbacks=cfg.get("model_fallbacks", [])
                    )
                except Exception as e:
                    self.log.add(f"planner request failed: {e}")
                    yield ev_msg_md(f"⚠️ Planner request failed: `{e}`. Stopping.")
                    for ev in self.log.all_events(): yield ev
                    break

                act = self._parse_json(plan_json)
                self.log.add_kv("planner.act", act)
                for ev in self.log.flush_events(): yield ev

                if not act:
                    yield ev_status("ℹ️ Planner returned no structured action; stopping.", done=False)
                    break

                if act.get("op") in ("done","finish","stop"):
                    reason = act.get("reason","done")
                    self.log.add(f"planner done: {reason}")
                    yield ev_status(f"✅ Done: {reason}", done=False)
                    break

                nice = self._nice_action(act)
                self.session["step_log"].append(nice)
                yield ev_status(f"🛠️ {nice}", done=False)

                ok, exec_msg = await asyncio.to_thread(self._exec, act)
                self.log.add_kv("exec.result", {"ok": ok, "msg": exec_msg})
                if not ok:
                    self.session["step_log"].append(f"⚠️ {exec_msg}")
                    yield ev_status(f"⚠️ {exec_msg} (retrying may occur)", done=False)

                # Snapshot + Screenshot
                snap_ok, snapshot = await asyncio.to_thread(self.browser.snapshot)
                if snap_ok:
                    self.session["last_snapshot"] = snapshot
                    yield ev_msg_md("📖 **Snapshot (YAML, clipped)**\n\n```\n" + self._clip(snapshot, 1200) + "\n```")

                shot_ok, url_or_err = await asyncio.to_thread(self.browser.screenshot, f"step_{step}_{int(time.time())}")
                if shot_ok:
                    yield ev_msg_md(f"📸 **View after step {step}:**\n\n![frame]({url_or_err})")
                else:
                    yield ev_status("⚠️ Screenshot failed", done=False)

                # Keep logs flowing
                for ev in self.log.flush_events(): yield ev
                yield ev_status(f"⏳ heartbeat after step {step}", done=False)

            # 5) Final summary
            yield ev_status("📦 Summarizing…", done=False)
            try:
                final_summary = await chat_complete(
                    [{"role":"system","content":SYS_SUMMARY},
                     {"role":"user","content":self._summary_prompt()}],
                    cfg["model_summary"], base=cfg["openai_base"], api_key=cfg["openai_api_key"],
                    logger=self.log, expect_json=False, fallbacks=cfg.get("model_fallbacks", [])
                )
                yield ev_msg_md(final_summary)
            except Exception as e:
                self.log.add(f"summary error: {e}")
                yield ev_msg_md(f"⚠️ Summary failed: `{e}`")

            for ev in self.log.flush_events(force=True): yield ev
            yield ev_status("✅ Finished.", done=True)

        except Exception as fatal:
            tb = traceback.format_exc()
            self.log.add("FATAL EXCEPTION")
            self.log.add(str(fatal))
            self.log.add(tb)
            for ev in self.log.all_events(): yield ev
            yield ev_msg_md("**Fatal error**\n\n```\n" + tb + "\n```")
            yield ev_status("❌ Aborted due to fatal error.", done=True)

    # ---------------- helpers ----------------
    def _exec(self, a: Dict) -> Tuple[bool,str]:
        op = a.get("op")
        if op == "navigate":
            return self.browser.navigate(a.get("url",""))
        if op == "click":
            return self.browser.click(ref=a.get("ref"), selector=a.get("selector"))
        if op == "type":
            return self.browser.type(text=a.get("text",""), ref=a.get("ref"), selector=a.get("selector"), submit=bool(a.get("submit", False)))
        if op == "press":
            return self.browser.press(a.get("key","Enter"))
        if op == "wait":
            ms = int(a.get("ms", 800))
            time.sleep(ms/1000.0); return True, f"waited {ms}ms"
        if op == "screenshot":
            return self.browser.screenshot(f"manual_{int(time.time())}")
        return False, f"unknown_op:{op}"

    def _observation_text(self) -> str:
        parts = []
        if self.session.get("last_snapshot"):
            parts.append("STRUCTURED_SNAPSHOT:\n" + self._clip(self.session["last_snapshot"], 2000))
        if self.session.get("visual_last"):
            parts.append("VISUAL_ANALYSIS_JSON:\n" + json.dumps(self.session["visual_last"], ensure_ascii=False))
        return "\n\n".join(parts) if parts else "(no observation yet)"

    def _summary_prompt(self) -> str:
        steps = "\n".join(f"- {s}" for s in self.session["step_log"][-20:])
        return f"Goal:\n{self.session.get('goal','')}\n\nRecent steps:\n{steps}\n"

    def _parse_json(self, s: str) -> Optional[Dict]:
        if not isinstance(s, str): return None
        try:
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(s[i:j+1])
        except Exception as e:
            self.log.add(f"parse_json error: {e}")
        return None

    def _nice_action(self, a: Dict) -> str:
        op = a.get("op","?")
        if op == "navigate": return f"navigate → {a.get('url','')}"
        if op == "click":    return f"click → {a.get('ref') or a.get('selector','?')}"
        if op == "type":     return f"type → {(a.get('ref') or a.get('selector','?'))} = '{a.get('text','')}'"
        if op == "press":    return f"press → {a.get('key','Enter')}"
        if op == "wait":     return f"wait → {a.get('ms',800)}ms"
        if op == "screenshot": return "screenshot"
        return op

    def _clip(self, t: str, n: int) -> str:
        return t if isinstance(t, str) and len(t) <= n else (t[:n] + "\n..." if isinstance(t, str) else str(t))

    def _extract_goal(self, messages: List[dict]) -> Optional[str]:
        for m in reversed(messages or []):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"]
        return None

# ---------------------- OpenWebUI-required module-level entrypoint -----------------

_pipeline_singleton = Pipeline()

async def pipe(user_message: str = "", model_id: Optional[str] = None,
               messages: Optional[List[dict]] = None, body: Optional[dict] = None):
    async for event in _pipeline_singleton.pipe(user_message, model_id, messages or [], body or {}):
        yield event
