"""
title: Browser MOA (Planner + Visual + Verifier)
author: You
version: 0.1.1
description: Multi-agent browser pipeline that plans, executes MCP browser tools, streams screenshots/status, and uses a visual LLM in parallel.
"""

import os, time, json, asyncio, httpx, yaml, re
from typing import Optional, Dict, Any, List, Tuple, AsyncGenerator

# --------------------------- helpers for OpenWebUI events ---------------------------

def ev_status(desc: str, done: bool=False) -> Dict[str, Any]:
    return {"event": {"type": "status", "data": {"description": desc, "done": done}}}

def ev_msg_md(md: str) -> Dict[str, Any]:
    # include role and markdown hint for better OpenWebUI rendering
    return {"event": {"type": "message", "data": {"role": "assistant", "content": md, "content_type": "text/markdown"}}}

# ------------------------------------ knobs ---------------------------------------

class Knobs:
    # Models
    MODEL_VISUAL    = os.getenv("MODEL_VISUAL",    "gpt-4o-mini")     # visual analyzer
    MODEL_PLANNER   = os.getenv("MODEL_PLANNER",   "gpt-4o-mini")     # planner
    MODEL_VERIFIER  = os.getenv("MODEL_VERIFIER",  "gpt-4o-mini")     # gatekeeper
    MODEL_SUMMARY   = os.getenv("MODEL_SUMMARY",   "gpt-4o-mini")     # finisher

    # OpenAI-compatible endpoint (or your proxy)
    # IMPORTANT: do not hard-code secrets. Let OpenWebUI provide env vars.
    OPENAI_BASE     = os.getenv("OPENAI_BASE",     "http://localhost:11434")  # change as needed
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")  # no default secret!

    # MCP Playwright base
    PLAYWRIGHT_BASE = os.getenv("PLAYWRIGHT_BASE", "http://127.0.0.1:3880/mcp_playwright")

    # Public screenshot base (nginx serving /tmp/playwright-output)
    SCREENSHOT_BASE = os.getenv("SCREENSHOT_BASE", "http://127.0.0.1:3888")

    # Retries / timing
    MAX_TOOL_RETRIES = int(os.getenv("MAX_TOOL_RETRIES", "5"))
    STEP_LIMIT       = int(os.getenv("STEP_LIMIT", "15"))

    # Session
    IMG_DIR          = os.getenv("IMG_DIR", "/tmp/playwright-output")

    SYS_VERIFIER = os.getenv("SYS_VERIFIER","""You are the gatekeeper for a web-browsing agent.
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

Never include extra text or comments.""")
    SYS_PLANNER = os.getenv("SYS_PLANNER","""You are the Planner. You receive: GOAL and OBSERVATION.
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
Reason in your head, output only JSON.""")
    SYS_VISUAL = os.getenv("SYS_VISUAL","""You analyze a screenshot URL plus a YAML snapshot. Return strict JSON:
{
 "view":"map|list|form|article|unknown",
 "center":"<text or null>",
 "zoom": "<int or null>",
 "notable_elements":[{"ref":"e12","role":"button","text":"Login"}],
 "obstacles":[ "cookie_banner" ],
 "next_click_hint": {"ref":"e12"}
}
No prose. Use nulls when unknown.""")
    SYS_SUMMARY = os.getenv("SYS_SUMMARY","""Write a concise, user-friendly summary of what the browsing agent did,
what it found, and the final outcome. Include key URLs if helpful. Keep it short and clear.""")

    SYS_GOAL_EXTRACTOR = os.getenv("SYS_GOAL_EXTRACTOR", """You are a goal normalizer for a web-browsing agent.
The user provides an ambiguous text. Return a normalized JSON goal description:

{
 "goal": "<concise statement of what to achieve>",
 "intent": "<search|form_filling|scraping|navigation|analysis|unknown>",
 "targets": ["optional list of domains or keywords"]
}

If the text is pure chat or not browser-related, keep "intent": "unknown".
Return only valid JSON.
""")


# ----------------------------------- tools ----------------------------------------

class BrowserTool:
    def __init__(self, base: str, img_dir: str, screenshot_base: str, retries: int):
        self.base = base.rstrip("/")
        self.img_dir = img_dir
        self.screenshot_base = screenshot_base.rstrip("/")
        self.retries = retries

    def _url(self, ep: str) -> str:
        return f"{self.base}/{ep.lstrip('/')}"

    def _retry(self):
        for i in range(self.retries):
            yield i, min(2 ** i * 0.2, 2.0)

    def _post(self, ep: str, json_body: Optional[Dict]=None) -> httpx.Response:
        with httpx.Client(timeout=httpx.Timeout(60, read=60, connect=30)) as http:
            return http.post(self._url(ep), json=json_body or {})

    def install(self) -> Tuple[bool, str]:
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_install", {})
                if r.status_code < 400:
                    return True, "installed"
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, "install_failed"

    def navigate(self, url: str) -> Tuple[bool, str]:
        payload = {"url": url}
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_navigate", payload)
                if r.status_code < 400:
                    return True, "navigated"
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, f"navigate_failed:{url}"

    def wait_for(self, selector_or_ref: str=None, timeout_ms: int=3000) -> Tuple[bool,str]:
        payload = {}
        if selector_or_ref:
            if selector_or_ref.startswith("e"): payload["ref"] = selector_or_ref
            else: payload["selector"] = selector_or_ref
        payload["timeout"] = timeout_ms
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_wait_for", payload)
                if r.status_code < 400:
                    return True, "ready"
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, "wait_failed"

    def snapshot(self) -> Tuple[bool, str]:
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_snapshot", {})
                if r.status_code < 400:
                    data = r.json()
                    result = data.get("result", data)
                    if isinstance(result, str) and (result.startswith("{") or result.startswith("[")):
                        try:
                            result = yaml.dump(json.loads(result), sort_keys=False, allow_unicode=True)
                        except Exception:
                            pass
                    return True, result if isinstance(result, str) else str(result)
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, "snapshot_failed"

    def click(self, ref: Optional[str]=None, selector: Optional[str]=None) -> Tuple[bool,str]:
        payload = {}
        if ref: payload["ref"] = ref
        elif selector: payload["selector"] = selector
        else: return False, "click_missing_target"
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_click", payload)
                if r.status_code < 400:
                    return True, "clicked"
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, "click_failed"

    def type(self, text: str, ref: Optional[str]=None, selector: Optional[str]=None, submit: bool=False) -> Tuple[bool,str]:
        payload = {"text": text, "submit": bool(submit)}
        if ref: payload["ref"] = ref
        elif selector: payload["selector"] = selector
        else: return False, "type_missing_target"
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_type", payload)
                if r.status_code < 400:
                    return True, "typed"
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, "type_failed"

    def press(self, key: str="Enter") -> Tuple[bool,str]:
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_press_key", {"key": key})
                if r.status_code < 400:
                    return True, "pressed"
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, "press_failed"

    def screenshot(self, descriptive: str) -> Tuple[bool, str]:
        fn = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", descriptive) + ".png"
        path = f"{self.img_dir}/{fn}"
        payload = {"fullPage": True, "filename": path}
        for _, sleep_s in self._retry():
            try:
                r = self._post("/browser_take_screenshot", payload)
                if r.status_code < 400:
                    return True, f"{Knobs.SCREENSHOT_BASE}/{fn}?ts={int(time.time()*1000)}"
                time.sleep(sleep_s)
            except Exception:
                time.sleep(sleep_s)
        return False, "screenshot_failed"


# ------------------------------- LLM wrapper --------------------------------------

async def chat_complete(messages: List[Dict], model: str, expect_json: bool=False) -> str:
    payload = {"model": model, "messages": messages, "stream": False}
    headers = {"Authorization": f"Bearer {Knobs.OPENAI_API_KEY}"} if Knobs.OPENAI_API_KEY else {}

    base = Knobs.OPENAI_BASE.rstrip("/")
    candidates = [
        f"{base}/v1/chat/completions",
        f"{base}/chat/completions",
        f"{base}/api/v1/chat/completions",
        f"{base}/api/chat/completions",
    ]

    async with httpx.AsyncClient(timeout=httpx.Timeout(90, read=90, connect=30)) as http:
        last_error = None
        for url in candidates:
            try:
                r = await http.post(url, json=payload, headers=headers)
                if r.status_code < 400:
                    data = r.json()
                    out = data["choices"][0]["message"]["content"]
                    if expect_json:
                        s, e = out.find("{"), out.rfind("}")
                        if s != -1 and e != -1 and e > s:
                            return out[s:e+1]
                    return out
            except Exception as e:
                last_error = e
                continue
        # If nothing worked, raise the last error to surface it in the UI
        raise RuntimeError(f"chat_complete failed against {candidates}: {last_error}")

# ---------------------------------- pipeline --------------------------------------

class Pipeline:
    def __init__(self):
        self.knobs = Knobs()
        self.session: Dict[str, Any] = {
            "goal": None,
            "last_snapshot": None,
            "step_log": [],
            "visual_last": None
        }
        self.browser = BrowserTool(
            base=Knobs.PLAYWRIGHT_BASE,
            img_dir=Knobs.IMG_DIR,
            screenshot_base=Knobs.SCREENSHOT_BASE,
            retries=Knobs.MAX_TOOL_RETRIES,
        )

    async def pipe(self, user_message: str, model_id: Optional[str], messages: List[dict], body: dict) -> AsyncGenerator[Dict[str, Any], None]:
        # 1) Gate: should we run the pipeline?
        raw_goal = self._extract_goal(messages) or (user_message or "")
        yield ev_status("📨 Reading messages...", done=False)
        if not raw_goal.strip():
            yield ev_status("❌ Empty goal received — stopping.", done=True)
            return

        try:
            decision_json = await chat_complete(
                [{"role": "system", "content": self.knobs.SYS_VERIFIER},
                 {"role": "user", "content": raw_goal}],
                self.knobs.MODEL_VERIFIER,
                expect_json=True
            )
            decision = json.loads(decision_json)
        except Exception:
            # Fallback if parsing fails
            decision = {"use_pipeline": True, "goal": raw_goal, "intent": "unknown", "targets": []}

        use_pipeline = bool(decision.get("use_pipeline", False))
        goal        = decision.get("goal") or raw_goal

        # ✅ Display decision
        yield ev_msg_md(f"🤖 **Verifier Decision:**\n\n```json\n{json.dumps(decision, indent=2)}\n```")

        # 🧩 If pipeline not needed → stop here
        if not use_pipeline:
            yield ev_status("💬 Verifier: pipeline not required.", done=True)
            yield ev_msg_md("This query doesn't require browser actions.")
            return

        # Otherwise set and announce the goal
        self.session["goal"] = goal
        yield ev_status(f"🧭 Pipeline activated for goal: {goal}", done=False)

        # 2) Start session / install browser
        yield ev_status("🧩 Installing/starting browser…", done=False)
        ok, msg = await asyncio.to_thread(self.browser.install)
        # mark install step
        yield ev_status(f"🧩 {msg}", done=True)
        if not ok:
            yield ev_msg_md("❌ Browser install failed. Check MCP Playwright server address and availability.")
            return

        # 3) Main loop
        for step in range(1, Knobs.STEP_LIMIT + 1):
            # Build observation for planner (include last snapshot + last visual)
            obs_text = self._observation_text()

            # Ask planner for next action (JSON only)
            try:
                plan_json = await chat_complete(
                    [{"role":"system","content":self.knobs.SYS_PLANNER},
                     {"role":"user","content":f"GOAL:\n{goal}\n\nOBSERVATION:\n{obs_text}"}],
                    self.knobs.MODEL_PLANNER, expect_json=True
                )
            except Exception as e:
                yield ev_msg_md(f"⚠️ Planner request failed: `{e}`. Stopping.")
                break

            act = self._parse_json(plan_json)
            if not act:
                yield ev_status("ℹ️ Planner returned no structured action; stopping.", done=True)
                break

            if act.get("op") in ("done","finish","stop"):
                reason = act.get("reason","done")
                yield ev_status(f"✅ Done: {reason}", done=True)
                break

            # 4) Execute action via BrowserTool
            nice = self._nice_action(act)
            self.session["step_log"].append(nice)
            yield ev_status(f"🛠️ {nice}", done=False)

            ok, exec_msg = await asyncio.to_thread(self._exec, act)
            if not ok:
                self.session["step_log"].append(f"⚠️ {exec_msg}")
                yield ev_status(f"⚠️ {exec_msg} (retrying may occur)", done=False)

            # 5) Snapshot + Screenshot + parallel visual analysis
            snap_ok, snapshot = await asyncio.to_thread(self.browser.snapshot)
            if snap_ok:
                self.session["last_snapshot"] = snapshot
                clipped = self._clip(snapshot, 1200)
                yield ev_msg_md("📖 **Snapshot (YAML, clipped)**\n\n```\n" + clipped + "\n```")

            shot_ok, url_or_err = await asyncio.to_thread(self.browser.screenshot, f"step_{step}_{int(time.time())}")
            if shot_ok:
                shot_url = url_or_err
                yield ev_msg_md(f"📸 **View after step {step}:**\n\n![frame]({shot_url})")
                # Launch visual analysis in parallel (do not block)
                asyncio.create_task(self._run_visual(shot_url, self.session.get("last_snapshot")))
            else:
                yield ev_status("⚠️ Screenshot failed", done=False)

        # 6) Final summary
        yield ev_status("📦 Summarizing…", done=False)
        try:
            final_summary = await chat_complete(
                [{"role":"system","content":self.knobs.SYS_SUMMARY},
                 {"role":"user","content":self._summary_prompt()}],
                self.knobs.MODEL_SUMMARY, expect_json=False
            )
            yield ev_msg_md(final_summary)
        except Exception as e:
            yield ev_msg_md(f"⚠️ Summary failed: `{e}`")

        yield ev_status("✅ Finished.", done=True)

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

    async def _run_visual(self, shot_url: str, snapshot: Optional[str]):
        obs = f"SCREENSHOT_URL: {shot_url}\n\nSNAPSHOT_YAML:\n{self._clip(snapshot or '', 4000)}"
        try:
            out = await chat_complete(
                [{"role": "system", "content": self.knobs.SYS_VISUAL},
                 {"role": "user", "content": obs}],
                self.knobs.MODEL_VISUAL, expect_json=True
            )
            parsed = json.loads(out)
        except Exception:
            parsed = {"view": "unknown", "center": None, "zoom": None,
                      "notable_elements": [], "obstacles": []}
        self.session["visual_last"] = parsed

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
        except Exception:
            pass
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
        return t if len(t) <= n else t[:n] + "\n..."

    def _extract_goal(self, messages: List[dict]) -> Optional[str]:
        for m in reversed(messages or []):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"]
        return None


# ---------------------- OpenWebUI-required module-level entrypoint -----------------

_pipeline_singleton = Pipeline()

async def pipe(user_message: str = "", model_id: Optional[str] = None,
               messages: Optional[List[dict]] = None, body: Optional[dict] = None):
    """
    OpenWebUI calls this function. It must be an async generator yielding event dicts.
    """
    async for event in _pipeline_singleton.pipe(user_message, model_id, messages or [], body or {}):
        yield event
