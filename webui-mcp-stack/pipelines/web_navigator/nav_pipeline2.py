"""
title: Nav Pipeline V3 (Browser bootstrap)
author: You
version: 1.1.0
description: External pipeline with awaited hooks fixed, Valves enabled, sync pipe(); installs MCP Playwright, navigates (google.com by default), returns screenshot URL + logs.
"""

from typing import Iterator, Optional, List
from pydantic import BaseModel, Field
import httpx, time, json, re

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTANT NOTES FOR OPEN WEBUI PIPELINES:
# - External pipelines must expose class Pipeline with a sync pipe() that returns
#   a str or a *synchronous* generator of str. Async generators are not supported
#   reliably; stick to sync yields. (See issues/discussions.)  [refs below]
# - The server *awaits* several optional hooks. If you don't define them,
#   it may "await None" -> TypeError. Define them as async no-ops.
# - Define a nested Valves(BaseModel) and set self.valves = self.Valves()
#   so /<id>/valves/spec returns 200 and the Admin UI can render knobs.
# - Pipeline id should match what your WebUI calls (e.g., nav_pipelinev3).
# ──────────────────────────────────────────────────────────────────────────────

class Pipeline:
    # ----------------------- Valves visible in Admin UI -----------------------
    class Valves(BaseModel):
        # MCP Playwright base (your running service that exposes /browser_* endpoints)
        playwright_base: str = Field(
            default="http://91.99.79.208:3880/mcp_playwright",
            description="MCP Playwright base URL"
        )
        # Public HTTP server that serves image files from img_dir
        screenshot_base: str = Field(
            default="http://91.99.79.208:3888",
            description="Public base URL that serves images from img_dir"
        )
        # Directory on the MCP host where /browser_take_screenshot writes files
        img_dir: str = Field(
            default="/tmp/playwright-output",
            description="Server-side path for screenshots"
        )
        # Behavior
        default_url: str = Field(
            default="https://www.google.com",
            description="Fallback URL when none found in user prompt"
        )
        max_tool_retries: int = Field(
            default=5, ge=1, le=10, description="Retries for MCP calls"
        )
        show_logs: bool = Field(
            default=True, description="Append log tail into chat output"
        )

    def __init__(self):
        # Make the id EXACTLY what WebUI calls (/api/v1/pipelines/nav_pipelinev3/…)
        self.id = "nav_pipelinev3"
        self.name = "Nav Pipeline V3 (Browser bootstrap)"
        self.type = "pipe"
        self.valves = self.Valves()

        # HTTP settings
        self._timeout = httpx.Timeout(60.0, read=60.0, connect=30.0)

    # ──────────────── HOOKS the server may AWAIT (define them!) ───────────────
    async def on_startup(self):
        # no-op but avoids "await None" TypeError
        return None

    async def on_shutdown(self):
        # no-op but avoids "await None" TypeError
        return None

    async def on_valves_updated(self):
        # no-op but avoids "await None" TypeError
        # You can validate endpoints here if you want.
        return None

    # ───────────────────────── internal utilities ─────────────────────────────
    def _retry(self, n: int):
        for i in range(n):
            yield i, min(2 ** i * 0.2, 2.0)

    def _post(self, base: str, ep: str, body: Optional[dict] = None) -> httpx.Response:
        url = f"{base.rstrip('/')}/{ep.lstrip('/')}"
        with httpx.Client(timeout=self._timeout) as http:
            return http.post(url, json=body or {})

    def _extract_url(self, text: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"(https?://[^\s]+)", text, flags=re.I)
        if m:
            return m.group(1)
        d = re.search(r"\b([a-z0-9\-]+\.[a-z]{2,})(/[^\s]*)?\b", text, flags=re.I)
        if d:
            dom = d.group(0)
            return dom if dom.startswith("http") else "https://" + dom
        if "google" in text.lower():
            return "https://www.google.com"
        return None

    def _screenshot(self, tag: str, V: Valves, logs: List[str]) -> str:
        fn = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", tag) + ".png"
        path = f"{V.img_dir.rstrip('/')}/{fn}"
        payload = {"fullPage": True, "filename": path}
        try:
            r = self._post(V.playwright_base, "/browser_take_screenshot", payload)
            logs.append(f"screenshot status={r.status_code} path={path}")
            if r.status_code < 400:
                return f"{V.screenshot_base.rstrip('/')}/{fn}?ts={int(time.time()*1000)}"
            return f"(screenshot_failed status={r.status_code} body={r.text[:160]!r})"
        except Exception as e:
            logs.append(f"screenshot exception: {e}")
            return f"(screenshot_exception: {e})"

    # ─────────────────────────────── core pipe ────────────────────────────────
    def pipe(
        self,
        body,                        # OpenAI-style ChatCompletion body (server schema)
        model_id: Optional[str] = None,
        messages: Optional[list] = None,
        stream: bool = False,
        **kwargs,
    ) -> Iterator[str]:
        """
        Synchronous generator that yields plain text.
        The Pipelines server will stream these chunks back to WebUI.
        """
        V = self.valves
        logs: List[str] = []

        def log(msg: str): logs.append(msg)

        # 0) Extract last user message (OpenAI schema)
        try:
            msgs = (messages or body.messages or [])
            user_text = ""
            for m in reversed(msgs):
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
                if role == "user":
                    content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
                    user_text = content if isinstance(content, str) else json.dumps(content)
                    break
            log(f"user_text_snip={user_text[:160]!r}")
        except Exception as e:
            user_text = ""
            log(f"user_text parse error: {e}")

        # Immediate heartbeat chunk so UI shows something
        yield "⏳ Starting browser bootstrap…\n"

        # 1) Install/start Playwright (MCP)
        ok_install = False
        for i, backoff in self._retry(V.max_tool_retries):
            try:
                r = self._post(V.playwright_base, "/browser_install", {})
                log(f"install attempt={i+1} status={r.status_code}")
                if r.status_code < 400:
                    ok_install = True
                    break
                time.sleep(backoff)
            except Exception as e:
                log(f"install exception: {e}")
                time.sleep(backoff)

        if not ok_install:
            msg = "❌ MCP Playwright install failed — check `playwright_base` reachability."
            if V.show_logs:
                msg += "\n\n--- LOGS (tail) ---\n" + "\n".join(logs[-50:])
            yield msg
            return

        yield "🧩 Browser installed.\n"

        # 2) Decide URL and navigate
        url = self._extract_url(user_text) or V.default_url
        nav_ok = False
        for i, backoff in self._retry(V.max_tool_retries):
            try:
                r = self._post(V.playwright_base, "/browser_navigate", {"url": url})
                log(f"navigate attempt={i+1} url={url} status={r.status_code}")
                if r.status_code < 400:
                    nav_ok = True
                    break
                time.sleep(backoff)
            except Exception as e:
                log(f"navigate exception: {e}")
                time.sleep(backoff)

        # 3) Snapshot (best effort) + Screenshot
        snap_txt = ""
        try:
            r = self._post(V.playwright_base, "/browser_snapshot", {})
            log(f"snapshot status={r.status_code}")
            if r.status_code < 400:
                data = r.json()
                result = data.get("result", data)
                snap_txt = json.dumps(result, ensure_ascii=False)[:1200]
        except Exception as e:
            log(f"snapshot exception: {e}")

        shot_url = self._screenshot("bootstrap", V, logs)

        # 4) Final message (single block)
        out_lines = []
        out_lines.append(f"{'✅' if nav_ok else '⚠️'} Navigated to: {url}")
        if shot_url:
            out_lines.append(f"📸 Screenshot: {shot_url}")
        if snap_txt:
            out_lines.append("📖 Snapshot (clipped):")
            out_lines.append("```json")
            out_lines.append(snap_txt)
            out_lines.append("```")
        if V.show_logs:
            out_lines.append("\n--- LOGS (tail) ---")
            out_lines.extend(logs[-80:])

        yield "\n".join(out_lines)
