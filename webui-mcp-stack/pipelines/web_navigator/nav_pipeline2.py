"""
title: Nav Pipeline V3 (Browser bootstrap)
author: You
version: 0.1.0
description: Sync, valves-enabled pipeline that installs MCP Playwright, navigates (e.g., google.com), and returns a screenshot URL + embedded logs.
"""

from typing import Iterator, List, Optional
from pydantic import BaseModel, Field
import httpx, time, json, re

# NOTE: This is an external Pipelines-server pipeline (NOT a local function).
# - pipe() MUST be synchronous (def), returning str or Iterator[str].
# - Define a Valves class so /<id>/valves/spec is available.

class Pipeline:
    class Valves(BaseModel):
        # --- MCP Playwright / static server
        playwright_base: str = Field(
            default="http://91.99.79.208:3880/mcp_playwright",
            description="Base URL of the MCP Playwright service",
        )
        screenshot_base: str = Field(
            default="http://91.99.79.208:3888",
            description="Public base URL serving /tmp/playwright-output",
        )
        img_dir: str = Field(
            default="/tmp/playwright-output",
            description="Server-side path where screenshots are saved",
        )

        # --- Behavior
        max_tool_retries: int = Field(default=5, ge=1, le=10)
        verbose: bool = Field(default=True, description="Return log tail in chat")
        default_url: str = Field(default="https://www.google.com", description="Fallback URL")

    def __init__(self):
        # IMPORTANT: Set id to match your UI’s calls (/nav_pipelinev3/valves/spec)
        self.id = "nav_pipelinev3"
        self.name = "Nav Pipeline V3 (Browser bootstrap)"
        self.type = "pipe"  # default, but explicit is fine
        self.valves = self.Valves()

        self._timeout = httpx.Timeout(60.0, read=60.0, connect=30.0)

    # Optional lifecycle: called when valves updated (exposed via /valves/update)
    def on_valves_updated(self):
        # No derived config, but you could validate endpoints here
        pass

    # --------------------------- helpers ---------------------------

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
            if not dom.lower().startswith("http"):
                return "https://" + dom
            return dom
        if "google" in text.lower():
            return "https://www.google.com"
        return None

    def _screenshot(self, tag: str, valves: Valves, logs: List[str]) -> str:
        fn = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", tag) + ".png"
        path = f"{valves.img_dir.rstrip('/')}/{fn}"
        payload = {"fullPage": True, "filename": path}
        try:
            r = self._post(valves.playwright_base, "/browser_take_screenshot", payload)
            logs.append(f"screenshot status={r.status_code} path={path}")
            if r.status_code < 400:
                return f"{valves.screenshot_base.rstrip('/')}/{fn}?ts={int(time.time()*1000)}"
            return f"(screenshot_failed status={r.status_code} body={r.text[:200]!r})"
        except Exception as e:
            logs.append(f"screenshot error: {e}")
            return f"(screenshot_exception: {e})"

    # --------------------------- core ---------------------------

    def pipe(
        self,
        body,                  # OpenAIChatCompletionForm (don’t import to avoid hard dep)
        model_id: Optional[str] = None,
        messages: Optional[list] = None,
        stream: bool = False,
        **kwargs,
    ) -> Iterator[str]:
        """
        Synchronous generator returning text chunks.
        This streams a single block at the end (you could yield smaller chunks if you prefer).
        """
        V = self.valves
        logs: List[str] = []
        def log(msg: str): logs.append(msg)

        # 0) Extract user text
        try:
            msgs = (messages or body.messages or [])
            # get last user message content (OpenAI schema)
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

        # 1) Install/start Playwright
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
            if V.verbose:
                msg += "\n\n--- LOGS (tail) ---\n" + "\n".join(logs[-50:])
            yield msg
            return

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

        # 3) Snapshot + Screenshot
        snap_txt = ""
        try:
            r = self._post(V.playwright_base, "/browser_snapshot", {})
            log(f"snapshot status={r.status_code}")
            if r.status_code < 400:
                data = r.json()
                result = data.get("result", data)
                snap_txt = json.dumps(result)[:1000]
        except Exception as e:
            log(f"snapshot exception: {e}")

        shot_url = self._screenshot("bootstrap", V, logs)

        # 4) Return a single, friendly message (with logs appended)
        out_lines = []
        if nav_ok:
            out_lines.append(f"✅ Navigated to: {url}")
        else:
            out_lines.append(f"⚠️ Navigate failed to: {url}")

        if shot_url:
            out_lines.append(f"📸 Screenshot: {shot_url}")

        if snap_txt:
            out_lines.append("📖 Snapshot (clipped):")
            out_lines.append("```json")
            out_lines.append(snap_txt)
            out_lines.append("```")

        if V.verbose:
            out_lines.append("\n--- LOGS (tail) ---")
            out_lines.extend(logs[-80:])

        yield "\n".join(out_lines)
