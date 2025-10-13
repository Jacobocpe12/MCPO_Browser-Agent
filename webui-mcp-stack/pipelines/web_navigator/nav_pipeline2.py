"""
Playwright MCP — Direct, Streaming Navigator for OpenWebUI

- Uses your proxied MCP at: http://91.99.79.208:3880/mcp_playwright
- Emits real-time status *around actual HTTP calls* (no simulated delays)
- Streams screenshots (inline base64) or via http://91.99.79.208:3888/<file>.png
- Accepts either a simple URL in the prompt, or a JSON plan of actions

Example user message (free-form):
  "Go to https://geoportal.nrw and take a full-page screenshot."

Example user message (JSON plan):
{
  "actions": [
    {"op": "navigate",   "url": "https://geoportal.nrw"},
    {"op": "wait_for",   "state": "load"},
    {"op": "screenshot", "filename": "geoportal-home.png", "fullPage": true}
  ]
}
"""

import os
import re
import json
import base64
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from open_webui.pipelines import Pipeline, Event, Context
except ImportError:
    from pipelines import Pipeline, Event, Context


MCP_BASE   = "http://91.99.79.208:3880/mcp_playwright"
PUBLIC_URL = "http://91.99.79.208:3888"
OUT_DIR    = "/tmp/playwright-output"


class PlaywrightDirectStream(Pipeline):
    def __init__(self):
        super().__init__(
            name="playwright_direct_stream",
            description="Drive Playwright MCP directly with real-time status and screenshots.",
            version="2.0.0",
        )
        os.makedirs(OUT_DIR, exist_ok=True)

    async def on_start(self, ctx: Context):
        await ctx.emit(Event(type="status", data={"message": "✅ Playwright Direct Stream ready"}))

    async def on_run(self, ctx: Context):
        payload = ctx.data or {}
        messages: List[Dict[str, Any]] = payload.get("messages", [])

        # 1) Extract instruction
        text = self._extract_text(messages)
        if not text:
            await ctx.emit(Event(type="status", data={"message": "❌ No instruction found"}))
            return

        # 2) Try parse JSON action plan; otherwise build a minimal plan from URL
        plan = self._parse_plan(text)
        if not plan:
            url = self._extract_url(text)
            if not url:
                await ctx.emit(Event(type="status", data={"message": "❌ No URL or valid plan provided"}))
                return
            # default minimal plan: navigate + screenshot
            plan = [
                {"op": "navigate",   "url": url, "wait_until": "load"},
                {"op": "screenshot", "fullPage": True}
            ]

        # 3) Execute each action, emitting live status around real HTTP calls
        async with httpx.AsyncClient(timeout=90) as client:
            for idx, action in enumerate(plan, start=1):
                op = (action.get("op") or "").lower()
                try:
                    if op == "navigate":
                        url = action["url"]
                        wait_until = action.get("wait_until", "load")
                        await ctx.emit(Event(type="status", data={"message": f"🧭 [{idx}] Navigating: {url} (wait_until={wait_until})"}))
                        r = await client.post(f"{MCP_BASE}/browser_navigate", json={"url": url, "wait_until": wait_until})
                        await self._check_ok(ctx, r, f"[{idx}] navigate")
                        await ctx.emit(Event(type="status", data={"message": f"✅ [{idx}] Page loaded"}))

                    elif op == "click":
                        selector = action["selector"]
                        await ctx.emit(Event(type="status", data={"message": f"🖱️ [{idx}] Click: {selector}"}))
                        r = await client.post(f"{MCP_BASE}/browser_click", json={"selector": selector})
                        await self._check_ok(ctx, r, f"[{idx}] click")
                        await ctx.emit(Event(type="status", data={"message": f"✅ [{idx}] Clicked {selector}"}))

                    elif op == "type":
                        selector = action["selector"]
                        textval  = action["text"]
                        await ctx.emit(Event(type="status", data={"message": f"⌨️ [{idx}] Type into {selector}: {textval}"}))
                        r = await client.post(f"{MCP_BASE}/browser_type", json={"selector": selector, "text": textval})
                        await self._check_ok(ctx, r, f"[{idx}] type")
                        await ctx.emit(Event(type="status", data={"message": f"✅ [{idx}] Typed"}))

                    elif op == "press_key":
                        key = action["key"]
                        await ctx.emit(Event(type="status", data={"message": f"⌨️ [{idx}] Press key: {key}"}))
                        r = await client.post(f"{MCP_BASE}/browser_press_key", json={"key": key})
                        await self._check_ok(ctx, r, f"[{idx}] press_key")
                        await ctx.emit(Event(type="status", data={"message": f"✅ [{idx}] Key pressed"}))

                    elif op == "wait_for":
                        # some servers provide a generic wait endpoint; otherwise use evaluate or wait_for state with navigate
                        state = action.get("state", "networkidle")
                        ms    = action.get("ms")
                        await ctx.emit(Event(type="status", data={"message": f"⏳ [{idx}] Wait: state={state} ms={ms}"}))
                        # Try /browser_wait_for (if available), else fall back to small evaluation wait
                        endpoint = f"{MCP_BASE}/browser_wait_for"
                        payload  = {}
                        if ms is not None: payload["ms"] = ms
                        if state:          payload["state"] = state
                        r = await client.post(endpoint, json=payload)
                        # If server returns 404 for wait_for, silently accept and continue
                        if r.status_code == 404:
                            await ctx.emit(Event(type="status", data={"message": f"ℹ️ [{idx}] /browser_wait_for not supported; continuing"}))
                        else:
                            await self._check_ok(ctx, r, f"[{idx}] wait_for")
                        await ctx.emit(Event(type="status", data={"message": f"✅ [{idx}] Wait complete"}))

                    elif op == "screenshot":
                        full = bool(action.get("fullPage", True))
                        filename = action.get("filename") or f"page-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.png"
                        # ensure absolute path in shared dir
                        if not filename.endswith(".png"):
                            filename += ".png"
                        path = filename if filename.startswith("/") else os.path.join(OUT_DIR, filename)

                        await ctx.emit(Event(type="status", data={"message": f"📸 [{idx}] Screenshot: fullPage={full}, file={path}"}))
                        r = await client.post(f"{MCP_BASE}/browser_take_screenshot", json={"fullPage": full, "filename": path})
                        # If success, try file first
                        if r.status_code == 200 and os.path.exists(path):
                            public = f"{PUBLIC_URL}/{os.path.basename(path)}"
                            await self._emit_public_image(ctx, public)
                            await ctx.emit(Event(type="status", data={"message": f"🖼️ [{idx}] Saved: {public}"}))
                        else:
                            # fallback: check JSON 'data' base64
                            try:
                                data = r.json()
                            except Exception:
                                data = {}
                            b64 = data.get("data")
                            if b64:
                                await self._emit_b64_image(ctx, b64)
                                await ctx.emit(Event(type="status", data={"message": f"🖼️ [{idx}] Inline screenshot ready"}))
                            else:
                                # last resort: request base64 explicitly
                                await ctx.emit(Event(type="status", data={"message": f"↩️ [{idx}] Retrying screenshot as base64"}))
                                r2 = await client.post(f"{MCP_BASE}/browser_take_screenshot", json={"fullPage": full, "return": "base64"})
                                r2.raise_for_status()
                                data2 = r2.json()
                                b64_2 = data2.get("data")
                                if b64_2:
                                    await self._emit_b64_image(ctx, b64_2)
                                    await ctx.emit(Event(type="status", data={"message": f"🖼️ [{idx}] Inline screenshot ready"}))
                                else:
                                    await ctx.emit(Event(type="status", data={"message": f"❌ [{idx}] Screenshot failed (no data)"}))

                    else:
                        await ctx.emit(Event(type="status", data={"message": f"ℹ️ [{idx}] Unknown op '{op}', skipping"}))

                except httpx.HTTPError as http_err:
                    await ctx.emit(Event(type="status", data={"message": f"❌ [{idx}] HTTP error: {http_err}"}))
                except Exception as e:
                    await ctx.emit(Event(type="status", data={"message": f"❌ [{idx}] Error: {e}"}))

        await ctx.emit(Event(type="status", data={"message": "✅ Done"}))

    # ---- helpers ----

    async def _check_ok(self, ctx: Context, resp: httpx.Response, label: str):
        if 200 <= resp.status_code < 300:
            return
        detail = ""
        try:
            detail = resp.text
        except Exception:
            pass
        await ctx.emit(Event(type="status", data={"message": f"❌ {label}: HTTP {resp.status_code} {detail[:300]}"}))
        resp.raise_for_status()

    async def _emit_public_image(self, ctx: Context, public_url: str):
        await ctx.emit(Event(type="image", data={"path": public_url, "mime_type": "image/png"}))

    async def _emit_b64_image(self, ctx: Context, b64: str):
        # Accept both raw b64 and data URLs
        if b64.startswith("data:image"):
            try:
                b64 = b64.split(",", 1)[1]
            except Exception:
                pass
        await ctx.emit(Event(type="image", data={"mime_type": "image/png", "base64": b64}))

    def _extract_text(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                    joined = "\n".join([t for t in texts if t])
                    if joined:
                        return joined
                elif isinstance(content, str):
                    return content
        return None

    def _parse_plan(self, text: str) -> Optional[List[Dict[str, Any]]]:
        text = text.strip()
        if not (text.startswith("{") or text.startswith("[")):
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "actions" in obj and isinstance(obj["actions"], list):
                return obj["actions"]
            if isinstance(obj, list):
                return obj
        except Exception:
            return None
        return None

    def _extract_url(self, text: str) -> Optional[str]:
        m = re.search(r"(https?://[^\s]+)", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"([a-zA-Z0-9.-]+\.[a-z]{2,})", text)
        if m:
            domain = m.group(1)
            if not domain.startswith("http"):
                domain = "https://" + domain
            return domain
        return None


def pipeline() -> Pipeline:
    return PlaywrightDirectStream()
