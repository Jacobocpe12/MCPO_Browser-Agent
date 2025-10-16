"""
Web Navigator (MCPO - Narrated)

Pipeline for OpenWebUI that proxies navigation requests through an MCPO instance
connected to Playwright MCP. Streams thought/action/observation/screenshot events
in real time — similar to Browserbase traces.
"""

import time
import base64
import httpx
from pprint import pformat
from typing import List, Dict, Optional, Union, Generator, Iterator
import os


class Pipeline:
    def __init__(self):
        self.name = "Web Navigator (MCPO Narrated)"
        self.description = (
            "Connects to your MCPO/Playwright stack and streams each browser step "
            "as thought/action/observation/screenshot messages."
        )
        self.version = "3.0.0"
        self.author = "Jorge Enrique + ChatGPT"
        self.debug = False

        # 🔗 Direct MCPO URL (no env knobs)
        self.MCPO_BASE_URL = "http://91.99.79.208:3880/mcp_playwright"
        self.POLL_INTERVAL = 1.5
        self.MAX_WAIT_SEC = 900
        self.LOCAL_IMG_DIR = "/tmp/playwright-output"

        os.makedirs(self.LOCAL_IMG_DIR, exist_ok=True)

    # ----------------------------------------------------------
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Pre-run hook."""
        if self.debug:
            print("[web-nav] inlet:", body.keys())
        return body

    # ----------------------------------------------------------
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:

        prompt = self._extract_prompt(messages) or user_message or ""
        if not prompt.strip():
            yield self._status("❌ No navigation prompt provided.", done=True)
            return

        yield self._status(f"🎯 Goal: {prompt}")

        started = time.monotonic()
        last_idx = 0

        try:
            with httpx.Client(timeout=None) as http:
                run_id = self._start_mcpo(http, prompt)
                if not run_id:
                    yield self._status("❌ Failed to start MCPO run.", done=True)
                    return

                yield self._status(f"🚀 Started MCPO run: {run_id}")

                while True:
                    if time.monotonic() - started > self.MAX_WAIT_SEC:
                        yield self._status("⏰ Timed out waiting for MCPO.", done=True)
                        break

                    payload = self._poll_mcpo(http, run_id)
                    if payload is None:
                        time.sleep(self.POLL_INTERVAL)
                        continue

                    finished = bool(payload.get("finished", False))
                    events: List[Dict] = payload.get("events", [])

                    new_events = events[last_idx:] if last_idx < len(events) else []
                    for evt in new_events:
                        self._emit_event(evt, yield_fn=lambda e: (yield e))

                    last_idx = len(events)

                    if finished:
                        yield self._status("✅ Run finished.", done=True)
                        break

                    time.sleep(self.POLL_INTERVAL)

        except Exception as e:
            if self.debug:
                import traceback; traceback.print_exc()
            yield self._status(f"💥 Pipeline error: {e}", done=True)

    # ----------------------------------------------------------
    def _extract_prompt(self, messages: List[dict]) -> Optional[str]:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    texts = [b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"]
                    txt = "\n".join(t for t in texts if t)
                    if txt:
                        return txt
        return None

    # ----------------------------------------------------------
    def _start_mcpo(self, http: httpx.Client, prompt: str) -> Optional[str]:
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/v1/run", json={"prompt": prompt, "mode": "stream"})
            r.raise_for_status()
            data = r.json()
            if self.debug:
                print("[web-nav] start:", pformat(data))
            return data.get("run_id") or data.get("id")
        except Exception as e:
            print("[web-nav] start error:", e)
            return None

    def _poll_mcpo(self, http: httpx.Client, run_id: str) -> Optional[Dict]:
        try:
            r = http.get(f"{self.MCPO_BASE_URL}/v1/run/{run_id}")
            r.raise_for_status()
            data = r.json()
            return data
        except Exception as e:
            print("[web-nav] poll error:", e)
            return None

    # ----------------------------------------------------------
    def _emit_event(self, evt: Dict, yield_fn):
        etype = (evt.get("type") or "").lower()

        if etype in ("thought", "reason"):
            msg = evt.get("message") or evt.get("text")
            if msg:
                yield_fn(self._status(f"🧠 Thought: {msg}"))

        elif etype in ("action", "act"):
            desc = evt.get("description") or evt.get("message") or evt.get("action")
            yield_fn(self._status(f"🛠️ Action: {desc or 'unknown'}"))
            code = evt.get("code") or evt.get("snippet")
            if code:
                yield_fn(self._status(f"```js\n{code}\n```"))

        elif etype in ("log", "observation", "status"):
            msg = evt.get("message") or evt.get("description")
            if msg:
                yield_fn(self._status(f"👀 {msg}"))

        elif etype == "screenshot":
            img_b64 = self._extract_image_b64(evt)
            if img_b64:
                yield_fn({
                    "event": {
                        "type": "image",
                        "data": {"mime_type": "image/png", "base64": img_b64},
                    }
                })
            else:
                yield_fn(self._status("⚠️ Screenshot event missing data/path"))

        elif etype in ("error", "exception"):
            yield_fn(self._status(f"❌ {evt.get('message', 'Unknown error')}"))

    # ----------------------------------------------------------
    def _extract_image_b64(self, evt: Dict) -> Optional[str]:
        data = evt.get("data")
        if data and isinstance(data, str):
            if data.startswith("data:image"):
                try:
                    return data.split(",", 1)[1]
                except Exception:
                    pass
            return data

        path = evt.get("path") or evt.get("file")
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        if path:
            fallback = os.path.join(self.LOCAL_IMG_DIR, os.path.basename(path))
            if os.path.exists(fallback):
                with open(fallback, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

        return None

    # ----------------------------------------------------------
    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}
