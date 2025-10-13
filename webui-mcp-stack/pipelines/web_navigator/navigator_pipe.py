"""
Web Navigator (MCPO proxy, narrated)

This OpenWebUI Pipeline starts a run on MCPO and streams a Browserbase-like
trace to the chat: 🧠 Thought, 🛠️ Action, 👀 Observation, 📸 Screenshot.

No Playwright MCP changes required. All steps are proxied via MCPO.
"""

import os
import time
import base64
from typing import List, Dict, Optional, Union, Generator, Iterator
from pprint import pformat
import httpx


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and str(v).strip() != "" else default


class Pipeline:
    def __init__(self):
        self.name = "Web Navigator (MCPO, narrated)"
        self.description = "Starts MCPO runs and streams step-by-step Thought/Action/Observation/Screenshot."
        self.version = "2.2.0"
        self.author = "You"
        self.debug = _env("WEB_NAV_DEBUG", "false").lower() == "true"

        # ---- Config (env) ----
        self.MCPO_BASE_URL   = _env("MCPO_BASE_URL", "http://mcpo:3879")
        self.POLL_INTERVAL   = float(_env("WEB_NAV_POLL_INTERVAL", "1.5"))
        self.MAX_WAIT_SEC    = float(_env("WEB_NAV_MAX_WAIT", "900"))
        self.SHOW_THOUGHTS   = _env("WEB_NAV_SHOW_THOUGHTS", "true").lower() == "true"
        self.SHOW_CODE       = _env("WEB_NAV_SHOW_CODE", "true").lower() == "true"
        self.SHOW_HTML_SNIP  = _env("WEB_NAV_SHOW_HTML_SNIP", "false").lower() == "true"
        self.HTML_SNIP_LEN   = int(float(_env("WEB_NAV_HTML_SNIP_LEN", "800")))
        # If MCPO emits screenshot paths, we may need to read them locally
        self.FALLBACK_IMG_DIR = _env("WEB_NAV_IMG_DIR", "/tmp/playwright-output")

        os.makedirs(self.FALLBACK_IMG_DIR, exist_ok=True)

    # ---------------- Pipeline entry ---------------- #

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.debug:
            print("[web-nav] inlet body keys:", list(body.keys()))
        return body

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
        last_idx = 0  # to only stream *new* events if MCPO returns a cumulative list

        try:
            with httpx.Client(timeout=None) as http:
                run_id = self._start_mcpo(http, prompt)
                if not run_id:
                    yield self._status("❌ Failed to start MCPO run.", done=True)
                    return
                yield self._status(f"🚀 MCPO run started: {run_id}")

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

                    # Stream only newly arrived events
                    new_events = events[last_idx:] if last_idx < len(events) else []
                    if new_events and self.debug:
                        print(f"[web-nav] streaming {len(new_events)} new event(s)")

                    for evt in new_events:
                        self._emit_event(evt, yield_fn=lambda e: (yield e))

                    last_idx = len(events)

                    if finished:
                        yield self._status("✅ MCPO reports run finished.", done=True)
                        break

                    time.sleep(self.POLL_INTERVAL)

        except Exception as e:
            if self.debug:
                import traceback; traceback.print_exc()
            yield self._status(f"💥 Pipeline error: {e}", done=True)

    # ---------------- Helpers ---------------- #

    def _extract_prompt(self, messages: List[dict]) -> Optional[str]:
        # prefer last user message
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

    def _start_mcpo(self, http: httpx.Client, prompt: str) -> Optional[str]:
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/v1/run", json={"prompt": prompt, "mode": "stream"})
            r.raise_for_status()
            data = r.json()
            run_id = data.get("run_id") or data.get("id")
            if self.debug:
                print("[web-nav] /v1/run ->", pformat(data))
            return run_id
        except Exception as e:
            print("[web-nav] start error:", e)
            return None

    def _poll_mcpo(self, http: httpx.Client, run_id: str) -> Optional[Dict]:
        try:
            r = http.get(f"{self.MCPO_BASE_URL}/v1/run/{run_id}")
            r.raise_for_status()
            data = r.json()
            if self.debug:
                # Don’t dump full events; too noisy. Show shape instead.
                print("[web-nav] poll:", {"finished": data.get("finished"), "events": len(data.get("events", []))})
            return data
        except Exception as e:
            print("[web-nav] poll error:", e)
            return None

    # Emitters

    def _emit_event(self, evt: Dict, yield_fn):
        etype = (evt.get("type") or "").lower()

        # Normalize different servers' vocabularies
        # common fields we look for:
        # - thought / reason / rationale
        # - action {name/description/code}
        # - log / observation / message
        # - screenshot {data/path}
        if etype in ("thought", "reason", "rationale") and self.SHOW_THOUGHTS:
            txt = evt.get("message") or evt.get("text") or evt.get("thought") or ""
            if txt.strip():
                yield_fn(self._status(f"🧠 Thought: {txt}"))

        elif etype in ("action", "act"):
            desc = evt.get("description") or evt.get("message") or ""
            name = evt.get("name") or evt.get("action") or ""
            line = f"🛠️ Action: {name or desc or 'unknown'}"
            yield_fn(self._status(line))
            # Optional code block
            if self.SHOW_CODE:
                code = evt.get("code") or evt.get("snippet") or evt.get("repeatable_code")
                if code and isinstance(code, str) and code.strip():
                    yield_fn(self._status(f"```js\n{code}\n```"))

        elif etype in ("log", "observation", "info", "status"):
            msg = evt.get("message") or evt.get("description") or ""
            if msg.strip():
                yield_fn(self._status(f"👀 {msg}"))

            if self.SHOW_HTML_SNIP:
                html = evt.get("html")
                if isinstance(html, str) and html:
                    snip = html[: self.HTML_SNIP_LEN]
                    yield_fn(self._status(f"```html\n{snip}\n```"))

        elif etype == "screenshot":
            img_b64 = self._extract_image_b64(evt)
            if img_b64:
                yield_fn({
                    "event": {
                        "type": "image",
                        "data": {
                            "mime_type": "image/png",
                            "base64": img_b64
                        }
                    }
                })
            else:
                yield_fn(self._status("⚠️ Screenshot event without data/path"))

        elif etype in ("error", "exception"):
            msg = evt.get("message") or evt.get("error") or "Unknown error"
            yield_fn(self._status(f"❌ {msg}"))

        else:
            # Unknown event type: show compact JSON
            if self.debug:
                yield_fn(self._status(f"ℹ️ Unhandled event: {etype or 'unknown'}"))
                yield_fn(self._status(f"```json\n{pformat(evt)}\n```"))

    def _extract_image_b64(self, evt: Dict) -> Optional[str]:
        # priority: explicit base64 data, then read from path if accessible
        data = evt.get("data")
        if data and isinstance(data, str):
            # May already be raw base64; if it looks like data URL, strip header
            if data.startswith("data:image"):
                try:
                    return data.split(",", 1)[1]
                except Exception:
                    pass
            return data

        path = evt.get("path") or evt.get("file") or evt.get("filename")
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None

        # some servers emit basename only; try fallback dir
        if path:
            fallback = os.path.join(self.FALLBACK_IMG_DIR, os.path.basename(path))
            if os.path.exists(fallback):
                try:
                    with open(fallback, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    return None

        return None

    # Format helpers
    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}
