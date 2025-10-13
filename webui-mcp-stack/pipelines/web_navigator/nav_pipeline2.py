"""
Playwright MCP — Direct streaming pipeline for OpenWebUI

- Talks directly to: http://91.99.79.208:3880/mcp_playwright
- Streams real-time status around real HTTP calls (no fake delays)
- Emits screenshots as inline base64 or via http://91.99.79.208:3888/<file>.png
"""

import os
import re
import json
import base64
from datetime import datetime
from typing import List, Dict, Optional, Union, Generator, Iterator, Tuple, Any

import httpx


class Pipeline:
    def __init__(self):
        self.name = "Playwright Direct (Streaming)"
        self.description = (
            "Drive Playwright MCP via /browser_* endpoints and stream live status + screenshots."
        )
        self.version = "2.3.0"
        self.author = "You"

        # Endpoints / paths
        self.MCP_BASE = "http://91.99.79.208:3880/mcp_playwright"
        self.PUBLIC_BASE = "http://91.99.79.208:3888"
        self.OUT_DIR = "/tmp/playwright-output"
        os.makedirs(self.OUT_DIR, exist_ok=True)

        # HTTP config
        self.TIMEOUT = 90.0

    # ---------------- Hooks ---------------- #

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body

    # ---------------- Main loop ---------------- #

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

        actions = self._parse_actions(prompt)
        if not actions:
            url = self._extract_url(prompt)
            if not url:
                yield self._status("❌ No URL found and no valid action plan provided.", done=True)
                return
            actions = [
                {"op": "navigate", "url": url, "wait_until": "load"},
                {"op": "screenshot", "fullPage": True},
            ]

        try:
            with httpx.Client(timeout=self.TIMEOUT) as http:
                # Ensure actions is a list of dicts
                if not isinstance(actions, list):
                    actions = [actions] if isinstance(actions, dict) else []

                for idx, act in enumerate(actions, start=1):
                    if not isinstance(act, dict):
                        yield self._status(f"ℹ️ [{idx}] Skipping non-dict action: {act!r}")
                        continue

                    op = (act.get("op") or "").lower()

                    # ---------- NAVIGATE ----------
                    if op == "navigate":
                        url = act.get("url")
                        if not isinstance(url, str) or not url:
                            yield self._status(f"❌ [{idx}] navigate: missing 'url'", done=True)
                            return
                        wait_until = act.get("wait_until", "load")
                        yield self._status(f"🛠️ [{idx}] Action: browser_navigate → {url} (wait_until={wait_until})")
                        r = http.post(f"{self.MCP_BASE}/browser_navigate", json={"url": url, "wait_until": wait_until})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] navigate", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: page loaded")

                    # ---------- CLICK ----------
                    elif op == "click":
                        selector = act.get("selector")
                        if not isinstance(selector, str) or not selector:
                            yield self._status(f"❌ [{idx}] click: missing 'selector'", done=True)
                            return
                        yield self._status(f"🛠️ [{idx}] Action: browser_click → {selector}")
                        r = http.post(f"{self.MCP_BASE}/browser_click", json={"selector": selector})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] click", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: clicked {selector}")

                    # ---------- TYPE ----------
                    elif op == "type":
                        selector = act.get("selector")
                        textval = act.get("text")
                        if not isinstance(selector, str) or not isinstance(textval, str):
                            yield self._status(f"❌ [{idx}] type: need 'selector' and 'text'", done=True)
                            return
                        yield self._status(f"🛠️ [{idx}] Action: browser_type → {selector} = {textval}")
                        r = http.post(f"{self.MCP_BASE}/browser_type", json={"selector": selector, "text": textval})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] type", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: typed into {selector}")

                    # ---------- PRESS KEY ----------
                    elif op == "press_key":
                        key = act.get("key")
                        if not isinstance(key, str) or not key:
                            yield self._status(f"❌ [{idx}] press_key: missing 'key'", done=True)
                            return
                        yield self._status(f"🛠️ [{idx}] Action: browser_press_key → {key}")
                        r = http.post(f"{self.MCP_BASE}/browser_press_key", json={"key": key})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] press_key", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: key pressed {key}")

                    # ---------- HOVER ----------
                    elif op == "hover":
                        selector = act.get("selector")
                        if not isinstance(selector, str) or not selector:
                            yield self._status(f"❌ [{idx}] hover: missing 'selector'", done=True)
                            return
                        yield self._status(f"🛠️ [{idx}] Action: browser_hover → {selector}")
                        r = http.post(f"{self.MCP_BASE}/browser_hover", json={"selector": selector})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] hover", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: hovered {selector}")

                    # ---------- SCREENSHOT ----------
                    elif op == "screenshot":
                        full = bool(act.get("fullPage", True))
                        filename = act.get("filename")
                        if not isinstance(filename, str) or not filename:
                            filename = f"page-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.png"
                        if not filename.endswith(".png"):
                            filename += ".png"
                        save_path = filename if filename.startswith("/") else os.path.join(self.OUT_DIR, filename)

                        yield self._status(f"🛠️ [{idx}] Action: browser_take_screenshot → fullPage={full}, file={save_path}")
                        r = http.post(
                            f"{self.MCP_BASE}/browser_take_screenshot",
                            json={"fullPage": full, "filename": save_path},
                        )

                        if self._ok(r):
                            # Prefer local file (shared volume)
                            if os.path.exists(save_path):
                                public_url = f"{self.PUBLIC_BASE}/{os.path.basename(save_path)}"
                                yield self._image_url(public_url)
                                yield self._status(f"👀 [{idx}] Observation: saved to {public_url}")
                            else:
                                # Parse tolerant response (dict|list|string|non-JSON)
                                obj, text = self._jsonish(r)
                                b64 = self._coerce_b64(obj, text)
                                remote_path = self._coerce_path(obj, text)

                                if b64:
                                    yield self._image_b64(b64)
                                    yield self._status(f"👀 [{idx}] Observation: inline screenshot ready")
                                elif remote_path:
                                    public_url = f"{self.PUBLIC_BASE}/{os.path.basename(remote_path)}"
                                    yield self._image_url(public_url)
                                    yield self._status(f"👀 [{idx}] Observation: saved to {public_url}")
                                else:
                                    # Final fallback: explicit base64 request
                                    yield self._status(f"↩️ [{idx}] Retrying screenshot as base64")
                                    r2 = http.post(
                                        f"{self.MCP_BASE}/browser_take_screenshot",
                                        json={"fullPage": full, "return": "base64"},
                                    )
                                    if not self._ok(r2):
                                        yield self._status(self._http_error(f"[{idx}] screenshot(base64)", r2), done=True)
                                        return
                                    obj2, text2 = self._jsonish(r2)
                                    b64_2 = self._coerce_b64(obj2, text2)
                                    if b64_2:
                                        yield self._image_b64(b64_2)
                                        yield self._status(f"👀 [{idx}] Observation: inline screenshot ready")
                                    else:
                                        yield self._status(f"❌ [{idx}] Screenshot failed (no file, no base64)", done=True)
                                        return
                        else:
                            yield self._status(self._http_error(f"[{idx}] screenshot", r), done=True)
                            return

                    # ---------- UNKNOWN ----------
                    else:
                        yield self._status(f"ℹ️ [{idx}] Unknown op '{op}', skipping")

            yield self._status("✅ Finished", done=True)

        except httpx.HTTPError as e:
            yield self._status(f"❌ HTTP error: {e}", done=True)
        except Exception as e:
            yield self._status(f"❌ Error: {e}", done=True)

    # ---------------- Helpers ---------------- #

    def _extract_prompt(self, messages: List[dict]) -> Optional[str]:
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    texts = [b.get("text") for b in c if isinstance(b, dict) and b.get("type") == "text"]
                    txt = "\n".join(t for t in texts if t)
                    if txt:
                        return txt
        return None

    def _parse_actions(self, text: str) -> Optional[List[Dict]]:
        text = text.strip()
        if not (text.startswith("{") or text.startswith("[")):
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("actions"), list):
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

    def _ok(self, resp: httpx.Response) -> bool:
        return 200 <= resp.status_code < 300

    def _http_error(self, label: str, resp: httpx.Response) -> str:
        body = ""
        try:
            body = resp.text[:400]
        except Exception:
            pass
        return f"{label}: HTTP {resp.status_code} {body}"

    # --- tolerant response parsing ---

    def _jsonish(self, resp: httpx.Response) -> Tuple[Optional[Any], str]:
        """Return (obj_or_None, raw_text). Accepts dict|list|str JSON or non-JSON."""
        raw = ""
        try:
            raw = resp.text or ""
        except Exception:
            raw = ""
        try:
            obj = resp.json()
            return obj, raw
        except Exception:
            return None, raw

    def _looks_like_b64(self, s: str) -> bool:
        if not s or len(s) < 64:
            return False
        return re.fullmatch(r"[A-Za-z0-9+/=\s]+", s) is not None

    def _coerce_b64(self, obj: Optional[Any], text: str) -> Optional[str]:
        # data URL in text
        if isinstance(text, str) and text.startswith("data:image"):
            try:
                return text.split(",", 1)[1]
            except Exception:
                return text

        # body looks like base64
        if isinstance(text, str) and self._looks_like_b64(text):
            return text.replace("\n", "")

        # dict shapes
        if isinstance(obj, dict):
            for key in ("data", "base64", "b64"):
                v = obj.get(key)
                if isinstance(v, str) and v:
                    if v.startswith("data:image"):
                        try:
                            return v.split(",", 1)[1]
                        except Exception:
                            return v
                    return v
            image = obj.get("image")
            if isinstance(image, dict):
                v = image.get("base64") or image.get("data")
                if isinstance(v, str) and v:
                    if v.startswith("data:image"):
                        try:
                            return v.split(",", 1)[1]
                        except Exception:
                            return v
                    return v

        # list of dicts
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    b = self._coerce_b64(it, "")
                    if b:
                        return b
                elif isinstance(it, str) and self._looks_like_b64(it):
                    return it.replace("\n", "")

        # obj itself is str?
        if isinstance(obj, str) and self._looks_like_b64(obj):
            return obj.replace("\n", "")

        return None

    def _coerce_path(self, obj: Optional[Any], text: str) -> Optional[str]:
        if isinstance(obj, dict):
            for key in ("path", "file", "filename", "filepath"):
                v = obj.get(key)
                if isinstance(v, str) and v:
                    return v
            res = obj.get("result")
            if isinstance(res, dict):
                for key in ("path", "file", "filename", "filepath"):
                    v = res.get(key)
                    if isinstance(v, str) and v:
                        return v

        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    p = self._coerce_path(it, "")
                    if p:
                        return p
                elif isinstance(it, str):
                    m = re.search(r"(/tmp/[^\s\"']+\.png)", it)
                    if m:
                        return m.group(1)

        if isinstance(text, str):
            m = re.search(r"(/tmp/[^\s\"']+\.png)", text)
            if m:
                return m.group(1)

        return None

    # --- event builders ---

    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": str(description), "done": done}}}

    def _image_b64(self, b64: str) -> Dict:
        return {"event": {"type": "image", "data": {"mime_type": "image/png", "base64": b64}}}

    def _image_url(self, url: str) -> Dict:
        return {"event": {"type": "image", "data": {"mime_type": "image/png", "path": url}}}
