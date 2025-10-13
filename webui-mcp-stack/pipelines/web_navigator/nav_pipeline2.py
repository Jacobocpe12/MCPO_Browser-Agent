"""
Playwright MCP — Direct streaming pipeline for OpenWebUI

- Uses proxied MCP at: http://91.99.79.208:3880/mcp_playwright
- No /v1/run. Calls /browser_* endpoints directly.
- Streams live status around real HTTP calls (no simulated sleeps).
- Emits screenshots as inline base64 or public URL (http://91.99.79.208:3888/<file>.png).
"""

import os
import re
import json
import base64
import time
from datetime import datetime
from typing import List, Dict, Optional, Union, Generator, Iterator, Tuple, Any

import httpx


class Pipeline:
    def __init__(self):
        self.name = "Playwright Direct (Streaming)"
        self.description = (
            "Drive Playwright MCP directly via /browser_* endpoints and stream live status + screenshots."
        )
        self.version = "2.2.0"
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

        # Try JSON action plan; else fall back to simple URL + screenshot
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
                for idx, act in enumerate(actions, start=1):
                    op = (act.get("op") or "").lower()

                    # ---------- NAVIGATE ----------
                    if op == "navigate":
                        url = act["url"]
                        wait_until = act.get("wait_until", "load")
                        yield self._status(f"🛠️ [{idx}] Action: browser_navigate → {url} (wait_until={wait_until})")
                        r = http.post(f"{self.MCP_BASE}/browser_navigate", json={"url": url, "wait_until": wait_until})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] navigate", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: page loaded")

                    # ---------- CLICK ----------
                    elif op == "click":
                        selector = act["selector"]
                        yield self._status(f"🛠️ [{idx}] Action: browser_click → {selector}")
                        r = http.post(f"{self.MCP_BASE}/browser_click", json={"selector": selector})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] click", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: clicked {selector}")

                    # ---------- TYPE ----------
                    elif op == "type":
                        selector = act["selector"]
                        textval = act["text"]
                        yield self._status(f"🛠️ [{idx}] Action: browser_type → {selector} = {textval}")
                        r = http.post(f"{self.MCP_BASE}/browser_type", json={"selector": selector, "text": textval})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] type", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: typed into {selector}")

                    # ---------- PRESS KEY ----------
                    elif op == "press_key":
                        key = act["key"]
                        yield self._status(f"🛠️ [{idx}] Action: browser_press_key → {key}")
                        r = http.post(f"{self.MCP_BASE}/browser_press_key", json={"key": key})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] press_key", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: key pressed {key}")

                    # ---------- HOVER ----------
                    elif op == "hover":
                        selector = act["selector"]
                        yield self._status(f"🛠️ [{idx}] Action: browser_hover → {selector}")
                        r = http.post(f"{self.MCP_BASE}/browser_hover", json={"selector": selector})
                        if not self._ok(r):
                            yield self._status(self._http_error(f"[{idx}] hover", r), done=True)
                            return
                        yield self._status(f"👀 [{idx}] Observation: hovered {selector}")

                    # ---------- SCREENSHOT ----------
                    elif op == "screenshot":
                        full = bool(act.get("fullPage", True))
                        filename = act.get("filename") or f"page-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.png"
                        if not filename.endswith(".png"):
                            filename += ".png"
                        save_path = filename if filename.startswith("/") else os.path.join(self.OUT_DIR, filename)

                        yield self._status(f"🛠️ [{idx}] Action: browser_take_screenshot → fullPage={full}, file={save_path}")
                        r = http.post(
                            f"{self.MCP_BASE}/browser_take_screenshot",
                            json={"fullPage": full, "filename": save_path},
                        )

                        if self._ok(r):
                            # 1) If we can see the file locally (shared volume), prefer public URL
                            if os.path.exists(save_path):
                                public_url = f"{self.PUBLIC_BASE}/{os.path.basename(save_path)}"
                                yield self._image_url(public_url)
                                yield self._status(f"👀 [{idx}] Observation: saved to {public_url}")
                            else:
                                # 2) Try to parse JSON-ish and extract data/path
                                obj, text = self._jsonish(r)
                                b64 = self._coerce_b64(obj, text)
                                remote_path = self._coerce_path(obj, text)

                                if b64:
                                    yield self._image_b64(b64)
                                    yield self._status(f"👀 [{idx}] Observation: inline screenshot ready")
                                elif remote_path:
                                    # Publish via base name
                                    public_url = f"{self.PUBLIC_BASE}/{os.path.basename(remote_path)}"
                                    yield self._image_url(public_url)
                                    yield self._status(f"👀 [{idx}] Observation: saved to {public_url}")
                                else:
                                    # 3) Final fallback: explicit base64 request
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
        """
        Try to parse JSON; if not JSON, return (None, text).
        If JSON is a string, return (that_string, that_string).
        """
        text = ""
        try:
            text = resp.text or ""
        except Exception:
            text = ""
        try:
            obj = resp.json()
            # Some servers return a raw string as JSON
            if isinstance(obj, str):
                return obj, obj
            return obj, text
        except Exception:
            return None, text

    def _looks_like_b64(self, s: str) -> bool:
        # Heuristic: long-ish, base64 charset
        if not s or len(s) < 64:
            return False
        return re.fullmatch(r"[A-Za-z0-9+/=\s]+", s) is not None

    def _coerce_b64(self, obj: Optional[Any], text: str) -> Optional[str]:
        """
        Try to extract a base64 image from various shapes:
        - {"data": "..."} or {"base64": "..."} or {"image": {"base64": "..."}}
        - list containing such dicts
        - response body itself is a base64 string
        - data: URL
        """
        # data-URL in text
        if text.startswith("data:image"):
            try:
                return text.split(",", 1)[1]
            except Exception:
                pass

        # body looks like base64
        if self._looks_like_b64(text):
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
                        try: return v.split(",", 1)[1]
                        except Exception: return v
                    return v

        # list of dicts
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, dict):
                    b = self._coerce_b64(it, text="")
                    if b:
                        return b

        # obj as raw string
        if isinstance(obj, str) and self._looks_like_b64(obj):
            return obj.replace("\n", "")

        return None

    def _coerce_path(self, obj: Optional[Any], text: str) -> Optional[str]:
        """
        Try to extract file path/filename from JSON-ish responses.
        """
        if isinstance(obj, dict):
            for key in ("path", "file", "filename", "filepath"):
                v = obj.get(key)
                if isinstance(v, str) and v:
                    return v
            # nested "result"
            res = obj.get("result")
            if isinstance(res, dict):
                for key in ("path", "file", "filename", "filepath"):
                    v = res.get(key)
                    if isinstance(v, str) and v:
                        return v

        if isinstance(obj, list):
            for it in obj:
                p = self._coerce_path(it, text="")
                if p:
                    return p

        # crude guess in plain text
        m = re.search(r"(/tmp/[^\s\"']+\.png)", text or "")
        if m:
            return m.group(1)

        return None

    # event builders
    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}

    def _image_b64(self, b64: str) -> Dict:
        return {"event": {"type": "image", "data": {"mime_type": "image/png", "base64": b64}}}

    def _image_url(self, url: str) -> Dict:
        return {"event": {"type": "image", "data": {"mime_type": "image/png", "path": url}}}
