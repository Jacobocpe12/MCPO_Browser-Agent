"""
ReAct Web Navigator (Playwright MCP + Structured Snapshot)
----------------------------------------------------------
Autonomous navigation pipeline for OpenWebUI that talks to your
MCPO/Playwright MCP server at http://91.99.79.208:3880/mcp_playwright.

✅ Features:
- Structured DOM snapshot reasoning (Cursor-style)
- "ref" based precise actions (click/type/etc)
- Screenshot + YAML structure hybrid observation
- Step-by-step narration in the OpenWebUI chat stream
- ReAct-style loop (plan → act → observe → retry)
"""

import time
import base64
import json
import os
import httpx
import yaml
from typing import List, Dict, Optional, Union, Generator, Iterator


class Pipeline:
    def __init__(self):
        self.name = "ReAct Web Navigator (Playwright MCP, autodetect)"
        self.description = (
            "Autonomous browser agent using Playwright MCP. "
            "Combines structured snapshots and screenshots for reasoning."
        )
        self.version = "4.0.0"
        self.author = "Jorge Enrique + ChatGPT"
        self.debug = False

        # Core endpoints (MCPO proxy)
        self.MCPO_BASE_URL = "http://91.99.79.208:3880/mcp_playwright"

        # Model configuration
        self.MODEL_NAME = "azure.gpt-4o-sweden"
        self.OPENAI_BASE = "https://ollama.gpu.lfi.rwth-aachen.de/api"

        # Performance tuning
        self.POLL_INTERVAL = 1.5
        self.MAX_WAIT_SEC = 600
        self.LOCAL_IMG_DIR = "/tmp/playwright-output"
        os.makedirs(self.LOCAL_IMG_DIR, exist_ok=True)

    # ----------------------------------------------------------
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.debug:
            print("[web-react] inlet:", body.keys())
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

        yield self._status(f"🔧 Using model: {self.MODEL_NAME} @ {self.OPENAI_BASE}")
        yield self._status(f"🎯 Goal: {prompt}")

        try:
            with httpx.Client(timeout=None) as http:
                # Initialize page session
                yield self._status("🛠️ Boot: initializing Playwright session…")

                # STEP 1 → Navigate
                nav_result = self._call_mcp(http, "browser_navigate", {"url": self._extract_url(prompt)})
                yield self._status(f"🛠️ navigate → {self._extract_url(prompt)}")

                # STEP 2 → Snapshot (structured)
                snapshot = self._browser_snapshot(http)
                if snapshot:
                    yield self._status("📖 Snapshot captured (structured YAML).")
                    yield self._status(snapshot[:1800] + ("\n..." if len(snapshot) > 1800 else ""))

                # STEP 3 → Screenshot
                screenshot_b64, screenshot_url = self._take_screenshot(http)
                if screenshot_url:
                    yield self._image_event(screenshot_b64)
                    yield self._status(f"👀 Screenshot saved to {screenshot_url}")

                # STEP 4 → LLM reasoning loop
                yield self._status("🧠 Reasoning based on snapshot…")
                observation_text = snapshot[:4000] if snapshot else "(no snapshot)"
                thought = self._reason(prompt, observation_text)

                yield self._status(f"🧩 Thought:\n{thought[:1500]}")

                # Optional: parse JSON action if LLM proposes one
                try:
                    act = json.loads(thought)
                    if isinstance(act, dict) and "op" in act:
                        op = act.get("op")
                        yield self._status(f"🛠️ Executing action: {op}")
                        result = self._exec_op(http, op, act)
                        yield self._status(f"✅ {op} → {result}")
                except Exception:
                    yield self._status("ℹ️ No structured action detected, ending loop.")

                yield self._status("✅ Finished navigation.", done=True)

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

    def _extract_url(self, text: str) -> str:
        import re
        match = re.search(r"(https?://[^\s]+)", text)
        return match.group(1) if match else f"https://{text.strip().split()[0]}"

    # ----------------------------------------------------------
    def _call_mcp(self, http, endpoint: str, payload: dict = None):
        """Generic call to Playwright MCP endpoints"""
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/{endpoint}", json=payload or {})
            if self.debug:
                print("[MCP]", endpoint, payload, "→", r.status_code)
            r.raise_for_status()
            data = r.json()
            return data.get("result") or data
        except Exception as e:
            return f"error: {e}"

    def _browser_snapshot(self, http):
        """Fetch structured page snapshot."""
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/browser_snapshot")
            r.raise_for_status()
            data = r.json()
            result = data.get("result") if isinstance(data, dict) else str(data)
            # Convert JSON → YAML for readability
            if result.strip().startswith("{") or result.strip().startswith("["):
                result = yaml.dump(json.loads(result), sort_keys=False, allow_unicode=True)
            return result
        except Exception as e:
            return f"[snapshot error: {e}]"

    def _take_screenshot(self, http):
        """Capture a screenshot and return base64 + URL."""
        try:
            filename = f"page-{time.strftime('%Y%m%dT%H%M%S')}.png"
            r = http.post(
                f"{self.MCPO_BASE_URL}/browser_take_screenshot",
                json={"fullPage": True, "filename": f"/tmp/playwright-output/{filename}"}
            )
            r.raise_for_status()
            # Public URL served by nginx viewer
            url = f"http://91.99.79.208:3888/{filename}"
            path = os.path.join(self.LOCAL_IMG_DIR, filename)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    return b64, url
            return None, url
        except Exception as e:
            return None, f"[screenshot error: {e}]"

    # ----------------------------------------------------------
    def _exec_op(self, http, op, args):
        """Execute MCP operation (supports ref-based clicks)."""
        try:
            if op == "click":
                payload = {"ref": args.get("ref")} if args.get("ref") else {"selector": args.get("selector")}
                r = http.post(f"{self.MCPO_BASE_URL}/browser_click", json=payload)
            elif op == "type":
                payload = {"ref": args.get("ref"), "text": args.get("text", "")}
                r = http.post(f"{self.MCPO_BASE_URL}/browser_type", json=payload)
            elif op == "navigate":
                payload = {"url": args.get("url")}
                r = http.post(f"{self.MCPO_BASE_URL}/browser_navigate", json=payload)
            else:
                return f"Unknown op: {op}"

            r.raise_for_status()
            return r.text[:200]
        except Exception as e:
            return f"[exec error: {e}]"

    # ----------------------------------------------------------
    def _reason(self, goal: str, observation: str) -> str:
        """Ask the LLM what to do next based on the structured snapshot."""
        payload = {
            "model": self.MODEL_NAME,
            "messages": [
                {"role": "system", "content": (
                    "You are a web navigation reasoning engine. "
                    "Use the STRUCTURED SNAPSHOT to decide what to click, type, or navigate next. "
                    "If goal is achieved, summarize success. Output JSON if possible: "
                    '{"op":"click","ref":"e9"} or {"op":"done"}'
                )},
                {"role": "user", "content": f"Goal: {goal}\n\n{observation}"}
            ],
            "stream": False,
        }
        try:
            r = httpx.post(f"{self.OPENAI_BASE}/v1/chat/completions", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ LLM error: {e}"

    # ----------------------------------------------------------
    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}

    def _image_event(self, b64img: str) -> Dict:
        return {
            "event": {
                "type": "image",
                "data": {"mime_type": "image/png", "base64": b64img},
            }
        }
