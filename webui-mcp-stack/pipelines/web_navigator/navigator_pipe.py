"""
Web Navigator pipeline for OpenWebUI.

This pipeline forwards natural language navigation instructions
to the MCPO hub that orchestrates Playwright MCP or Puppeteer MCP.
It streams progress and screenshots back to the OpenWebUI chat
so users can visually follow the browsing process.
"""

import base64
import os
import time
from typing import Any, Dict, Generator, Iterator, List, Optional, Union
import httpx
from pprint import pprint


class Pipeline:
    def __init__(self):
        self.name = "Web Navigator"
        self.description = "Navigate via MCPO (Playwright/Puppeteer) and stream screenshots."
        self.author = "Raúl"
        self.version = "2.0.0"
        self.debug = True

        # Tunable settings (valves)
        self.mcpo_base_url = os.getenv("MCPO_BASE_URL", "http://mcpo:3879")
        self.poll_interval = float(os.getenv("WEB_NAVIGATOR_POLL_INTERVAL", 2.0))
        self.max_wait = float(os.getenv("WEB_NAVIGATOR_MAX_WAIT", 900.0))

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Called before sending request to LLM (debug hook)."""
        if self.debug:
            print("[WebNavigator] inlet received body:")
            pprint(body)
        return body

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """Core pipeline logic for OpenWebUI runtime."""

        # 1️⃣ Extract prompt/instruction
        instruction = self._extract_instruction(messages) or user_message
        if not instruction:
            yield {"event": {"type": "status", "data": {"description": "No navigation prompt provided.", "done": True}}}
            return

        yield {"event": {"type": "status", "data": {"description": "Starting MCPO navigation...", "done": False}}}

        try:
            with httpx.Client(timeout=None) as client:
                run = self._start_mcpo_run(client, instruction)
                if not run:
                    yield {"event": {"type": "status", "data": {"description": "Failed to start MCPO run", "done": True}}}
                    return

                start_time = time.monotonic()
                yield {"event": {"type": "status", "data": {"description": f"Tracking MCPO run {run['id']}...", "done": False}}}

                while True:
                    if time.monotonic() - start_time > self.max_wait:
                        yield {"event": {"type": "status", "data": {"description": "Navigation timed out", "done": True}}}
                        break

                    progress = self._fetch_progress(client, run["id"])
                    if not progress:
                        time.sleep(self.poll_interval)
                        continue

                    # Handle MCPO progress events
                    for item in progress.get("events", []):
                        if item.get("type") == "log":
                            yield {"event": {"type": "status", "data": {"description": item.get("message", "")}}}
                        elif item.get("type") == "screenshot":
                            img_data = self._load_image(item)
                            if img_data:
                                yield {
                                    "event": {
                                        "type": "image",
                                        "data": {
                                            "mime_type": "image/png",
                                            "base64": img_data,
                                        },
                                    }
                                }
                        elif item.get("type") == "action":
                            yield {"event": {"type": "status", "data": {"description": item.get("description", "")}}}

                    if progress.get("finished", False):
                        yield {"event": {"type": "status", "data": {"description": "✅ Navigation complete", "done": True}}}
                        break

                    time.sleep(self.poll_interval)

        except Exception as e:
            yield {"event": {"type": "status", "data": {"description": f"Error: {str(e)}", "done": True}}}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _extract_instruction(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract user instruction text from messages list."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                if isinstance(msg.get("content"), list):
                    blocks = [
                        block.get("text") for block in msg["content"]
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    return "\n".join(filter(None, blocks))
                elif isinstance(msg.get("content"), str):
                    return msg["content"]
        return None

    def _start_mcpo_run(self, client: httpx.Client, instruction: str) -> Optional[Dict[str, Any]]:
        """POST /v1/run to start navigation task."""
        try:
            r = client.post(f"{self.mcpo_base_url}/v1/run", json={"prompt": instruction, "mode": "stream"})
            r.raise_for_status()
            return {"id": r.json().get("run_id") or r.json().get("id")}
        except Exception as e:
            print(f"[WebNavigator] Failed to start MCPO run: {e}")
            return None

    def _fetch_progress(self, client: httpx.Client, run_id: str) -> Optional[Dict[str, Any]]:
        """GET /v1/run/{id} for progress updates."""
        try:
            r = client.get(f"{self.mcpo_base_url}/v1/run/{run_id}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[WebNavigator] Polling error: {e}")
            return None

    def _load_image(self, item: Dict[str, Any]) -> Optional[str]:
        """Decode or read screenshot data from MCPO event."""
        if path := item.get("path"):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None
        elif data := item.get("data"):
            try:
                return base64.b64encode(base64.b64decode(data)).decode("utf-8")
            except Exception:
                return None
        return None
