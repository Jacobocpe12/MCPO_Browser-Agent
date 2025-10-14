"""
ReAct Web Navigator (Playwright MCP + Snapshot + Final Summary)
---------------------------------------------------------------
Autonomous navigation pipeline for OpenWebUI that talks to your
MCPO/Playwright MCP server at http://91.99.79.208:3880/mcp_playwright.

Features:
- Structured DOM snapshot reasoning (Cursor-style)
- "ref" based precise actions (click/type/etc)
- Screenshot + YAML structure hybrid observation
- Multi-step ReAct loop (plan → act → observe → retry) with stop conditions
- Streams narrated steps AND produces a final human summary at the end
"""

import os
import re
import time
import json
import base64
import httpx
import yaml
from typing import List, Dict, Optional, Union, Generator, Iterator


class Pipeline:
    def __init__(self):
        self.name = "ReAct Web Navigator (Playwright MCP, snapshot, summary)"
        self.description = (
            "Autonomous browser agent using Playwright MCP. "
            "Combines structured snapshots and screenshots for reasoning, "
            "streams steps, and returns a final summary."
        )
        self.version = "5.0.0"
        self.author = "You + ChatGPT"
        self.debug = False

        # === Core endpoints (MCPO proxy) ===
        self.MCPO_BASE_URL = "http://91.99.79.208:3880/mcp_playwright"

        # === Model config (OpenAI-compatible) ===
        self.OPENAI_BASE = os.getenv("OPENAI_BASE", "https://ollama.gpu.lfi.rwth-aachen.de/api")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "azure.gpt-4o-sweden")

        # === Loop & I/O ===
        self.LOCAL_IMG_DIR = "/tmp/playwright-output"
        os.makedirs(self.LOCAL_IMG_DIR, exist_ok=True)
        self.MAX_STEPS = 12
        self.NO_PROGRESS_LIMIT = 3
        self.SCREENSHOT_PUBLIC_BASE = "http://91.99.79.208:3888"

        self._step_log: List[str] = []
        self._last_snapshots: List[str] = []

    # ----------------------------------------------------------
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.debug:
            print("[nav] inlet keys:", list((body or {}).keys()))
        return body

    # ----------------------------------------------------------
    # MAIN PIPELINE (OpenWebUI-compatible)
    # ----------------------------------------------------------
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:

        # === Maintain in-memory session between calls ===
        if not hasattr(self, "session"):
            class _Session:
                last_goal = None
                last_snapshot = None
                last_error = None
                tool_mode = False
            self.session = _Session()

        # === Detect intent (chat vs tool) ===
        intent = self._infer_intent(user_message)
        is_followup = (
            self.session.last_goal
            and len(user_message.split()) < 30
            and not re.search(r"https?://", user_message)
            and intent == "tool"
        )

        # === If it's a normal chat ===
        if intent == "chat" and not self.session.tool_mode:
            yield self._status("💬 Detected chat intent — replying normally.")
            reply = self._chat([
                {"role": "system", "content": "You are a helpful conversational assistant."},
                {"role": "user", "content": user_message},
            ])
            yield reply
            yield self._status("✅ Finished.", done=True)
            return

        # === Continue previous navigation ===
        if is_followup:
            yield self._status("🧩 Detected follow-up — continuing previous navigation.")
            goal = f"{self.session.last_goal}. Update based on user feedback: {user_message}"
        else:
            goal = self._extract_prompt(messages) or user_message or ""
            goal = self._rewrite_goal(goal)
            self.session.last_goal = goal
            self.session.tool_mode = True

        if not goal.strip():
            yield self._status("❌ No navigation prompt provided.", done=True)
            return

        # === Start navigation ===
        yield self._status(f"🔧 Using model: {self.MODEL_NAME} @ {self.OPENAI_BASE}")
        yield self._status(f"🎯 Goal: {goal}")

        self._step_log.clear()
        self._last_snapshots.clear()

        try:
            with httpx.Client(timeout=httpx.Timeout(60, read=60, connect=30)) as http:
                yield self._status("🛠️ Boot: initializing Playwright session…")

                start_url = self._extract_url(goal)
                self._step_log.append(f"Navigate → {start_url}")
                nav_res = self._call_mcp(http, "browser_navigate", {"url": start_url})
                yield self._status(f"🛠️ navigate → {start_url}")

                snapshot = self._browser_snapshot(http)
                if snapshot:
                    self._last_snapshots.append(snapshot)
                    yield self._status("📖 Snapshot captured (structured YAML).")
                    yield self._status(self._clip(snapshot, 1800))

                b64, pub = self._take_screenshot(http)
                if b64:
                    yield self._image_event(b64)
                    self._step_log.append("Screenshot displayed inline.")
                elif pub:
                    yield self._status(f"👀 Screenshot available at {pub}")

                # === ReAct Loop ===
                for step in range(1, self.MAX_STEPS + 1):
                    obs = self._build_observation(http, snapshot)

                    action = self._decide_next_action(goal, obs)
                    action_json = self._to_json(action)

                    if not action_json or "op" not in action_json:
                        if isinstance(action, str) and "done" in action.lower():
                            yield self._status("✅ Model indicated completion.")
                            break
                        yield self._status("ℹ️ No structured action detected; stopping.")
                        break

                    op = action_json.get("op")
                    if op in ("done", "finish", "stop"):
                        reason = action_json.get("reason", "No reason provided.")
                        yield self._status(f"✅ Done: {reason}")
                        break

                    nice = self._pretty_action(op, action_json)
                    self._step_log.append(nice)
                    yield self._status(f"🛠️ Step {step}: {nice}")

                    exec_ok, exec_msg = self._exec_op(http, op, action_json)
                    if not exec_ok:
                        self.session.last_error = exec_msg
                        self._step_log.append(f"⚠️ Action failed → {exec_msg}")
                        yield self._status(f"⚠️ Action failed → {exec_msg}")
                    else:
                        self._step_log.append(f"Executed: {op}")

                    snapshot = self._browser_snapshot(http)
                    if snapshot:
                        yield self._status("📖 Snapshot updated.")
                        yield self._status(self._clip(snapshot, 1800))
                        self._last_snapshots.append(snapshot)
                        self.session.last_snapshot = snapshot

                    b64, pub = self._take_screenshot(http)
                    if b64:
                        yield self._image_event(b64)
                        self._step_log.append("Screenshot displayed inline.")
                    elif pub:
                        yield self._status(f"👀 Screenshot available at {pub}")

                    if self._is_stuck():
                        yield self._status("🤔 Page state unchanged — stopping.")
                        break

                final_summary = self._final_summary(goal, self._step_log, snapshot)
                yield self._status("📦 Preparing final answer…")
                yield final_summary
                yield self._status("✅ Finished.", done=True)

        except Exception as e:
            if self.debug:
                import traceback; traceback.print_exc()
            self.session.last_error = str(e)
            yield self._status(f"💥 Pipeline error: {e}", done=True)

    # ----------------------------------------------------------
    # Fixed exec_op
    # ----------------------------------------------------------
    def _exec_op(self, http, op: str, args: Dict) -> (bool, str):
        """Execute Playwright MCP operation with schema-correct payloads."""
        try:
            if op == "click":
                if args.get("ref"):
                    element = {"ref": args["ref"]}
                    for k, v in args.items():
                        if k not in ("op", "ref"):
                            element[k] = v
                    payload = {"element": element}
                elif args.get("selector"):
                    payload = {"selector": args.get("selector")}
                else:
                    return False, "click missing ref/selector"
                r = http.post(f"{self.MCPO_BASE_URL}/browser_click", json=payload)

            elif op == "type":
                if args.get("ref"):
                    payload = {"element": {"ref": args["ref"]}, "text": args.get("text", "")}
                elif args.get("selector"):
                    payload = {"selector": args["selector"], "text": args.get("text", "")}
                else:
                    return False, "type missing ref/selector"
                r = http.post(f"{self.MCPO_BASE_URL}/browser_type", json=payload)

            elif op == "fill_form":
                r = http.post(f"{self.MCPO_BASE_URL}/browser_fill_form", json=args)

            elif op == "press":
                key = args.get("key", "Enter")
                r = http.post(f"{self.MCPO_BASE_URL}/browser_press_key", json={"key": key})

            elif op == "hover":
                if args.get("ref"):
                    payload = {"element": {"ref": args["ref"]}}
                elif args.get("selector"):
                    payload = {"selector": args["selector"]}
                else:
                    return False, "hover missing ref/selector"
                r = http.post(f"{self.MCPO_BASE_URL}/browser_hover", json=payload)

            elif op == "wait":
                ms = int(args.get("ms", 1000))
                time.sleep(ms / 1000.0)
                return True, f"waited {ms}ms"

            elif op == "navigate":
                r = http.post(f"{self.MCPO_BASE_URL}/browser_navigate", json={"url": args.get("url")})

            else:
                return False, f"unknown op: {op}"

            if r.status_code >= 400:
                msg = r.text[:200]
                if "422" in msg or "Field required" in msg:
                    msg += " (💡 missing element field)"
                return False, f"{op}: HTTP {r.status_code} {msg}"
            return True, r.text[:200]

        except Exception as e:
            return False, f"{op} error: {e}"
