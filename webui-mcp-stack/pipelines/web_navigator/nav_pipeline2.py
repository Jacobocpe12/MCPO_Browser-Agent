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
        # You can override via environment:
        #   OPENAI_API_KEY, OPENAI_BASE, MODEL_NAME
        self.OPENAI_BASE = os.getenv("OPENAI_BASE", "https://ollama.gpu.lfi.rwth-aachen.de/api")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "azure.gpt-4o-sweden")

        # === Loop & I/O ===
        self.LOCAL_IMG_DIR = "/tmp/playwright-output"
        os.makedirs(self.LOCAL_IMG_DIR, exist_ok=True)
        self.MAX_STEPS = 12
        self.NO_PROGRESS_LIMIT = 3  # if snapshot unchanged N times → ask user / stop
        self.SCREENSHOT_PUBLIC_BASE = "http://91.99.79.208:3888"

        # Track for summary
        self._step_log: List[str] = []
        self._last_snapshots: List[str] = []

    # ----------------------------------------------------------
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Pre-run hook (can modify body if needed)
        if self.debug:
            print("[nav] inlet keys:", list((body or {}).keys()))
        return body

    # ----------------------------------------------------------
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:

        goal = self._extract_prompt(messages) or user_message or ""
        if not goal.strip():
            yield self._status("❌ No navigation prompt provided.", done=True)
            return

        # Intro
        yield self._status(f"🔧 Using model: {self.MODEL_NAME} @ {self.OPENAI_BASE}")
        yield self._status(f"🎯 Goal: {goal}")

        self._step_log.clear()
        self._last_snapshots.clear()

        try:
            with httpx.Client(timeout=httpx.Timeout(60, read=60, connect=30)) as http:
                # Boot
                yield self._status("🛠️ Boot: initializing Playwright session…")

                # Step 0 → Navigate (based on goal)
                start_url = self._extract_url(goal)
                self._step_log.append(f"Navigate → {start_url}")
                nav_res = self._call_mcp(http, "browser_navigate", {"url": start_url})
                yield self._status(f"🛠️ navigate → {start_url}")

                # Observe initial state
                snapshot = self._browser_snapshot(http)
                if snapshot:
                    self._last_snapshots.append(snapshot)
                    yield self._status("📖 Snapshot captured (structured YAML).")
                    yield self._status(self._clip(snapshot, 1800))

                b64, pub = self._take_screenshot(http)
                if pub:
                    if b64:
                        yield self._image_event(b64)
                    self._step_log.append(f"Screenshot → {pub}")
                    yield self._status(f"👀 Screenshot saved to {pub}")

                # === ReAct loop ===
                for step in range(1, self.MAX_STEPS + 1):
                    # Build observation text
                    obs = self._build_observation(http, snapshot)

                    # Ask model what to do next
                    action = self._decide_next_action(goal, obs)
                    if self.debug:
                        print("[nav] action raw:", action)

                    # Try parse JSON action
                    action_json = self._to_json(action)
                    if not action_json or "op" not in action_json:
                        # If model thinks we’re done, allow it to stop with a message
                        if isinstance(action, str) and "done" in action.lower():
                            self._step_log.append("Model indicated completion.")
                            yield self._status("✅ Model indicated completion.", done=False)
                            break
                        # Otherwise stop with info
                        yield self._status("ℹ️ No structured action detected; stopping loop.", done=False)
                        break

                    op = action_json.get("op")
                    # End condition op
                    if op in ("done", "finish", "stop"):
                        reason = action_json.get("reason") or "No reason provided."
                        self._step_log.append(f"Done: {reason}")
                        yield self._status(f"✅ Done: {reason}", done=False)
                        break

                    # Narrate action
                    nice = self._pretty_action(op, action_json)
                    self._step_log.append(nice)
                    yield self._status(f"🛠️ Step {step}: {nice}")

                    # Execute
                    exec_ok, exec_msg = self._exec_op(http, op, action_json)
                    if not exec_ok:
                        self._step_log.append(f"Action failed → {exec_msg}")
                        yield self._status(f"⚠️ Action failed → {exec_msg}", done=False)
                        # Let the loop continue; model can react to failure via new observation
                    else:
                        self._step_log.append(f"Executed: {op}")

                    # Observe new state
                    snapshot = self._browser_snapshot(http)
                    if snapshot:
                        yield self._status("📖 Snapshot updated.")
                        yield self._status(self._clip(snapshot, 1800))
                        self._last_snapshots.append(snapshot)

                    b64, pub = self._take_screenshot(http)
                    if pub:
                        if b64:
                            yield self._image_event(b64)
                        self._step_log.append(f"Screenshot → {pub}")
                        yield self._status(f"👀 Screenshot saved to {pub}")

                    # Stuck detection
                    if self._is_stuck():
                        self._step_log.append("Detected no progress across observations.")
                        yield self._status("🤔 Page state unchanged for several steps — stopping.", done=False)
                        break

                # === Final summary ===
                final_summary = self._final_summary(goal, self._step_log, snapshot)
                yield self._status("📦 Preparing final answer…", done=False)
                # Emit as assistant text so it remains after events
                yield final_summary
                yield self._status("✅ Finished.", done=True)

        except Exception as e:
            if self.debug:
                import traceback; traceback.print_exc()
            yield self._status(f"💥 Pipeline error: {e}", done=True)

    # ----------------------------------------------------------
    # Helpers: Prompt, URL, Observation, Stuck
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
        # full URL present?
        m = re.search(r"(https?://[^\s]+)", text)
        if m:
            return m.group(1)

        # domain-like (example.de, example.com)
        m = re.search(r"\b([\w-]+\.(com|org|net|edu|gov|info|io|ai|de|nl|fr|es|it|ch))\b", text, re.I)
        if m:
            return f"https://{m.group(1)}"

        # keyword fallback → www.<keyword>.com
        tokens = re.findall(r"[a-zA-Z0-9\-]+", text)
        if tokens:
            return f"https://www.{tokens[0].lower()}.com"

        return "https://example.com"

    def _build_observation(self, http, snapshot_text: Optional[str]) -> str:
        parts = []
        # Page URL + Title (lightweight query)
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/browser_evaluate", json={"script": "({url: location.href, title: document.title})"})
            if r.status_code == 200:
                data = r.json().get("result", {})
                parts.append(f"URL: {data.get('url','?')}")
                parts.append(f"TITLE: {data.get('title','?')}")
        except Exception:
            pass

        # Visible text (truncated) for context
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/browser_evaluate", json={"script": "document.body.innerText.slice(0,4000)"})
            if r.status_code == 200:
                txt = r.json().get("result", "") or ""
                parts.append("VISIBLE_TEXT:")
                parts.append(self._clip(txt, 1200))
        except Exception:
            pass

        # Structured snapshot
        if snapshot_text:
            parts.append("STRUCTURED SNAPSHOT:")
            parts.append(self._clip(snapshot_text, 4000))

        return "\n".join(parts)

    def _is_stuck(self) -> bool:
        if len(self._last_snapshots) < self.NO_PROGRESS_LIMIT:
            return False
        tail = self._last_snapshots[-self.NO_PROGRESS_LIMIT:]
        # crude identical check
        return all(s.strip() == tail[0].strip() for s in tail)

    # ----------------------------------------------------------
    # MCP calls: generic + snapshot + screenshot + exec
    def _call_mcp(self, http, endpoint: str, payload: dict = None):
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/{endpoint}", json=payload or {})
            r.raise_for_status()
            data = r.json()
            return data.get("result") or data
        except Exception as e:
            return {"error": str(e)}

    def _browser_snapshot(self, http) -> Optional[str]:
        try:
            r = http.post(f"{self.MCPO_BASE_URL}/browser_snapshot")
            r.raise_for_status()
            data = r.json()
            result = data.get("result") if isinstance(data, dict) else str(data)
            # If JSON-ish, pretty YAML it for LLM
            if isinstance(result, str) and (result.strip().startswith("{") or result.strip().startswith("[")):
                result = yaml.dump(json.loads(result), sort_keys=False, allow_unicode=True)
            return result
        except Exception as e:
            return f"[snapshot error: {e}]"

    def _take_screenshot(self, http):
        try:
            filename = f"page-{time.strftime('%Y%m%dT%H%M%S')}.png"
            r = http.post(
                f"{self.MCPO_BASE_URL}/browser_take_screenshot",
                json={"fullPage": True, "type": "png", "filename": f"/tmp/playwright-output/{filename}"}
            )
            r.raise_for_status()
            url = f"{self.SCREENSHOT_PUBLIC_BASE}/{filename}"
            # Try to open local path to embed as base64 (if volume is shared)
            local_path = os.path.join(self.LOCAL_IMG_DIR, filename)
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    return b64, url
            # If not accessible locally, still return URL
            return None, url
        except Exception as e:
            return None, f"[screenshot error: {e}]"

    def _exec_op(self, http, op: str, args: Dict) -> (bool, str):
        try:
            if op == "click":
                payload = {"ref": args.get("ref")} if args.get("ref") else {"selector": args.get("selector")}
                r = http.post(f"{self.MCPO_BASE_URL}/browser_click", json=payload)
            elif op == "type":
                payload = {"ref": args.get("ref"), "text": args.get("text", "")}
                if not payload.get("ref") and not args.get("selector"):
                    return False, "missing ref/selector"
                if not payload.get("ref"):
                    payload = {"selector": args.get("selector"), "text": args.get("text", "")}
                r = http.post(f"{self.MCPO_BASE_URL}/browser_type", json=payload)
            elif op == "fill_form":
                r = http.post(f"{self.MCPO_BASE_URL}/browser_fill_form", json=args)
            elif op == "press":
                r = http.post(f"{self.MCPO_BASE_URL}/browser_press_key", json={"key": args.get("key", "Enter")})
            elif op == "hover":
                payload = {"ref": args.get("ref")} if args.get("ref") else {"selector": args.get("selector")}
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
                return False, f"{op}: HTTP {r.status_code} {r.text[:200]}"
            return True, r.text[:200]
        except Exception as e:
            return False, f"{op} error: {e}"

    # ----------------------------------------------------------
    # LLM calls: decide next action + final summary
    def _decide_next_action(self, goal: str, observation: str) -> str:
        """
        Returns either:
          - JSON string like {"op":"click","ref":"e14"} / {"op":"type","ref":"e22","text":"Aachen"}
          - or a natural language statement (we handle missing JSON)
        """
        sys = (
            "You are a web navigation reasoning engine.\n"
            "You receive STRUCTURED SNAPSHOT (with [ref=e##]) and visible text.\n"
            "Choose ONE next action in JSON when possible.\n"
            "Allowed ops: navigate(url), click(ref|selector), type(ref|selector,text), "
            "press(key), hover(ref|selector), wait(ms), done(reason).\n"
            "Return ONLY JSON when you can. If already done, use {\"op\":\"done\",\"reason\":\"...\"}."
        )
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Goal: {goal}\n\n{observation}"},
        ]
        return self._chat(messages)

    def _final_summary(self, goal: str, steps: List[str], last_snapshot: Optional[str]) -> str:
        # Try LLM summary; if it fails, fallback to deterministic summary.
        bullets = "\n".join(f"- {s}" for s in steps[-12:])
        prompt = (
            "Summarize the browsing session clearly for the user. "
            "Explain what was attempted, what worked, and the final state. "
            "Keep it concise and actionable.\n\n"
            f"Goal: {goal}\nRecent steps:\n{bullets}\n"
            "If a screenshot URL is mentioned, include it as the result link."
        )
        messages = [
            {"role": "system", "content": "You write clear, concise summaries for non-technical users."},
            {"role": "user", "content": prompt},
        ]
        summary = self._chat(messages, expect_json=False)
        if summary.startswith("❌"):
            # Fallback
            final = ["### Result",
                     f"- Goal: {goal}",
                     f"- Steps executed: {len(steps)}",
                     "- Recent actions:"]
            final += [f"  {s}" for s in steps[-6:]]
            return "\n".join(final)
        return summary

    def _chat(self, messages: List[Dict], expect_json: bool = False) -> str:
        payload = {
            "model": self.MODEL_NAME,
            "messages": messages,
            "stream": False,
        }
        headers = {}
        if self.OPENAI_API_KEY:
            headers["Authorization"] = f"Bearer {self.OPENAI_API_KEY}"

        try:
            # Some OpenAI-compatible servers use /api/chat/completions (OpenWebUI),
            # others use /v1/chat/completions. Try /v1 first, then fallback.
            url1 = f"{self.OPENAI_BASE.rstrip('/')}/v1/chat/completions"
            r = httpx.post(url1, json=payload, headers=headers, timeout=90)
            if r.status_code == 404:
                url2 = f"{self.OPENAI_BASE.rstrip('/')}/api/chat/completions"
                r = httpx.post(url2, json=payload, headers=headers, timeout=90)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ LLM error: {e}"

    # ----------------------------------------------------------
    # Utilities
    def _to_json(self, s: str) -> Optional[Dict]:
        if not isinstance(s, str):
            return None
        # Extract first {...} block
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            pass
        return None

    def _pretty_action(self, op: str, a: Dict) -> str:
        if op == "navigate":
            return f"navigate → {a.get('url','')}"
        if op == "click":
            target = a.get("ref") or a.get("selector") or "?"
            return f"click → {target}"
        if op == "type":
            target = a.get("ref") or a.get("selector") or "?"
            text = a.get("text","")
            return f"type → {target} = '{text}'"
        if op == "press":
            return f"press → {a.get('key','Enter')}"
        if op == "hover":
            target = a.get("ref") or a.get("selector") or "?"
            return f"hover → {target}"
        if op == "wait":
            return f"wait → {a.get('ms',1000)}ms"
        return f"{op}"

    def _clip(self, text: str, n: int) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text if len(text) <= n else text[:n] + "\n..."

    def _status(self, description: str, done: bool = False) -> Dict:
        return {"event": {"type": "status", "data": {"description": description, "done": done}}}

    def _image_event(self, b64img: str) -> Dict:
        return {"event": {"type": "image", "data": {"mime_type": "image/png", "base64": b64img}}}
