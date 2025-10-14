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
        self.version = "5.1.0" # Version incremented for fix
        self.author = "You + Gemini"
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

        # Map high-level ops → MCP endpoints (aliases resolved separately)
        self._tool_endpoints = {
            "click": "browser_click",
            "type": "browser_type",
            "navigate": "browser_navigate",
            "scroll": "browser_scroll",
            "hover": "browser_hover",
            "evaluate": "browser_evaluate",
            "close": "browser_close",
            "resize": "browser_resize",
            "snapshot": "browser_snapshot",
            "take_screenshot": "browser_take_screenshot",
            "drag": "browser_drag",
            "select_option": "browser_select_option",
            "mouse_move_xy": "browser_mouse_move_xy",
            "mouse_click_xy": "browser_mouse_click_xy",
            "handle_dialog": "browser_handle_dialog",
            "network_requests": "browser_network_requests",
            "console_messages": "browser_console_messages",
            "press": "browser_press_key",
            "file_upload": "browser_file_upload",
            "fill_form": "browser_fill_form",
        }

        self._op_aliases = {
            "screenshot": "take_screenshot",
            "press_key": "press",
            "mouse_move": "mouse_move_xy",
            "mouse_click": "mouse_click_xy",
            "take_snapshot": "snapshot",
        }

    # ----------------------------------------------------------
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Pre-run hook (can modify body if needed)
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
                nav_payload = self._make_payload("navigate", {"url": start_url})
                nav_res = self._call_mcp(http, "browser_navigate", nav_payload)
                yield self._status(f"🛠️ navigate → {start_url}")

                # First observation
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

                # Final summary
                final_summary = self._final_summary(goal, self._step_log, snapshot)
                yield self._status("📦 Preparing final answer…")
                yield final_summary
                yield self._status("✅ Finished.", done=True)

        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            self.session.last_error = str(e)
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
        url = None
        # full URL present?
        m = re.search(r"(https?://[^\s\"']+)", text) # Avoid trailing quotes
        if m:
            url = m.group(1)
        
        # domain-like (example.de, example.com)
        elif re.search(r"\b([\w-]+\.(com|org|net|edu|gov|info|io|ai|de|nl|fr|es|it|ch))\b", text, re.I):
             m = re.search(r"\b([\w-]+\.(com|org|net|edu|gov|info|io|ai|de|nl|fr|es|it|ch))\b", text, re.I)
             if m:
                url = f"https://{m.group(1)}"
        
        # Fallback to search if no clear URL
        else:
            # Use a search engine for ambiguous queries
            from urllib.parse import quote_plus
            query = re.sub(r'[^a-zA-Z0-9\s-]', '', text).strip()
            url = f"https://www.google.com/search?q={quote_plus(query)}"

        # ✅ FIX: Final sanitization to remove common trailing punctuation.
        if url:
            return url.strip().rstrip('.,;:"\'')
            
        return "https://www.google.com" # Default fallback


    def _build_observation(self, http, snapshot_text: Optional[str]) -> str:
        parts = []
        # Page URL + Title (lightweight query)
        try:
            # ✅ FIX: The API expects the key "expression", not "script".
            payload = {"expression": "({url: location.href, title: document.title})"}
            r = http.post(f"{self.MCPO_BASE_URL}/browser_evaluate", json=payload)
            if r.status_code == 200:
                data = r.json().get("result", {})
                parts.append(f"URL: {data.get('url','?')}")
                parts.append(f"TITLE: {data.get('title','?')}")
        except Exception:
            pass

        # Visible text (truncated) for context
        try:
            # ✅ FIX: The API expects the key "expression", not "script".
            payload = {"expression": "document.body.innerText.slice(0,4000)"}
            r = http.post(f"{self.MCPO_BASE_URL}/browser_evaluate", json=payload)
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
            body = self._sanitize_payload(payload or {})
            r = http.post(f"{self.MCPO_BASE_URL}/{endpoint}", json=body)
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
            screenshot_payload = self._make_payload("take_screenshot", {"fullPage": True})
            r = http.post(
                f"{self.MCPO_BASE_URL}/{self._tool_endpoints['take_screenshot']}",
                json=self._sanitize_payload(screenshot_payload),
            )
            r.raise_for_status()
            result = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
            if isinstance(result, dict):
                b64 = result.get("result") or result.get("base64")
                if b64:
                    return b64, None
                path = result.get("path")
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    return b64, None
            elif isinstance(result, str):
                return result, None
            # Fallback: try to load from shared directory if Playwright saved the file.
            local_path = os.path.join(self.LOCAL_IMG_DIR, filename)
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return b64, None
            return None, None
        except Exception as e:
            return None, f"[screenshot error: {e}]"

    def _exec_op(self, http, op: str, args: Dict) -> (bool, str):
        """Execute Playwright MCP operation with schema-correct payloads."""
        try:
            canonical = self._op_aliases.get(op, op)
            if canonical == "wait":
                ms = int(args.get("ms", 1000))
                time.sleep(ms / 1000.0)
                return True, f"waited {ms}ms"

            endpoint = self._tool_endpoints.get(canonical)
            if not endpoint:
                return False, f"unknown op: {op}"

            payload = self._make_payload(canonical, args)
            body = self._sanitize_payload(payload)

            r = http.post(f"{self.MCPO_BASE_URL}/{endpoint}", json=body)
            if r.status_code >= 400:
                msg = r.text[:250]
                if r.status_code == 422:
                    msg += " (Hint: The request body may not match the API's expected schema.)"
                return False, f"{op}: HTTP {r.status_code} - {msg}"
            return True, r.text[:200]

        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"{op} execution error: {e}"


    def _make_payload(self, op: str, args: Optional[Dict]) -> Dict:
        """Map high-level actions to the Playwright MCP request schema."""
        args = args or {}
        op = self._op_aliases.get(op, op)

        if op in {"click", "hover"}:
            ref = args.get("ref")
            selector = args.get("selector")
            if ref:
                body: Dict[str, Union[str, bool]] = {"ref": ref}
            elif selector:
                body = {"selector": selector}
            else:
                raise ValueError(f"{op} requires 'ref' or 'selector'")
            if op == "click":
                if "doubleClick" in args:
                    body["doubleClick"] = bool(args["doubleClick"])
                if "button" in args:
                    body["button"] = str(args["button"])
            return body

        if op == "type":
            text = args.get("text")
            ref = args.get("ref")
            selector = args.get("selector")
            if not text:
                raise ValueError("type requires 'text'")
            body: Dict[str, Union[str, bool]] = {"text": text}
            if ref:
                body["ref"] = ref
            elif selector:
                body["selector"] = selector
            else:
                raise ValueError("type requires 'ref' or 'selector'")
            if "submit" in args:
                body["submit"] = bool(args["submit"])
            return body

        if op == "press":
            key = args.get("key") or args.get("keys")
            if not key:
                raise ValueError("press requires 'key'")
            body: Dict[str, Union[str, bool]] = {"key": str(key)}
            if args.get("ref"):
                body["ref"] = args["ref"]
            elif args.get("selector"):
                body["selector"] = args["selector"]
            return body

        if op == "navigate":
            url = args.get("url")
            if not url:
                raise ValueError("navigate requires 'url'")
            return {"url": url}

        if op == "take_screenshot":
            body: Dict[str, bool] = {}
            if "fullPage" in args:
                body["fullPage"] = bool(args["fullPage"])
            return body

        if op == "scroll":
            direction = args.get("direction")
            if direction not in {"up", "down"}:
                raise ValueError("scroll requires 'direction' of 'up' or 'down'")
            body: Dict[str, Union[str, bool]] = {"direction": direction}
            if args.get("ref"):
                body["ref"] = args["ref"]
            elif args.get("selector"):
                body["selector"] = args["selector"]
            return body

        if op == "evaluate":
            function = args.get("function")
            if not function:
                raise ValueError("evaluate requires 'function'")
            body: Dict[str, str] = {"function": function}
            if args.get("ref"):
                body["ref"] = args["ref"]
            if args.get("element"):
                body["element"] = args["element"]
            elif args.get("selector"):
                body["element"] = args["selector"]
            return body

        if op == "close":
            return {}

        if op == "resize":
            width = args.get("width")
            height = args.get("height")
            if width is None or height is None:
                raise ValueError("resize requires 'width' and 'height'")
            return {"width": int(width), "height": int(height)}

        if op == "snapshot":
            return {}

        if op == "drag":
            body: Dict[str, str] = {}
            start_ref = args.get("startRef") or args.get("start_ref")
            start_selector = args.get("startSelector") or args.get("start_selector")
            end_ref = args.get("endRef") or args.get("end_ref")
            end_selector = args.get("endSelector") or args.get("end_selector")
            if start_ref:
                body["startRef"] = start_ref
            elif start_selector:
                body["startSelector"] = start_selector
            else:
                raise ValueError("drag requires 'startRef' or 'startSelector'")
            if end_ref:
                body["endRef"] = end_ref
            elif end_selector:
                body["endSelector"] = end_selector
            else:
                raise ValueError("drag requires 'endRef' or 'endSelector'")
            return body

        if op == "select_option":
            values = args.get("values") or args.get("value")
            if isinstance(values, str):
                values = [values]
            if not values or not isinstance(values, (list, tuple)):
                raise ValueError("select_option requires 'values' list")
            body: Dict[str, Union[str, List[str]]] = {"values": list(values)}
            if args.get("ref"):
                body["ref"] = args["ref"]
            elif args.get("selector"):
                body["selector"] = args["selector"]
            else:
                raise ValueError("select_option requires 'ref' or 'selector'")
            return body

        if op == "mouse_move_xy":
            x = args.get("x")
            y = args.get("y")
            if x is None or y is None:
                raise ValueError("mouse_move requires 'x' and 'y'")
            return {"x": float(x), "y": float(y)}

        if op == "mouse_click_xy":
            x = args.get("x")
            y = args.get("y")
            if x is None or y is None:
                raise ValueError("mouse_click requires 'x' and 'y'")
            body: Dict[str, Union[float, str]] = {"x": float(x), "y": float(y)}
            if args.get("button"):
                body["button"] = str(args["button"])
            return body

        if op == "handle_dialog":
            if "accept" not in args:
                raise ValueError("handle_dialog requires 'accept'")
            return {"accept": bool(args["accept"])}

        if op == "network_requests":
            return {}

        if op == "console_messages":
            return {}

        if op == "file_upload":
            paths = args.get("paths") or args.get("path")
            if isinstance(paths, str):
                paths = [paths]
            if not paths or not isinstance(paths, (list, tuple)):
                raise ValueError("file_upload requires 'paths' list")
            return {"paths": list(paths)}

        if op == "fill_form":
            fields = args.get("fields") if isinstance(args, dict) else args
            if not isinstance(fields, list):
                raise ValueError("fill_form requires 'fields' list")
            return {"fields": fields}

        raise ValueError(f"Unsupported payload request for op '{op}'")

    def _sanitize_payload(self, body: Optional[Dict]) -> Dict:
        body = dict(body or {})
        if "element" in body and isinstance(body["element"], dict) and "ref" in body["element"]:
            body["ref"] = body["element"]["ref"]
            del body["element"]
        elif "element" in body and isinstance(body["element"], str):
            body["selector"] = body["element"]
            del body["element"]
        return body


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
            "Allowed ops: navigate(url), click(ref|selector,doubleClick?,button?), type(ref|selector,text,submit?), "
            "press(key,ref|selector?), hover(ref|selector), scroll(direction,ref?), wait(ms), "
            "take_screenshot(fullPage?), snapshot(), evaluate(function,ref?|element?), drag(startRef/startSelector,endRef/endSelector), "
            "select_option(ref|selector,values), mouse_move(x,y), mouse_click(x,y,button?), handle_dialog(accept), "
            "file_upload(paths), fill_form(fields), resize(width,height), close(), network_requests(), console_messages(), done(reason).\n"
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
        canonical = self._op_aliases.get(op, op)
        if canonical == "navigate":
            return f"navigate → {a.get('url','')}"
        if canonical == "click":
            target = a.get("ref") or a.get("selector") or "?"
            dbl = " (double)" if a.get("doubleClick") else ""
            button = f" [{a.get('button')}]" if a.get("button") else ""
            return f"click{dbl}{button} → {target}"
        if canonical == "type":
            target = a.get("ref") or a.get("selector") or "?"
            text = a.get("text","")
            submit = " (submit)" if a.get("submit") else ""
            return f"type{submit} → {target} = '{text}'"
        if canonical == "press":
            target = a.get("ref") or a.get("selector")
            scope = f" on {target}" if target else ""
            return f"press → {a.get('key','Enter')}{scope}"
        if canonical == "hover":
            target = a.get("ref") or a.get("selector") or "?"
            return f"hover → {target}"
        if canonical == "scroll":
            direction = a.get("direction", "?")
            target = a.get("ref") or a.get("selector")
            scope = f" around {target}" if target else ""
            return f"scroll {direction}{scope}"
        if canonical == "take_screenshot":
            mode = " fullPage" if a.get("fullPage") else ""
            return f"screenshot{mode}"
        if canonical == "snapshot":
            return "snapshot"
        if canonical == "evaluate":
            return "evaluate JS"
        if canonical == "drag":
            start = a.get("startRef") or a.get("startSelector") or "?"
            end = a.get("endRef") or a.get("endSelector") or "?"
            return f"drag {start} → {end}"
        if canonical == "select_option":
            target = a.get("ref") or a.get("selector") or "?"
            values = ",".join(a.get("values", [])) if isinstance(a.get("values"), list) else a.get("values", "")
            return f"select_option → {target} = [{values}]"
        if canonical == "mouse_move_xy":
            return f"mouse_move → ({a.get('x')},{a.get('y')})"
        if canonical == "mouse_click_xy":
            button = f" [{a.get('button')}]" if a.get("button") else ""
            return f"mouse_click{button} → ({a.get('x')},{a.get('y')})"
        if canonical == "handle_dialog":
            return "handle_dialog → accept" if a.get("accept") else "handle_dialog → dismiss"
        if canonical == "file_upload":
            return f"file_upload → {len(a.get('paths', []))} file(s)"
        if canonical == "fill_form":
            return f"fill_form → {len(a.get('fields', []))} field(s)"
        if canonical == "resize":
            return f"resize → {a.get('width')}x{a.get('height')}"
        if canonical == "close":
            return "close browser"
        if canonical == "network_requests":
            return "fetch network requests"
        if canonical == "console_messages":
            return "fetch console messages"
        if canonical == "wait":
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

    # ----------------------------------------------------------
    # Intent + goal helpers
    def _infer_intent(self, message: Optional[str]) -> str:
        """Heuristically detect whether a prompt is small-talk or navigation work."""

        if not message:
            return "tool"

        text = message.strip().lower()
        if not text:
            return "tool"

        if "http://" in text or "https://" in text:
            return "tool"

        nav_keywords = (
            "navigate",
            "open",
            "search",
            "browser",
            "website",
            "page",
            "click",
            "scroll",
            "type",
            "fill",
            "visit",
            "go to",
        )
        if any(kw in text for kw in nav_keywords):
            return "tool"

        chat_markers = ("hi", "hello", "hey", "thank", "thanks", "how are")
        if any(text.startswith(marker) for marker in chat_markers):
            return "chat"

        if text.endswith("?") and not any(kw in text for kw in nav_keywords):
            return "chat"

        if len(text.split()) <= 6 and not any(kw in text for kw in nav_keywords):
            return "chat"

        return "tool"

    def _rewrite_goal(self, goal: str) -> str:
        """Normalize user instructions and append context from the active session."""

        cleaned = (goal or "").strip()
        if not cleaned:
            return ""

        # Collapse internal whitespace so the goal stays compact for the LLM call.
        cleaned = re.sub(r"\s+", " ", cleaned)

        session = getattr(self, "session", None)
        context_bits: List[str] = []
        if session:
            last_error = getattr(session, "last_error", None)
            if last_error:
                context_bits.append(f"Previous error to avoid: {last_error}")

            last_snapshot = getattr(session, "last_snapshot", None)
            if last_snapshot:
                context_bits.append("Continue from the current page state; avoid reloading unnecessarily.")

        if context_bits:
            cleaned = f"{cleaned}. " + " ".join(context_bits)

        return cleaned
