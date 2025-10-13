"""Web Navigator pipeline for OpenWebUI's pipelines service.

This pipeline is designed to forward natural language navigation
instructions to the MCPO hub that orchestrates the Playwright MCP
browser server.  It streams status updates and screenshots back to the
OpenWebUI chat via pipeline events so operators can watch the browsing
session in real time.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import httpx

from open_webui.pipelines import Context, Event, Pipeline, Valve



@dataclass
class _RunResult:
    """Container for MCPO run metadata returned by the hub."""

    run_id: str
    status: str
    finished: bool


class WebNavigator(Pipeline):
    """Pipeline that brokers requests between OpenWebUI and MCPO.

    The pipeline exposes a small set of tunable valves so that the
    target MCPO endpoint, polling cadence, and maximum wait can be
    adjusted from the OpenWebUI Pipelines admin screen without editing
    this file.
    """

    def __init__(self) -> None:
        super().__init__(
            name="web_navigator",
            description=(
                "Drive the Playwright MCP through MCPO to complete "
                "multi-step navigation tasks with streamed screenshots."
            ),
            version="1.0.0",
            valves={
                "mcpo_base_url": Valve(
                    key="MCPO_BASE_URL",
                    prompt="Base URL for the MCPO container",
                    default="http://mcpo:3879",
                ),
                "poll_interval": Valve(
                    key="WEB_NAVIGATOR_POLL_INTERVAL",
                    prompt="Seconds to wait between MCPO status checks",
                    default=2.0,
                    minimum=0.5,
                    maximum=30.0,
                ),
                "max_wait": Valve(
                    key="WEB_NAVIGATOR_MAX_WAIT",
                    prompt="Maximum seconds to wait for MCPO to finish",
                    default=900.0,
                    minimum=30.0,
                    maximum=3600.0,
                ),
            },
        )

    async def on_start(self, ctx: Context) -> None:
        """Emit a short status message when the pipeline boots."""

        await ctx.emit(
            Event(
                type="status",
                data={"message": "✅ web_navigator pipeline ready"},
            )
        )

    async def on_run(self, ctx: Context) -> None:
        """Handle a single pipeline run initiated from OpenWebUI."""

        instruction = self._extract_instruction(ctx)
        if not instruction:
            await ctx.emit(
                Event(
                    type="status",
                    data={
                        "message": (
                            "No navigation instructions supplied. "
                            "Send a prompt describing the browsing task."
                        )
                    },
                )
            )
            return

        await ctx.emit(
            Event(
                type="status",
                data={
                    "message": "Starting browser navigation via MCPO…",
                    "instruction": instruction,
                },
            )
        )

        async with httpx.AsyncClient(timeout=None) as client:
            run = await self._start_mcpo_run(ctx, client, instruction)
            if not run:
                return

            await self._stream_run(ctx, client, run)

    def _extract_instruction(self, ctx: Context) -> Optional[str]:
        """Extract the latest user instruction from the incoming payload."""

        payload: Dict[str, Any] = ctx.data or {}
        messages: Iterable[Dict[str, Any]] = payload.get("messages", [])
        instruction: Optional[str] = None

        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, list):
                    # OpenWebUI may send structured content blocks.
                    text_blocks = [
                        block.get("text")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    instruction = "\n".join(filter(None, text_blocks)) or instruction
                elif isinstance(content, str):
                    instruction = content

        # Fall back to plain "prompt" or "query" keys if provided.
        if not instruction:
            for key in ("prompt", "query", "input"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    instruction = value
                    break

        return instruction.strip() if isinstance(instruction, str) else None

    async def _start_mcpo_run(
        self, ctx: Context, client: httpx.AsyncClient, instruction: str
    ) -> Optional[_RunResult]:
        """Send the navigation request to MCPO and parse the run identifier."""

        mcpo_url = self.valves["mcpo_base_url"].get(ctx)
        payload = {"prompt": instruction, "mode": "stream"}

        try:
            response = await client.post(f"{mcpo_url}/v1/run", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - runtime safety
            await ctx.emit(
                Event(
                    type="error",
                    data={
                        "message": "Failed to start MCPO navigation run",
                        "details": str(exc),
                    },
                )
            )
            return None

        data = response.json()
        run_id = data.get("run_id") or data.get("id")
        status = data.get("status", "unknown")

        if not run_id:
            await ctx.emit(
                Event(
                    type="error",
                    data={
                        "message": "MCPO response missing run identifier",
                        "details": data,
                    },
                )
            )
            return None

        return _RunResult(run_id=run_id, status=status, finished=False)

    async def _stream_run(
        self, ctx: Context, client: httpx.AsyncClient, run: _RunResult
    ) -> None:
        """Poll MCPO for progress updates until the run completes."""

        mcpo_url = self.valves["mcpo_base_url"].get(ctx)
        poll_interval = float(self.valves["poll_interval"].get(ctx))
        max_wait = float(self.valves["max_wait"].get(ctx))

        started_at = time.monotonic()
        await ctx.emit(
            Event(
                type="status",
                data={"message": f"Tracking MCPO run {run.run_id}"},
            )
        )

        while not run.finished:
            if time.monotonic() - started_at > max_wait:
                await ctx.emit(
                    Event(
                        type="error",
                        data={
                            "message": "MCPO navigation timed out",
                            "run_id": run.run_id,
                        },
                    )
                )
                return

            await self._fetch_progress(ctx, client, mcpo_url, run)

            if run.finished:
                break

            await ctx.sleep(poll_interval)

        await ctx.emit(
            Event(
                type="status",
                data={
                    "message": "✅ Finished navigation.",
                    "run_id": run.run_id,
                },
            )
        )

    async def _fetch_progress(
        self,
        ctx: Context,
        client: httpx.AsyncClient,
        mcpo_url: str,
        run: _RunResult,
    ) -> None:
        """Fetch the latest MCPO run information and stream events."""

        try:
            response = await client.get(f"{mcpo_url}/v1/run/{run.run_id}")
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - runtime safety
            await ctx.emit(
                Event(
                    type="error",
                    data={
                        "message": "Unable to poll MCPO status",
                        "details": str(exc),
                        "run_id": run.run_id,
                    },
                )
            )
            return

        payload = response.json()
        run.status = payload.get("status", run.status)
        run.finished = payload.get("finished", run.finished)

        for item in payload.get("events", []):
            event_type = item.get("type")
            if event_type == "log":
                await ctx.emit(
                    Event(
                        type="status",
                        data={
                            "message": item.get("message", ""),
                            "run_id": run.run_id,
                        },
                    )
                )
            elif event_type == "screenshot":
                await self._emit_screenshot(ctx, item, run.run_id)
            elif event_type == "action":
                await ctx.emit(
                    Event(
                        type="status",
                        data={
                            "message": item.get("description", ""),
                            "run_id": run.run_id,
                        },
                    )
                )

    async def _emit_screenshot(
        self, ctx: Context, item: Dict[str, Any], run_id: str
    ) -> None:
        """Send screenshot data back to OpenWebUI as an image event."""

        path = item.get("path")
        image_bytes: Optional[bytes] = None

        if path and os.path.exists(path):
            with open(path, "rb") as handle:
                image_bytes = handle.read()
        elif item.get("data"):
            try:
                image_bytes = base64.b64decode(item["data"])  # pragma: no cover
            except Exception:  # pragma: no cover - defensive decoding
                image_bytes = None

        if not image_bytes:
            await ctx.emit(
                Event(
                    type="status",
                    data={
                        "message": "Received MCPO screenshot event without data",
                        "run_id": run_id,
                    },
                )
            )
            return

        await ctx.emit(
            Event(
                type="image",
                data={
                    "mime_type": "image/png",
                    "base64": base64.b64encode(image_bytes).decode("utf-8"),
                    "run_id": run_id,
                },
            )
        )


def pipeline() -> Pipeline:
    """Pipeline entrypoint used by the OpenWebUI runtime."""

    return WebNavigator()
