"""FastAPI router that exposes a /tool/public_file endpoint.

The handler accepts either text or base64-encoded content, persists the
payload in ``/app/public`` and returns a URL that is publicly accessible
through the configured base. Each stored file is scheduled for deletion
four hours after it is written.
"""

from __future__ import annotations

import base64
import os
import secrets
import threading
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

PUBLIC_DIR = "/app/public"
PUBLIC_URL_BASE = os.getenv("PUBLIC_URL_BASE", "https://shots.choype.com/public").rstrip("/")
TOOL_API_KEY_FILE = os.getenv("TOOL_API_KEY_FILE", "/config/tool_api_key")


def _load_api_key() -> str | None:
    key = os.getenv("TOOL_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(TOOL_API_KEY_FILE):
        with open(TOOL_API_KEY_FILE, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return None


def ensure_dir() -> None:
    """Ensure the public directory exists."""
    os.makedirs(PUBLIC_DIR, exist_ok=True)


def _schedule_deletion(path: str) -> None:
    """Schedule deletion of ``path`` four hours after creation."""

    def delete_later() -> None:
        time.sleep(4 * 3600)
        if os.path.exists(path):
            os.remove(path)

    threading.Thread(target=delete_later, daemon=True).start()


def save_public_file(content: str, ext: str = "txt", filename: str | None = None) -> Dict[str, Any]:
    """Persist content to disk and return a public URL payload."""
    ensure_dir()
    name = filename or f"{uuid.uuid4()}.{ext}"
    path = os.path.join(PUBLIC_DIR, name)

    try:
        # Attempt to decode as base64; if it fails treat as plain text.
        try:
            data = base64.b64decode(content, validate=True)
            with open(path, "wb") as file_handle:
                file_handle.write(data)
        except Exception:
            with open(path, "w", encoding="utf-8") as file_handle:
                file_handle.write(content)

        _schedule_deletion(path)
        return {"url": f"{PUBLIC_URL_BASE}/{name}"}
    except Exception as exc:  # pragma: no cover - defensive response shaping
        return {"error": str(exc)}


def get_router() -> APIRouter:
    """Return a router exposing the public_file tool endpoint."""
    router = APIRouter()

    @router.post("/tool/public_file")
    async def public_file_endpoint(request: Request) -> JSONResponse:
        tool_api_key = _load_api_key()
        if tool_api_key:
            supplied = (
                request.headers.get("x-api-key")
                or request.headers.get("x-tool-api-key")
                or request.query_params.get("api_key")
            )
            if not supplied or not secrets.compare_digest(supplied, tool_api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing API key",
                )
        data = await request.json()
        content = data.get("content", "")
        filename = data.get("filename")
        ext = filename.split(".")[-1] if filename and "." in filename else "txt"
        result = save_public_file(content, ext, filename)
        return JSONResponse(result)

    return router
