"""Patch helpers for extending the MCPO FastAPI application."""

from __future__ import annotations

import os

from fastapi.staticfiles import StaticFiles

from .tools.public_file import get_router


def patch_mcpo_app(app):
    """Attach the public_file router and static /public mount to ``app``."""
    app.include_router(get_router())
    public_dir = "/app/public"
    if not os.path.exists(public_dir):
        os.makedirs(public_dir, exist_ok=True)
    app.mount("/public", StaticFiles(directory=public_dir), name="public")
    return app
