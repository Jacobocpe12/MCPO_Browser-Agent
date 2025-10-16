"""Launch MCPO with config-only startup and local patches applied."""

from __future__ import annotations

import os
import secrets
import shutil
import sys

from mcpo import app
from mcpo.main import main
from mcpo.patch_main import patch_mcpo_app

CONFIG_PATH = "/config/config.json"
DEFAULT_CONFIG = "/app/mcpo/default_config.json"
TOOL_API_KEY_ENV = "TOOL_API_KEY"
TOOL_API_KEY_FILE = os.getenv("TOOL_API_KEY_FILE", "/config/tool_api_key")


def ensure_config() -> None:
    """Make sure the runtime config file exists before launching MCPO."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        shutil.copy(DEFAULT_CONFIG, CONFIG_PATH)
        print(
            f"[launch_mcpo] Created default config at {CONFIG_PATH}",
            flush=True,
        )
    else:
        print(
            f"[launch_mcpo] Using existing config at {CONFIG_PATH}",
            flush=True,
        )


def ensure_api_key() -> None:
    """Ensure a persistent tool API key is available on disk."""

    provided = os.getenv(TOOL_API_KEY_ENV)
    if provided:
        os.makedirs(os.path.dirname(TOOL_API_KEY_FILE), exist_ok=True)
        with open(TOOL_API_KEY_FILE, "w", encoding="utf-8") as handle:
            handle.write(provided.strip())
        print(
            f"[launch_mcpo] Stored TOOL_API_KEY from environment at {TOOL_API_KEY_FILE}",
            flush=True,
        )
        return

    if os.path.exists(TOOL_API_KEY_FILE):
        print(
            f"[launch_mcpo] Loaded TOOL_API_KEY from {TOOL_API_KEY_FILE}",
            flush=True,
        )
        return

    key = secrets.token_urlsafe(32)
    os.makedirs(os.path.dirname(TOOL_API_KEY_FILE), exist_ok=True)
    with open(TOOL_API_KEY_FILE, "w", encoding="utf-8") as handle:
        handle.write(key)
    try:
        os.chmod(TOOL_API_KEY_FILE, 0o600)
    except PermissionError:
        # Some platforms (e.g. mounted volumes on Windows) may not support chmod.
        pass
    print(
        f"[launch_mcpo] Generated TOOL_API_KEY at {TOOL_API_KEY_FILE}",
        flush=True,
    )


def run() -> None:
    """Patch the MCPO FastAPI app and start it in config-only mode."""
    ensure_config()
    ensure_api_key()
    patch_mcpo_app(app)
    sys.argv = [
        "mcpo",
        "--config",
        CONFIG_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        "3879",
    ]
    main()


if __name__ == "__main__":
    run()
