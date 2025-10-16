"""Launch MCPO with config-only startup and local patches applied."""

from __future__ import annotations

import os
import shutil
import sys

from mcpo import app
from mcpo.main import main
from mcpo.patch_main import patch_mcpo_app

CONFIG_PATH = "/config/config.json"
DEFAULT_CONFIG = "/app/mcpo/default_config.json"


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


def run() -> None:
    """Patch the MCPO FastAPI app and start it in config-only mode."""
    ensure_config()
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
