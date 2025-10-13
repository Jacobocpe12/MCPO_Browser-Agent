FROM ghcr.io/open-webui/mcpo:latest

# Copy the additional FastAPI tooling into the image.
COPY tools/public_file.py /app/mcpo/tools/public_file.py
COPY patch_main.py /app/mcpo/patch_main.py

# Base URL used when returning public links.
ENV PUBLIC_URL_BASE=http://shots.choype.com/public

# Patch the MCPO app before starting the default main entry point with the
# expected runtime arguments.
CMD python3 - <<'PY'
import sys
from mcpo.main import main
from mcpo import app
from mcpo.patch_main import patch_mcpo_app

patch_mcpo_app(app)
sys.argv = [
    "mcpo",
    "--config",
    "/config/config.json",
    "--host",
    "0.0.0.0",
    "--port",
    "3879",
    "--server-type",
    "streamable-http",
]
main()
sleep 500
PY
