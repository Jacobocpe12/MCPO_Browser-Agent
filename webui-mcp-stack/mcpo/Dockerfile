FROM ghcr.io/open-webui/mcpo:latest

# Provide the default config and FastAPI patches used by the stack.
COPY config.json /app/mcpo/default_config.json
COPY tools/public_file.py /app/mcpo/tools/public_file.py
COPY patch_main.py /app/mcpo/patch_main.py
COPY launch_mcpo.py /app/mcpo/launch_mcpo.py
# Ensure config directory and default config file exist for MCPO runtime
RUN mkdir -p /config && cp /app/mcpo/default_config.json /config/config.json

# Base URL used when returning public links.
ENV PUBLIC_URL_BASE=http://shots.choype.com/public

# Launch MCPO with the config-only startup path and patched app.
CMD ["python3", "/app/mcpo/launch_mcpo.py"]
