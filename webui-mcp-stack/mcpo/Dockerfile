FROM node:22-slim AS ocr_builder

WORKDIR /opt/openai-ocr-mcp

COPY openai-ocr-mcp/package.json openai-ocr-mcp/package-lock.json ./
RUN npm ci

COPY openai-ocr-mcp/tsconfig.json ./
COPY openai-ocr-mcp/src ./src
RUN npm run build
RUN npm prune --omit=dev

FROM ghcr.io/open-webui/mcpo:latest

# Include Node runtime and the compiled OCR MCP server inside the MCPO image.
COPY --from=ocr_builder /usr/local /usr/local
COPY --from=ocr_builder /opt/openai-ocr-mcp /opt/openai-ocr-mcp

# Provide the default config and FastAPI patches used by the stack.
COPY config.json /app/mcpo/default_config.json
COPY tools/public_file.py /app/mcpo/tools/public_file.py
COPY patch_main.py /app/mcpo/patch_main.py
COPY launch_mcpo.py /app/mcpo/launch_mcpo.py
# Ensure config directory and default config file exist for MCPO runtime
RUN mkdir -p /config && cp /app/mcpo/default_config.json /config/config.json

# Make sure Node binaries are on PATH for the stdio MCP server.
ENV PATH="/usr/local/bin:${PATH}" \
    PUBLIC_URL_BASE=http://shots.choype.com/public

# Launch MCPO with the config-only startup path and patched app.
CMD ["python3", "/app/mcpo/launch_mcpo.py"]
