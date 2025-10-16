
# WebUI MCP Stack

A Docker Compose stack that bundles two Model Context Protocol (MCP) servers with a single MCPO hub, the OpenWebUI Pipelines service, and a lightweight screenshot viewer. The configuration is designed for OpenWebUI and ready for future reverse-proxy integration.

## Stack components

| Service | Purpose |
| ------- | ------- |
| **Playwright MCP** | Provides browser automation with Playwright. Configured with vision, PDF, and install capabilities, and writes artifacts to the shared `exports/` folder. |
| **UI-TARS MCP** | Node-based vision and interaction MCP server that leverages Google Chrome for UI automation. |
| **OpenAI OCR MCP** | STDIO-based OCR service from [`cjus/openai-ocr-mcp`](https://github.com/cjus/openai-ocr-mcp) compiled into the MCPO container. Reads images from `/exports` and extracts text using OpenAI vision models. |
| **MCPO Hub** | Aggregates the MCP servers and exposes a single endpoint for OpenWebUI or other MCP-compatible clients. |
| **OpenWebUI Pipelines** | Hosts custom pipelines—including `web_navigator`—and exposes them to OpenWebUI over HTTP. |
| **Screenshot Viewer** | Serves files from the shared `exports/` directory via HTTP so captured images can be accessed in a browser. |

## Prerequisites

* Docker
* Docker Compose v2
* A Docker network named `mcpnet` (create it once with `docker network create mcpnet`)

## Usage

1. Clone this repository and change into the project directory:
   ```bash
   git clone <your-repo-url>
   cd webui-mcp-stack
   ```
2. (One time) clone the OpenWebUI Pipelines repository and sync the custom pipeline:
   ```bash
   git clone https://github.com/open-webui/pipelines.git ./pipelines-upstream
   mkdir -p /data/pipelines
   rsync -av ./pipelines-upstream/ /data/pipelines/
   rsync -av ./pipelines/web_navigator/ /data/pipelines/pipelines/web_navigator/
   ```
   Alternatively set `PIPELINES_PATH` to point at any writable directory that already contains the upstream Pipelines project.
3. Start the stack (the first run builds the patched MCPO image):
   ```bash
   docker compose up --build -d
   ```
4. Connect OpenWebUI (or another MCP client) to the MCPO endpoint:
   ```
   http://<host>:3880/mcp
   ```
5. Register the pipeline endpoint in OpenWebUI **Settings → Pipelines**:
   * Base URL: `http://pipelines:9099`
   * Pipeline: `web_navigator`
6. Access generated screenshots or other exports via the viewer:
   ```
   http://<host>:3888/<file>.png
   ```

### How MCPO is configured

The MCPO container now runs in **config-only** mode so it can attach to the already running Playwright and TARS MCP servers while spawning the bundled OpenAI OCR MCP binary via stdio. The bundled [`mcpo/config.json`](./mcpo/config.json) advertises all endpoints:

```json
{
  "mcpServers": {
    "mcp_playwright": {
      "type": "streamable-http",
      "url": "http://playwright-mcp:8931/mcp"
    },
    "mcp_tars": {
      "type": "streamable-http",
      "url": "http://ui-tars-mcp:8000/mcp"
    },
    "mcp_openai_ocr": {
      "command": "node",
      "args": ["/opt/openai-ocr-mcp/dist/ocr.js"]
    }
  }
}
```

When `docker compose up` runs, the MCPO container executes:

```bash
mcpo --config /config/config.json --host 0.0.0.0 --port 3879
```

This avoids the `TypeError: 'NoneType' object is not subscriptable` crash that occurred when mixing `--server-type` CLI arguments with the JSON configuration.

### OpenAI OCR MCP service

The MCPO image now vendors the [`openai-ocr-mcp`](https://github.com/cjus/openai-ocr-mcp) project inside the container and exposes it as a stdio MCP server. MCPO spawns the compiled Node binary directly, so there is no separate OCR container to manage. The service inherits upstream behavior: it validates image paths, performs OCR via OpenAI's vision API, stores extracted text alongside the input image, and offers an `append_analysis` helper for augmenting the generated files. Point MCP-aware clients at the `mcp_openai_ocr` tool namespace when invoking it from OpenWebUI or other orchestrators. For Codex-style orchestrators, reuse the upstream-aligned system prompt stored at [`mcpo/openai-ocr-mcp/CODEX_PROMPT.md`](./mcpo/openai-ocr-mcp/CODEX_PROMPT.md).

### Adding the `web_navigator` pipeline

The [`pipelines/web_navigator/navigator_pipe.py`](./pipelines/web_navigator/navigator_pipe.py) module defines a pipeline that forwards navigation instructions from OpenWebUI to MCPO while streaming log messages and screenshots back to the chat interface. To enable it:

1. Ensure the Pipelines container is mounting a volume that contains the upstream OpenWebUI Pipelines project plus this repository's `web_navigator` directory. By default the compose file points to `${PIPELINES_PATH:-./pipelines}`, so either set `PIPELINES_PATH` before running `docker compose up` or copy the repositories into `./pipelines`.
2. Start (or restart) the stack. The Pipelines container log should include `Loaded pipeline: web_navigator`.
3. Issue pipeline commands from OpenWebUI chats using the `#pipeline=web_navigator` directive, for example:
   ```
   #pipeline=web_navigator
   Navigate to https://geoportal.nrw and find flood hazard maps for Aachen.
   ```

The pipeline polls MCPO for status updates, streams log events, and publishes screenshots as base64 images so they render inline in OpenWebUI.

### Environment variables

The compose file honors the following environment variables:

| Variable | Purpose | Default |
| -------- | ------- | ------- |
| `OPENAI_API_KEY` | API key for the vision-capable LLM used by the Pipelines server. | _None_ (must be set) |
| `OPENAI_API_BASE` | Base URL for the LLM endpoint. | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Model identifier passed to the Pipelines runtime. | `gpt-4o-mini` |
| `PIPELINES_PATH` | Host directory mounted into `/app/pipelines` inside the Pipelines container. | `./pipelines` |

## Networking and proxy readiness

All services communicate over a shared Docker bridge network (`mcpnet`). Only the MCPO hub, Pipelines API, and screenshot viewer expose ports on the host. This stack is proxy-ready (Traefik planned for `*.choype.com`) so it can be fronted by a reverse proxy without restructuring the containers.

## Repository layout

```
webui-mcp-stack/
├── compose.yml
├── mcpo/
│   ├── config.json
│   ├── launch_mcpo.py
│   ├── patch_main.py
│   ├── tools/
│   │   └── public_file.py
│   └── Dockerfile
├── pipelines/
│   └── web_navigator/
│       └── navigator_pipe.py
├── exports/
│   └── .gitkeep
├── .gitignore
└── README.md
```

## Deploying to GitHub and Portainer

1. Initialize a Git repository and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin git@github.com:<user>/webui-mcp-stack.git
   git push -u origin main
   ```
2. In Portainer, create a new stack and paste the contents of `compose.yml`, or point Portainer to the GitHub repository. Deploy the stack and monitor service logs from the Portainer UI.
