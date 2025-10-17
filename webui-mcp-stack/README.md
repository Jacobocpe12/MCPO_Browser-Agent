
# WebUI MCP Stack

A Docker Compose stack that bundles two Model Context Protocol (MCP) servers with a single MCPO hub, the OpenWebUI Pipelines service, and a lightweight screenshot viewer. The configuration is designed for OpenWebUI and ready for future reverse-proxy integration.

## Stack components

| Service | Purpose |
| ------- | ------- |
| **Playwright MCP** | Provides browser automation with Playwright. Configured with vision, PDF, and install capabilities, and writes artifacts to the shared `exports/` folder. |
| **UI-TARS MCP** | Node-based vision and interaction MCP server that leverages Google Chrome for UI automation. |
| **Playwright Vision MCP** | Adds browser automation, screenshot capture, and GPT-backed vision analysis via the `playwright-vision-mcp` service. Artifacts are written to the shared `exports/` folder. |
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

The MCPO container now runs in **config-only** mode so it can attach to the already running Playwright, TARS, and Playwright Vision MCP servers without extra CLI flags. The bundled [`mcpo/config.json`](./mcpo/config.json) uses the required hyphenated `streamable-http` type and advertises all endpoints:

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
    "mcp_playwright_vision": {
      "type": "streamable-http",
      "url": "http://playwright-vision-mcp:8500/mcp"
    }
  }
}
```

When `docker compose up` runs, the MCPO container executes:

```bash
mcpo --config /config/config.json --host 0.0.0.0 --port 3879
```

This avoids the `TypeError: 'NoneType' object is not subscriptable` crash that occurred when mixing `--server-type` CLI arguments with the JSON configuration.

### Playwright Vision MCP service

The [`playwright-vision-mcp`](./playwright-vision-mcp) directory contains a Dockerfile that builds the upstream [playwright-vision-mcp](https://github.com/davidkim9/playwright-vision-mcp) project directly from Git. This keeps the stack lightweight—no vendored source tree—while still exposing the full tool suite that combines Playwright-driven browser automation with GPT-powered vision analysis and image interpretation. Override the `PLAYWRIGHT_VISION_REPO` or `PLAYWRIGHT_VISION_REF` build arguments in `docker compose build` to point at a fork or different commit.

Key defaults applied by the Dockerfile and compose definition:

* Runs in headless mode on port `8500`.
* Persists screenshots to the shared `/exports/screenshots` directory so the screenshot viewer and other services can read them.
* Uses Chromium by default, with overrides available through the `BROWSER_TYPE` environment variable.

Expose the MCP through OpenWebUI using the `mcp_playwright_vision` entry emitted into `config.json`.

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
├── playwright-vision-mcp/
│   ├── Dockerfile
│   └── README.md
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
