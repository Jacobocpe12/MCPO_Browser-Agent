# WebUI MCP Stack

A Docker Compose stack that bundles two Model Context Protocol (MCP) servers with a single MCPO hub, a lightweight screenshot viewer, and a built-in tool for publishing artifacts. The configuration is designed for OpenWebUI and ready for future reverse-proxy integration.

## Stack components

| Service | Purpose |
| ------- | ------- |
| **Playwright MCP** | Provides browser automation with Playwright. Configured with vision, PDF, and install capabilities, and writes artifacts to the shared `exports/` folder. |
| **UI-TARS MCP** | Node-based vision and interaction MCP server that leverages Google Chrome for UI automation. |
| **MCPO Hub** | Aggregates the two MCP servers, exposes a single endpoint for OpenWebUI or other MCP-compatible clients, and now ships with a built-in `/tool/public_file` FastAPI endpoint for sharing files. |
| **Screenshot Viewer** | Serves files from the shared `exports/` directory via HTTP so captured images can be accessed in a browser. |

## Prerequisites

* Docker
* Docker Compose v2

## Usage

1. Clone this repository and change into the project directory:
   ```bash
   git clone <your-repo-url>
   cd webui-mcp-stack
   ```
2. Start the stack (Compose will build the custom MCPO image automatically):
   ```bash
   docker compose up -d
   ```
3. Connect OpenWebUI (or another MCP client) to the MCPO endpoint:
   ```
   http://<host>:3880/mcp
   ```
4. Access generated screenshots or other exports via the viewer:
   ```
   http://<host>:3888/<file>.png
   ```

## Public file tool

The MCPO container exposes an internal `/tool/public_file` endpoint that allows MCP services or external callers to publish either text or base64-encoded content. Files are written to `/app/public` (mapped to the local `exports/` directory), served under `/public/<filename>`, and automatically removed four hours after creation.

Example request:

```bash
curl -X POST http://<host>:3880/tool/public_file \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello World"}'
```

Response payload:

```json
{"url": "http://shots.choype.com/public/2c73a3f2.txt"}
```

Use the returned URL directly or proxy it through the `screenshot-viewer` service when a reverse proxy is in place.

## Networking and proxy readiness

All services communicate over an internal Docker bridge network (`mcpnet`). Only the MCPO hub and screenshot viewer expose ports on the host. This stack is proxy-ready (Traefik planned for `*.choype.com`) so it can be fronted by a reverse proxy without restructuring the containers.

## Repository layout

```
webui-mcp-stack/
├── compose.yml
├── mcpo/
│   ├── config.json
│   ├── Dockerfile
│   ├── patch_main.py
│   └── tools/
│       └── public_file.py
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
