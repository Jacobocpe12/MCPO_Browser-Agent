# WebUI MCP Stack

A Docker Compose stack that bundles two Model Context Protocol (MCP) servers with a single MCPO hub and a lightweight screenshot viewer. The configuration is designed for OpenWebUI and ready for future reverse-proxy integration.

## Stack components

| Service | Purpose |
| ------- | ------- |
| **Playwright MCP** | Provides browser automation with Playwright. Configured with vision, PDF, and install capabilities, and writes artifacts to the shared `exports/` folder. |
| **UI-TARS MCP** | Node-based vision and interaction MCP server that leverages Google Chrome for UI automation. |
| **MCPO Hub** | Aggregates the two MCP servers and exposes a single endpoint for OpenWebUI or other MCP-compatible clients. |
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
2. Start the stack (the first run builds the patched MCPO image):
   ```bash
   docker compose up --build -d
   ```
3. Connect OpenWebUI (or another MCP client) to the MCPO endpoint:
   ```
   http://<host>:3880/mcp
   ```
4. Access generated screenshots or other exports via the viewer:
   ```
   http://<host>:3888/<file>.png
   ```

### How MCPO is configured

The MCPO container now runs in **config-only** mode so it can attach to the already running Playwright and TARS MCP servers without extra CLI flags. The bundled [`mcpo/config.json`](./mcpo/config.json) uses the required hyphenated `streamable-http` type and advertises both endpoints:

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
    }
  }
}
```

When `docker compose up` runs, the MCPO container executes:

```bash
mcpo --config /config/config.json --host 0.0.0.0 --port 3879
```

This avoids the `TypeError: 'NoneType' object is not subscriptable` crash that occurred when mixing `--server-type` CLI arguments with the JSON configuration.

## Networking and proxy readiness

All services communicate over an internal Docker bridge network (`mcpnet`). Only the MCPO hub and screenshot viewer expose ports on the host. This stack is proxy-ready (Traefik planned for `*.choype.com`) so it can be fronted by a reverse proxy without restructuring the containers.

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

