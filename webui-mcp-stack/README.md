
# WebUI MCP Stack

A Docker Compose stack that bundles two Model Context Protocol (MCP) servers with a single MCPO hub, the OpenWebUI Pipelines service, and a lightweight screenshot viewer. The configuration is designed for OpenWebUI and ready for future reverse-proxy integration.

## Stack components

| Service | Purpose |
| ------- | ------- |
| **Caddy reverse proxy** | Terminates TLS, issues Let's Encrypt certificates, and routes public hostnames to the internal services. |
| **Playwright MCP** | Provides browser automation with Playwright. Configured with vision, PDF, and install capabilities, and writes artifacts to the shared `exports/` folder. |
| **UI-TARS MCP** | Node-based vision and interaction MCP server that leverages Google Chrome for UI automation. |
| **MCPO Hub** | Aggregates the two MCP servers and exposes a single endpoint for OpenWebUI or other MCP-compatible clients. |
| **OpenWebUI Pipelines** | Hosts custom pipelines—including `web_navigator`—and exposes them to OpenWebUI over HTTP. |
| **Screenshot Viewer** | Serves files from the shared `exports/` directory via HTTP so captured images can be accessed in a browser. |

## Prerequisites

* Docker
* Docker Compose v2
* A Docker network named `mcpnet` (create it once with `docker network create mcpnet`)
* DNS records for the hostnames you intend to publish (e.g. `mcpo.choype.com`, `shots.choype.com`, `pipelines.choype.com`) pointing to your server

## Usage

1. Clone this repository and change into the project directory:
   ```bash
   git clone <your-repo-url>
   cd webui-mcp-stack
   ```
2. Copy the example environment file and edit it to match your deployment:
   ```bash
   cp .env.example .env
   # Update PUBLIC_URL_BASE/MCPO_BASE_URL/MCPO_HOST_BIND/etc. to match your host
   ```
   The stack now binds service ports to `127.0.0.1` by default, so external access
   must go through the bundled Caddy reverse proxy (or another proxy of your choice).
   Adjust the `*_HOST_BIND` values only if you intentionally want to publish ports
   directly. Set `MCPO_DOMAIN`, `SHOTS_DOMAIN`, `PIPELINES_DOMAIN`, and
   `CADDY_ACME_EMAIL` so the proxy can request certificates for the correct hosts.
3. (One time) clone the OpenWebUI Pipelines repository and sync the custom pipeline:
   ```bash
   git clone https://github.com/open-webui/pipelines.git ./pipelines-upstream
   mkdir -p /data/pipelines
   rsync -av ./pipelines-upstream/ /data/pipelines/
   rsync -av ./pipelines/web_navigator/ /data/pipelines/pipelines/web_navigator/
   ```
   Alternatively set `PIPELINES_PATH` to point at any writable directory that already contains the upstream Pipelines project.
4. Start the stack (the first run builds the patched MCPO image):
   ```bash
   docker compose up --build -d
   ```
5. Capture the generated tool API key from `./mcpo/tool_api_key` (or provide
   your own via the `TOOL_API_KEY` environment variable). The `/tool/public_file`
   endpoint now requires the key in an `X-API-Key` header.
6. Connect OpenWebUI (or another MCP client) to the MCPO endpoint through your
   reverse proxy (example hostnames shown):
   ```
   https://mcpo.choype.com/mcp
   ```
7. Register the pipeline endpoint in OpenWebUI **Settings → Pipelines**:
   * Base URL: `http://pipelines:9099` (or `https://pipelines.<your-domain>` if proxied)
   * Pipeline: `web_navigator`
8. Access generated screenshots or other exports via the viewer domain:
   ```
   https://shots.choype.com/<file>.png
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
| `PUBLIC_URL_BASE` | Base URL advertised by the `/tool/public_file` endpoint. | `https://shots.choype.com/public` |
| `MCPO_DOMAIN` | Hostname that Caddy should route to the MCPO hub. | `mcpo.localhost` |
| `SHOTS_DOMAIN` | Hostname that Caddy should route to the screenshot viewer. | `shots.localhost` |
| `PIPELINES_DOMAIN` | Hostname that Caddy should route to the pipelines API. | `pipelines.localhost` |
| `MCPO_BASE_URL` | MCPO URL (including the server slug, e.g. `/mcp_playwright`) used by the `web_navigator` pipeline. | `http://mcpo:3879/mcp_playwright` |
| `MCPO_HOST_BIND` | Host bind address for the MCPO container port. | `127.0.0.1:3880` |
| `PIPELINES_HOST_BIND` | Host bind address for the pipelines service. | `127.0.0.1:9099` |
| `SHOTS_HOST_BIND` | Host bind address for the screenshot viewer. | `127.0.0.1:3888` |
| `CADDY_ACME_EMAIL` | Email address used by Caddy when requesting Let's Encrypt certificates. | `admin@example.com` |

The MCPO hub exposes each upstream tool on its own path under the same domain.
For example `https://mcpo.choype.com/mcp_playwright` reaches the Playwright MCP
server while `https://mcpo.choype.com/mcp_tars` targets the UI-TARS MCP. Set
`MCPO_BASE_URL` to the specific path you want the `web_navigator` pipeline to
call; the pipeline will append `/v1/...` to that base when issuing requests.

The MCPO container writes a randomly generated tool API key to
`./mcpo/tool_api_key` the first time it launches. Provide the same key via the
`X-API-Key` header when calling the `/tool/public_file` endpoint from external
clients. To supply your own value, set `TOOL_API_KEY` before running
`docker compose up`.

## Networking and proxy readiness

All services communicate over a shared Docker bridge network (`mcpnet`). The MCPO hub, Pipelines API, and screenshot viewer now bind to the loopback interface on the host to prevent unsolicited public access. The bundled Caddy service terminates TLS and publishes the domains (`mcpo.choype.com`, `shots.choype.com`, `pipelines.choype.com`). If you prefer a different proxy, disable the Caddy service and route traffic to the loopback-bound ports yourself.

### Bundled Caddy reverse proxy

The stack now ships with a dedicated [Caddy](https://caddyserver.com/) service that
terminates TLS, issues certificates via Let's Encrypt, and proxies each public
hostname to the appropriate internal container. The default
[`caddy/Caddyfile`](./caddy/Caddyfile) expects the domain names provided in `.env`:

```caddyfile
{$MCPO_DOMAIN} {
  reverse_proxy mcpo:3879
}

{$SHOTS_DOMAIN} {
  reverse_proxy screenshot-viewer:80
}

{$PIPELINES_DOMAIN} {
  reverse_proxy pipelines:9099
}
```

Update the domain variables to match your DNS records. If you expose multiple
MCPO tools, add additional `reverse_proxy` blocks that point to the same MCPO
container port (`mcpo:3879`) and let MCPO route requests based on the path
(`mcp_playwright`, `mcp_tars`, etc.). When using custom TLS certificates instead of
Let's Encrypt, replace the automatic certificate issuance with a
[`tls`](https://caddyserver.com/docs/caddyfile/directives/tls) directive inside each
site block. Set `CADDY_ACME_EMAIL` in `.env` to an address you control so Let's
Encrypt renewal notices reach you.

## Repository layout

```
webui-mcp-stack/
├── compose.yml
├── caddy/
│   └── Caddyfile
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
