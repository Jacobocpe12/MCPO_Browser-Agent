# MCPO_Browser-Agent

An MCP -> MCPO Proxy Browser Agent.

## MCPO Fix – Config-Only Startup for Streamable HTTP Servers

### Problem

`mcpo` previously exited immediately with `TypeError: 'NoneType' object is not subscriptable` whenever a `--server-type` argument was combined with `--config` or when a JSON configuration used `"streamable_http"`. In this state, `server_command` was `None`, so MCPO assumed stdio mode instead of HTTP and shut down.

### Root Cause

The `type` field must use a hyphenated value (`streamable-http`) and MCPO must be launched in config-only mode when connecting to pre-running MCP services (Playwright, TARS, etc.). Passing both CLI and config parameters confused MCPO’s startup logic.

### Solution

1. Remove trailing `--server-type … -- <url>` arguments.
2. Create `/config/config.json` with proper `streamable-http` entries.
3. Start MCPO using only the `--config` flag.

### Reference config.json

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

### Startup command

```bash
docker run -it --rm \
  --name mcpo-debug \
  --network mcpo_browser-agent_v1_mcpnet \
  -v $(pwd)/mcpo:/config \
  -p 3880:3879 \
  ghcr.io/open-webui/mcpo:latest \
  mcpo --config /config/config.json --host 0.0.0.0 --port 3879
```

### Expected result

```
Starting MCP OpenAPI Proxy with config file: /config/config.json
INFO - Loaded MCP server: mcp_playwright
INFO - Loaded MCP server: mcp_tars
INFO - Connecting to http://playwright-mcp:8931/mcp
INFO - Connecting to http://ui-tars-mcp:8000/mcp
INFO - Uvicorn running on http://0.0.0.0:3879
```

### Implementation notes

* Use config-only startup for MCPO.
* Ensure the `type` key in JSON uses `streamable-http`.
* Remove redundant `--server-type` flags from all docker-compose and CLI invocations.
* Add health-check logic if needed before starting MCPO.
* The Docker stack now builds a patched MCPO image whose launcher copies a default config into `/config/config.json` if the bind mount is empty, eliminating `FileNotFoundError` startup crashes.
