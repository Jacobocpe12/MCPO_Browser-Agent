# Playwright Vision MCP image

This directory only contains the Dockerfile that builds the upstream
[playwright-vision-mcp](https://github.com/davidkim9/playwright-vision-mcp)
project at image build time. The upstream repository is cloned, built, and
pruned inside the container so the service stays up to date without vendoring
its source tree into this repository.

To pin to a different commit or fork, override the build arguments that the
Dockerfile exposes:

```sh
docker compose build \
  --build-arg PLAYWRIGHT_VISION_REPO=https://github.com/<fork>/playwright-vision-mcp.git \
  --build-arg PLAYWRIGHT_VISION_REF=<commit>
```

The resulting image runs the upstream HTTP server entry point on port `8500` and
stores screenshots in `/exports/screenshots`.
