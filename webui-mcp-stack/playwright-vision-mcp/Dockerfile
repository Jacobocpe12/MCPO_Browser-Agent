# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/playwright:v1.49.0-jammy AS builder

ARG PLAYWRIGHT_VISION_REPO=https://github.com/davidkim9/playwright-vision-mcp.git
ARG PLAYWRIGHT_VISION_REF=7320c67a55b7d95fec0288cd036e7d9b16df23ca

WORKDIR /tmp/playwright-vision

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone "$PLAYWRIGHT_VISION_REPO" . \
    && git checkout "$PLAYWRIGHT_VISION_REF" \
    && npm install \
    && npm run build \
    && npm prune --omit=dev \
    && rm -rf .git

FROM mcr.microsoft.com/playwright:v1.49.0-jammy

WORKDIR /app

COPY --from=builder /tmp/playwright-vision /app

ENV NODE_ENV=production \
    PORT=8500 \
    PLAYWRIGHT_HEADLESS=true \
    SCREENSHOT_DIR=/exports/screenshots \
    BROWSER_TYPE=chromium

EXPOSE 8500

CMD ["node", "dist/http-server.js"]
