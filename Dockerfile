FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    VIMAX_WEB_HOST=0.0.0.0 \
    VIMAX_WEB_PORT=4173 \
    VIMAX_PYTHON_CMD=python3

WORKDIR /app

# Install system FFmpeg (with libass for subtitle burn-in), fonts (Vietnamese/Unicode glyphs), Node.js, and build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto-core \
    nodejs \
    npm \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies declared in pyproject.toml
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r pyproject.toml

# Build Web Studio frontend assets
COPY web/package.json web/package-lock.json ./web/
RUN cd web && npm ci --no-audit --no-fund

COPY . .
RUN cd web && npm run build && npm prune --omit=dev

RUN mkdir -p /app/.working_dir /app/.vimax /app/configs

EXPOSE 4173

CMD ["node", "web/server.mjs"]
