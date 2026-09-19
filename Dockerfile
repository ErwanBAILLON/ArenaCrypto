# syntax=docker/dockerfile:1
# Multi-stage build: uv-managed venv in builder, slim non-root runtime.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# lightgbm wheels need libgomp at runtime; nothing is compiled here.
COPY pyproject.toml uv.lock README.md ./
COPY arena/ ./arena/
RUN uv sync --frozen --no-dev --no-editable --extra web

FROM python:3.12-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 1000 arena && useradd -u 1000 -g 1000 -m -s /usr/sbin/nologin arena
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv
WORKDIR /app
COPY --chown=1000:1000 config/ ./config/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UNIVERSE_PATH=/app/config/universe.yaml
USER 1000:1000
ENTRYPOINT ["arena"]
CMD ["--help"]
