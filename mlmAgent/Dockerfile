# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./
RUN mkdir -p src

# Install deps from lock; then add aiohttp used by rag_client
# If your lock already has aiohttp, the add is a no-op.
RUN uv sync --locked && uv add aiohttp

# Copy the rest of the source
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

# Removed: 'download-files' step (no such command in your agent)
# RUN uv run src/agent.py download-files

# Start the worker
CMD ["uv", "run", "src/agent.py", "start"]
