# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

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

# Add local reranking stack (LlamaIndex core + FlagEmbedding reranker + torch)
# Pin core to a stable 0.13.x to avoid API churn; adjust torch wheel for your CUDA if needed.
RUN uv add "llama-index-core==0.13.5" \
           "llama-index-postprocessor-flag-embedding-reranker>=0.1.2,<1.0" \
           "FlagEmbedding>=1.2.0,<2.0" \
           "torch>=2.2,<3.0"

# Copy the rest of the source
COPY . .

# Pre-download multilingual turn detector assets into the image for offline use
RUN uv run python scripts/download_turn_detector_assets.py

ENV HF_HUB_OFFLINE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

RUN rm -rf /root/.cache && mkdir -p /tmp/uv-cache

# Removed: 'download-files' step (no such command in your agent)
# RUN uv run src/agent.py download-files

# Health check for container readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD uv run python -c "import asyncio, os; from dotenv import load_dotenv; from src.agent import health_check; load_dotenv('.env.local'); result = asyncio.run(health_check()); print(f'Health status: {result[\"status\"]}'); exit(0 if result['status'] == 'healthy' else 1)"

# Start the worker
CMD ["uv","run","-m","src.agent","start"]
