# syntax=docker/dockerfile:1
# Misinformation Detection Environment — Hugging Face Spaces compatible Docker image
#
# Single server on port 7860 (the only port HF Spaces exposes publicly).
# task_id is selected per-episode via the /reset request body — no multi-port needed.
#
# Default task: TASK_ID env var (default: "easy"). Clients override per-episode
# by passing task_id in the /reset body: {"task_id": "medium"}.

FROM python:3.11-slim

# System deps: curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Install the lazarus package (makes 'from lazarus.x import y' resolve correctly)
RUN pip install --no-cache-dir -e .

# HF Spaces standard port
EXPOSE 7860

# Default task difficulty (clients can override per-episode via reset body)
ENV TASK_ID=easy
ENV PYTHONPATH=/app

# Health check — /health provided by openenv-core create_app
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Single server on 7860 — canonical OpenEnv pattern
CMD ["uvicorn", "lazarus.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
