# Multi-stage lightweight Python base
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.2

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services ./services
COPY pyproject.toml* poetry.lock* requirements.txt* ./

# For now, use pip directly (can switch to poetry later if desired)
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi || true && \
    pip install --no-cache-dir fastapi uvicorn[standard]

EXPOSE 8000

# Default command expects SERVICE module env (e.g., ingestion)
ENV SERVICE=ingestion \
    SERVICE_NAME=ingestion-service
CMD ["/bin/sh", "-c", "uvicorn services.${SERVICE}.main:app --host 0.0.0.0 --port 8000"]
