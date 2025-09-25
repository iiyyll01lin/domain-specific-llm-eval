# Multi-stage lightweight Python base
FROM python:3.11-slim AS base

ARG APP_USER=rag
ARG APP_UID=1000
ARG APP_GID=1000
ARG MODELS_CACHE=/var/cache/rag-models

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.2 \
    MODELS_CACHE_PATH=${MODELS_CACHE} \
    EXTENSIONS_DIR=/extensions

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid ${APP_GID} ${APP_USER} \
    && useradd --system --uid ${APP_UID} --gid ${APP_GID} --home-dir /app --create-home --shell /usr/sbin/nologin ${APP_USER}

WORKDIR /app

# Pre-copy dependency manifests for better layer caching
COPY pyproject.toml* poetry.lock* requirements.txt* ./

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir fastapi uvicorn[standard] \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY services ./services

RUN mkdir -p ${MODELS_CACHE_PATH} ${EXTENSIONS_DIR} \
    && chown -R ${APP_USER}:${APP_USER} /app ${MODELS_CACHE_PATH} ${EXTENSIONS_DIR}

VOLUME ["${MODELS_CACHE}"]

ENV PATH="/home/${APP_USER}/.local/bin:${PATH}" \
    SERVICE=ingestion \
    SERVICE_NAME=ingestion-service \
    EXTENSIONS_DIR=/extensions

EXPOSE 8000

USER ${APP_USER}

# Default command expects SERVICE module env (e.g., ingestion)
CMD ["/bin/sh", "-c", "uvicorn services.${SERVICE}.main:app --host 0.0.0.0 --port 8000"]
