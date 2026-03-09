# syntax=docker/dockerfile:1.6

FROM python:3.11-slim-bookworm AS builder

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG PIP_NETWORK_CHECK_URL=https://pypi.org/simple/
ARG PIP_NETWORK_TIMEOUT=5
# GPU build profile: set ENABLE_GPU=true to install torch with CUDA support
ARG ENABLE_GPU=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=45 \
    PIP_RETRIES=3

ENV PIP_INDEX_URL=$PIP_INDEX_URL \
    PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL \
    PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST \
    PIP_NETWORK_CHECK_URL=$PIP_NETWORK_CHECK_URL \
    PIP_NETWORK_TIMEOUT=$PIP_NETWORK_TIMEOUT

WORKDIR /app

RUN set -eux; \
    if apt-get update; then \
        apt-get upgrade -y --no-install-recommends || true; \
        apt-get install -y --no-install-recommends build-essential || true; \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "⚠️  Skipping apt packages in builder stage (network unavailable)"; \
    fi; \
    python -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml* poetry.lock* requirements.txt* ./

RUN set -eux; \
    network_available=0; \
    if python -c "import os, urllib.request; urllib.request.urlopen(os.environ.get('PIP_NETWORK_CHECK_URL','https://pypi.org/simple/'), timeout=float(os.environ.get('PIP_NETWORK_TIMEOUT','5')))" >/dev/null 2>&1; then \
        network_available=1; \
    else \
        echo "PyPI unreachable during image build" >&2; \
    fi; \
    if [ "${network_available}" -eq 1 ]; then \
        python -m pip install --no-cache-dir --default-timeout=30 --retries=3 --upgrade pip setuptools wheel; \
    else \
        exit 1; \
    fi; \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir --default-timeout=90 --retries=5 -r requirements.txt; \
    fi; \
    if [ "${ENABLE_GPU}" = "true" ]; then \
        echo "🔧 GPU profile: installing torch with CUDA 12.1 support"; \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || echo "⚠️  GPU torch install failed; falling back to CPU torch"; \
    fi; \
    python -c "import fastapi, uvicorn, pydantic" >/dev/null

FROM python:3.11-slim-bookworm AS runtime

ARG APP_USER=rag
ARG APP_UID=1000
ARG APP_GID=1000
ARG MODELS_CACHE=/var/cache/rag-models
ARG EXTENSIONS_DIR=/extensions

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELS_CACHE_PATH=${MODELS_CACHE} \
    EXTENSIONS_DIR=${EXTENSIONS_DIR}

WORKDIR /app

RUN set -eux; \
    if apt-get update; then \
        apt-get upgrade -y --no-install-recommends || true; \
        apt-get install -y --no-install-recommends curl || true; \
        rm -rf /var/lib/apt/lists/*; \
    else \
        echo "⚠️  Skipping apt packages in runtime stage (network unavailable)"; \
    fi; \
    groupadd --system --gid ${APP_GID} ${APP_USER}; \
    useradd --system --uid ${APP_UID} --gid ${APP_GID} --home-dir /app --create-home --shell /usr/sbin/nologin ${APP_USER}

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

COPY services ./services

RUN mkdir -p ${MODELS_CACHE_PATH} ${EXTENSIONS_DIR} \
    && chown -R ${APP_USER}:${APP_USER} /app ${MODELS_CACHE_PATH} ${EXTENSIONS_DIR}

VOLUME ["${MODELS_CACHE}"]

ENV SERVICE=ingestion \
    SERVICE_NAME=ingestion-service

EXPOSE 8000

USER ${APP_USER}

CMD ["/bin/sh", "-c", "uvicorn services.${SERVICE}.main:app --host 0.0.0.0 --port 8000"]
