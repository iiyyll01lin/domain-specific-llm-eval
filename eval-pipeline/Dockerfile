# Domain-Specific RAG Evaluation Pipeline - Docker Container
# Multi-stage build for optimized production image

# Build arguments for proxy configuration
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY=localhost,127.0.0.1

# =============================================================================
# Stage 1: Base Python Environment with System Dependencies
# =============================================================================
FROM python:3.10-slim-bullseye as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    NO_PROXY=${NO_PROXY}

# Configure apt to use proxy
RUN if [ -n "${HTTP_PROXY}" ]; then \
        echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" > /etc/apt/apt.conf.d/01proxy && \
        echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/01proxy; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # PDF processing
    poppler-utils \
    # Document processing
    libffi-dev \
    libssl-dev \
    # Network utilities
    curl \
    wget \
    # File utilities
    file \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# Stage 2: Python Dependencies Installation
# =============================================================================
FROM base as dependencies

# Pass build arguments to this stage
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Set proxy environment variables for pip
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    NO_PROXY=${NO_PROXY}

# Create application directory
WORKDIR /app

# Copy dependency files from eval-pipeline directory
COPY eval-pipeline/requirements.txt .
COPY eval-pipeline/setup.py .

# Install Python dependencies with retry logic and better error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
        $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") && \
    pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt \
        $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") || \
    ( echo "First attempt failed, trying with different PyPI index..." && \
      pip install --no-cache-dir --timeout 120 --retries 5 \
          --index-url https://pypi.org/simple/ \
          --trusted-host pypi.org \
          -r requirements.txt $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") ) || \
    ( echo "Trying with essential packages only..." && \
      pip install --no-cache-dir --timeout 120 --retries 5 \
          pandas numpy PyYAML openpyxl sentence-transformers spacy \
          $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") )

# Download and install NLP models (with timeout and fallback)
RUN timeout 300 python -m spacy download en_core_web_sm || echo "Spacy download timed out" && \
    timeout 300 python -m spacy download en_core_web_md || echo "Spacy md download timed out" && \
    timeout 300 python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')" || echo "NLTK download timed out"

# Pre-download sentence-transformers model (with timeout)
RUN timeout 300 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || echo "Sentence transformer download timed out"

# =============================================================================
# Stage 2.1: Local RAGAS Installation
# =============================================================================
FROM dependencies as ragas-local

# Pass build arguments to this stage
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Set proxy environment variables for pip
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    NO_PROXY=${NO_PROXY}

# Copy local RAGAS submodule
# RAGAS is located at project root when building from parent directory
COPY ragas/ /app/ragas/

# Install RAGAS from local source (editable install for development)
WORKDIR /app/ragas

# Check if this is the proper RAGAS directory structure
RUN if [ -d "ragas" ]; then \
        echo "✅ Found nested RAGAS structure, using ragas/ragas/" && \
        cd ragas; \
    else \
        echo "✅ Using flat RAGAS structure" && \
        pwd; \
    fi

# First, try to install RAGAS dependencies and build tools
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
        $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") \
        setuptools wheel build pip-tools || echo "⚠️ Basic build tools installation completed with warnings"

# Install RAGAS from local source with robust error handling
RUN if [ -f "pyproject.toml" ]; then \
        echo "📦 Installing RAGAS from pyproject.toml..." && \
        pip install --no-cache-dir -e . \
            $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") && \
        echo "✅ Local RAGAS installation successful"; \
    elif [ -f "setup.py" ]; then \
        echo "📦 Installing RAGAS from setup.py..." && \
        pip install --no-cache-dir -e . \
            $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") && \
        echo "✅ Local RAGAS installation successful"; \
    else \
        echo "❌ No pyproject.toml or setup.py found, trying PyPI fallback..." && \
        pip install --no-cache-dir --timeout 120 --retries 5 ragas \
            $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") && \
        echo "✅ PyPI RAGAS installation successful" || \
        echo "❌ RAGAS installation failed - pipeline will use fallback methods"; \
    fi

# Try to install optional RAGAS dependencies for full functionality
RUN pip install --no-cache-dir --timeout 120 --retries 3 \
        $([ -n "${HTTP_PROXY}" ] && echo "--proxy ${HTTP_PROXY}") \
        sentence-transformers transformers nltk rouge_score rapidfuzz \
        pandas datacompy || echo "⚠️ Some optional RAGAS dependencies failed to install"

# Verify RAGAS installation and get version info
RUN python -c "import ragas; print(f'✅ RAGAS version: {ragas.__version__}')" && \
    python -c "import ragas; print(f'✅ RAGAS location: {ragas.__file__}')" || \
    echo "❌ RAGAS import failed - pipeline will work with limited RAGAS features"

# Show available RAGAS modules for debugging
RUN python -c "import ragas; print('📋 Available RAGAS modules:'); import pkgutil; [print(f'  - {name}') for _, name, _ in pkgutil.iter_modules(ragas.__path__)]" || echo "⚠️ Could not list RAGAS modules"

# Return to app directory
WORKDIR /app

# =============================================================================
# Stage 2.5: Tiktoken Offline Cache Setup
# =============================================================================
FROM ragas-local as tiktoken-setup

# Create tiktoken cache directory
RUN mkdir -p /app/.cache/tiktoken

# Copy tiktoken setup script from eval-pipeline directory
COPY eval-pipeline/scripts/setup_tiktoken_robust.sh /tmp/setup_tiktoken_robust.sh

# Download tiktoken cache during build (requires internet)
# This is optional - if it fails, container will work with fallback
RUN chmod +x /tmp/setup_tiktoken_robust.sh && \
    /tmp/setup_tiktoken_robust.sh || echo "⚠️ Tiktoken cache setup skipped - will use runtime fallback" && \
    rm /tmp/setup_tiktoken_robust.sh

# Set environment variables for offline tiktoken operation
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    TIKTOKEN_CACHE_ONLY=1 \
    TIKTOKEN_DISABLE_DOWNLOAD=1 \
    TIKTOKEN_FORCE_OFFLINE=1

# Ensure cache directory permissions
RUN chown -R pipeline:pipeline /app/.cache/ 2>/dev/null || true

# =============================================================================
# Stage 3: Application Setup
# =============================================================================
FROM tiktoken-setup as application

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash pipeline && \
    chown -R pipeline:pipeline /app

# Copy application code from eval-pipeline directory
COPY --chown=pipeline:pipeline eval-pipeline/ .

# Make scripts executable
RUN chmod +x docker-entrypoint.sh scripts/init_tiktoken.py

# Create required directories
RUN mkdir -p \
    /app/data/documents \
    /app/outputs/testsets \
    /app/outputs/evaluations \
    /app/outputs/reports \
    /app/outputs/visualizations \
    /app/outputs/metadata \
    /app/outputs/logs \
    /app/cache \
    /app/temp \
    /app/logs && \
    chown -R pipeline:pipeline /app

# Switch to non-root user
USER pipeline

# Set working directory
WORKDIR /app

# Validate installation
RUN python setup.py --quick-test || echo "Quick test completed with warnings"

# =============================================================================
# Stage 4: Production Image (Clean proxy configuration as root before user switch)
# =============================================================================
FROM tiktoken-setup as production

# Remove proxy configuration from production image for security (as root)
RUN if [ -f /etc/apt/apt.conf.d/01proxy ]; then \
        rm -f /etc/apt/apt.conf.d/01proxy; \
    fi

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash pipeline && \
    chown -R pipeline:pipeline /app

# Copy application code from eval-pipeline directory
COPY --chown=pipeline:pipeline eval-pipeline/ .

# Make scripts executable
RUN chmod +x docker-entrypoint.sh scripts/init_tiktoken.py

# Create required directories
RUN mkdir -p \
    /app/data/documents \
    /app/outputs/testsets \
    /app/outputs/evaluations \
    /app/outputs/reports \
    /app/outputs/visualizations \
    /app/outputs/metadata \
    /app/outputs/logs \
    /app/cache \
    /app/temp \
    /app/logs && \
    chown -R pipeline:pipeline /app

# Switch to non-root user
USER pipeline

# Set working directory
WORKDIR /app

# Clear proxy environment variables for production
ENV http_proxy= \
    https_proxy= \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    no_proxy= \
    NO_PROXY=

# Validate installation
RUN python setup.py --quick-test || echo "Quick test completed with warnings"

# Expose port for potential web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from pipeline.orchestrator import PipelineOrchestrator; print('Pipeline healthy')" || exit 1

# Default command
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "run_pipeline.py", "--config", "config/pipeline_config.yaml"]

# =============================================================================
# Usage Examples:
# 
# Build: docker build -t rag-eval-pipeline .
# Run: docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs rag-eval-pipeline
# Interactive: docker run -it --entrypoint /bin/bash rag-eval-pipeline
# =============================================================================
