#!/bin/bash

# =============================================================================
# Docker Container Startup Script for RAG Evaluation Pipeline
# Handles initialization and graceful startup
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Initialization Functions
# =============================================================================

init_tiktoken() {
    print_info "Initializing tiktoken for offline operation..."
    
    # Check if we have tiktoken cache
    if [ -d "/app/.cache/tiktoken" ] && [ "$(ls -A /app/.cache/tiktoken 2>/dev/null)" ]; then
        print_info "Tiktoken cache found - using cached encodings"
    else
        print_warning "No tiktoken cache found - will use fallback tokenizer"
    fi
    
    # CRITICAL: Pre-patch tiktoken BEFORE any imports that might use it
    print_info "Pre-patching tiktoken to prevent import errors..."
    python -c "
import sys
import os
sys.path.append('/app/scripts')

# Import and immediately apply the fallback patch
try:
    from tiktoken_fallback import patch_tiktoken_with_fallback
    patch_tiktoken_with_fallback()
    print('✅ Tiktoken pre-patched for offline operation')
except Exception as e:
    print(f'❌ Failed to pre-patch tiktoken: {e}')
    sys.exit(1)
" || {
        print_error "Failed to pre-patch tiktoken"
        return 1
    }
    
    # Now run the full initialization script
    if python scripts/init_tiktoken_offline.py; then
        print_success "Tiktoken initialization completed successfully"
        return 0
    else
        print_warning "Tiktoken initialization completed with warnings"
        return 1
    fi
}

setup_environment() {
    print_info "Setting up container environment..."
    
    # Set tiktoken environment variables for offline operation GLOBALLY
    export TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
    export TIKTOKEN_CACHE_ONLY=1
    export TIKTOKEN_DISABLE_DOWNLOAD=1
    export TIKTOKEN_FORCE_OFFLINE=1
    
    # Also set Python path to ensure scripts are available
    export PYTHONPATH="/app/scripts:/app/src:${PYTHONPATH:-}"
    
    print_info "Tiktoken configured for offline operation"
    print_info "Cache directory: $TIKTOKEN_CACHE_DIR"
    print_info "Python path: $PYTHONPATH"
    
    # Create necessary directories if they don't exist
    mkdir -p \
        /app/data/documents \
        /app/outputs/testsets \
        /app/outputs/evaluations \
        /app/outputs/reports \
        /app/outputs/visualizations \
        /app/outputs/metadata \
        /app/outputs/logs \
        /app/cache \
        /app/temp \
        /app/logs
    
    print_success "Environment setup completed"
}

validate_installation() {
    print_info "Validating pipeline installation..."
    
    # First, check and fix any missing packages
    print_info "Checking for missing packages..."
    python scripts/fix_missing_packages.py
    if [ $? -ne 0 ]; then
        print_warning "Some packages are still missing, but continuing..."
    fi
    
    # Test basic Python imports
    python -c "
import sys
import os
sys.path.append('/app/src')

try:
    import pandas
    import numpy
    print('✅ Core data libraries available')
except ImportError as e:
    print(f'❌ Core data libraries missing: {e}')
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print('✅ Sentence transformers available')
except ImportError as e:
    print(f'⚠️ Sentence transformers not available: {e}')

try:
    import tiktoken
    print('✅ Tiktoken available')
except ImportError as e:
    print(f'⚠️ Tiktoken not available: {e}')
" || {
        print_error "Installation validation failed"
        return 1
    }
    
    print_success "Installation validation completed"
}

print_startup_info() {
    echo "============================================"
    echo "  RAG Evaluation Pipeline - Container"
    echo "============================================"
    echo ""
    echo "Container: rag-eval-pipeline"
    echo "Python: $(python --version)"
    echo "Working Directory: $(pwd)"
    echo "User: $(whoami)"
    echo ""
    echo "Available Commands:"
    echo "  - Main Pipeline: python run_pipeline.py --config config/pipeline_config.yaml"
    echo "  - Test Documents: python test_custom_documents.py"
    echo "  - Setup Test: python setup.py --quick-test"
    echo ""
}

# =============================================================================
# Main Startup Function
# =============================================================================

main() {
    print_startup_info
    
    # Step 0: CRITICAL - Pre-patch tiktoken before any Python imports
    print_info "Pre-patching tiktoken before application startup..."
    python -c "
import sys
import os
sys.path.append('/app/scripts')
from tiktoken_fallback import patch_tiktoken_with_fallback
patch_tiktoken_with_fallback()
print('✅ Tiktoken globally patched for all imports')
" || {
        print_error "Failed to pre-patch tiktoken globally"
        exit 1
    }
    
    # Step 1: Setup environment
    setup_environment
    
    # Step 2: Initialize tiktoken (non-blocking)
    init_tiktoken || true
    
    # Step 3: Validate installation
    validate_installation || {
        print_error "Container validation failed"
        exit 1
    }
    
    print_success "Container initialization completed successfully!"
    print_info "Starting application..."
    echo ""
    
    # Execute the command passed to the container
    exec "$@"
}

# Handle different startup modes
case "${1:-default}" in
    "bash"|"sh"|"/bin/bash"|"/bin/sh")
        print_info "Starting interactive shell..."
        exec "$@"
        ;;
    "test")
        print_info "Running test mode..."
        main python test_custom_documents.py
        ;;
    "validate")
        print_info "Running validation only..."
        setup_environment
        validate_installation
        print_success "Validation completed successfully!"
        ;;
    *)
        # Default: Run full initialization and start application
        main "$@"
        ;;
esac
