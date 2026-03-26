#!/bin/bash

# =============================================================================
# Docker Health Check Script for RAG Evaluation Pipeline
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Health check configuration
HEALTH_CHECK_TIMEOUT=30
HEALTH_CHECK_RETRIES=3

# =============================================================================
# Health Check Functions
# =============================================================================

check_python_environment() {
    echo "Checking Python environment..."
    
    # Check Python version
    python --version || return 1
    
    # Check critical imports
    python -c "
import sys
import os
sys.path.append('/app/src')

try:
    # Core dependencies
    import pandas
    import numpy
    import yaml
    import requests
    
    # NLP dependencies
    import spacy
    import nltk
    from sentence_transformers import SentenceTransformer
    
    # Pipeline modules
    from pipeline.orchestrator import PipelineOrchestrator
    from data.document_processor import DocumentProcessor
    
    print('✅ All critical modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
    " || return 1
    
    return 0
}

check_models() {
    echo "Checking NLP models..."
    
    # Check spaCy models
    python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy model loaded successfully')
except OSError as e:
    print(f'❌ spaCy model error: {e}')
    exit(1)
    " || return 1
    
    # Check sentence-transformers model
    python -c "
from sentence_transformers import SentenceTransformer
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Sentence transformer model loaded successfully')
except Exception as e:
    print(f'❌ Sentence transformer error: {e}')
    exit(1)
    " || return 1
    
    return 0
}

check_directories() {
    echo "Checking required directories..."
    
    required_dirs=(
        "/app/src"
        "/app/config"
        "/app/data"
        "/app/outputs"
        "/app/cache"
        "/app/logs"
        "/app/temp"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "❌ Missing directory: $dir"
            return 1
        fi
    done
    
    # Check write permissions
    touch /app/outputs/.health_check_test 2>/dev/null || {
        echo "❌ Cannot write to outputs directory"
        return 1
    }
    rm -f /app/outputs/.health_check_test
    
    touch /app/logs/.health_check_test 2>/dev/null || {
        echo "❌ Cannot write to logs directory"
        return 1
    }
    rm -f /app/logs/.health_check_test
    
    echo "✅ All directories accessible"
    return 0
}

check_configuration() {
    echo "Checking configuration..."
    
    # Check if main config exists
    if [ ! -f "/app/config/pipeline_config.yaml" ]; then
        echo "❌ Main configuration file not found"
        return 1
    fi
    
    # Validate configuration structure
    python -c "
import yaml
import sys

try:
    with open('/app/config/pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    required_sections = ['pipeline', 'data_sources', 'testset_generation', 'evaluation', 'output']
    missing_sections = []
    
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        print(f'❌ Missing configuration sections: {missing_sections}')
        sys.exit(1)
    
    print('✅ Configuration structure valid')
except Exception as e:
    print(f'❌ Configuration validation error: {e}')
    sys.exit(1)
    " || return 1
    
    return 0
}

check_disk_space() {
    echo "Checking disk space..."
    
    # Check available disk space (require at least 1GB)
    available_space=$(df /app | awk 'NR==2 {print $4}')
    required_space=1048576  # 1GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        echo "❌ Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB"
        return 1
    fi
    
    echo "✅ Sufficient disk space available: ${available_space}KB"
    return 0
}

check_memory() {
    echo "Checking memory usage..."
    
    # Check available memory
    if [ -f "/proc/meminfo" ]; then
        available_mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        required_mem=1048576  # 1GB in KB
        
        if [ "$available_mem" -lt "$required_mem" ]; then
            echo "⚠️ Low available memory: ${available_mem}KB"
            # Don't fail on low memory, just warn
        else
            echo "✅ Sufficient memory available: ${available_mem}KB"
        fi
    else
        echo "ℹ️ Memory info not available"
    fi
    
    return 0
}

run_basic_functionality_test() {
    echo "Running basic functionality test..."
    
    # Create test configuration
    python -c "
import sys
import os
sys.path.append('/app/src')

try:
    from pipeline.orchestrator import PipelineOrchestrator
    from data.document_processor import DocumentProcessor
    
    # Test document processor
    processor = DocumentProcessor({})
    
    # Test basic pipeline initialization
    test_config = {
        'pipeline': {'name': 'health_check'},
        'data_sources': {'documents': {'primary_docs': []}},
        'testset_generation': {'samples_per_document': 1},
        'evaluation': {'methods': {'contextual_keywords': True}},
        'output': {'base_dir': '/app/outputs'}
    }
    
    orchestrator = PipelineOrchestrator(
        config=test_config,
        run_id='health_check',
        output_dirs={'base': '/app/outputs/health_check'}
    )
    
    print('✅ Basic functionality test passed')
except Exception as e:
    print(f'❌ Functionality test failed: {e}')
    sys.exit(1)
    " || return 1
    
    return 0
}

# =============================================================================
# Main Health Check Function
# =============================================================================

run_health_check() {
    echo -e "${GREEN}Starting RAG Evaluation Pipeline Health Check...${NC}"
    echo "================================================="
    
    local checks=(
        "check_python_environment"
        "check_models"
        "check_directories" 
        "check_configuration"
        "check_disk_space"
        "check_memory"
        "run_basic_functionality_test"
    )
    
    local failed_checks=0
    
    for check in "${checks[@]}"; do
        echo ""
        if ! $check; then
            echo -e "${RED}❌ Health check failed: $check${NC}"
            ((failed_checks++))
        else
            echo -e "${GREEN}✅ Health check passed: $check${NC}"
        fi
    done
    
    echo ""
    echo "================================================="
    
    if [ $failed_checks -eq 0 ]; then
        echo -e "${GREEN}🎉 All health checks passed! Pipeline is healthy.${NC}"
        return 0
    else
        echo -e "${RED}❌ $failed_checks health check(s) failed. Pipeline may not function correctly.${NC}"
        return 1
    fi
}

# =============================================================================
# Script Execution
# =============================================================================

# Run health check
if run_health_check; then
    exit 0
else
    exit 1
fi
