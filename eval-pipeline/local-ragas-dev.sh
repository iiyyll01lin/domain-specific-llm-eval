#!/bin/bash

# =============================================================================
# Local RAGAS Development Helper Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local missing_deps=()
    
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies and try again."
        return 1
    fi
    
    print_success "All prerequisites met"
}

# Function to setup local RAGAS development environment
setup_local_ragas() {
    print_header "Setting up Local RAGAS Development"
    
    # Ensure we're in the project root
    cd "$PROJECT_ROOT"
    
    # Initialize/update RAGAS submodule
    if [ -f "eval-pipeline/manage-ragas.sh" ]; then
        chmod +x eval-pipeline/manage-ragas.sh
        eval-pipeline/manage-ragas.sh --init
    else
        print_warning "manage-ragas.sh not found, using git directly"
        if [ ! -d "ragas" ]; then
            git submodule add https://github.com/explodinggradients/ragas.git ragas
        fi
        git submodule init ragas
        git submodule update ragas
    fi
    
    # Create local Python environment for RAGAS development
    print_info "Setting up local Python environment..."
    
    if [ ! -d "venv-ragas" ]; then
        python3 -m venv venv-ragas
        print_success "Created virtual environment: venv-ragas"
    fi
    
    # Activate virtual environment and install RAGAS in development mode
    source venv-ragas/bin/activate
    
    cd ragas
    
    # Handle nested RAGAS structure
    if [ -d "ragas" ]; then
        cd ragas
    fi
    
    # Install RAGAS in editable mode
    if [ -f "pyproject.toml" ]; then
        print_info "Installing RAGAS from pyproject.toml in editable mode..."
        pip install -e "."
        pip install -e ".[all]" || print_warning "Could not install all optional dependencies"
    else
        print_error "No pyproject.toml found in RAGAS directory"
        return 1
    fi
    
    deactivate
    
    print_success "Local RAGAS development environment setup complete"
    print_info "To use the environment: source venv-ragas/bin/activate"
}

# Function to build Docker image with local RAGAS
build_docker_local() {
    print_header "Building Docker Image with Local RAGAS"
    
    cd "$SCRIPT_DIR"
    
    # Build using the proxy-aware build script
    if [ -f "build-with-proxy.sh" ]; then
        chmod +x build-with-proxy.sh
        ./build-with-proxy.sh
    else
        print_warning "build-with-proxy.sh not found, using direct docker build"
        docker build -t rag-eval-pipeline:latest .
    fi
}

# Function to run development container with local RAGAS mounted
run_dev_container() {
    print_header "Running Development Container"
    
    cd "$SCRIPT_DIR"
    
    print_info "Starting development container with local RAGAS mounted..."
    
    docker run -it --rm \
        -v "$(pwd)/../ragas:/app/ragas" \
        -v "$(pwd)/../data:/app/data" \
        -v "$(pwd)/../outputs:/app/outputs" \
        -p 8080:8080 \
        --name rag-eval-dev \
        rag-eval-pipeline:latest \
        /bin/bash
}

# Function to test local RAGAS installation
test_ragas_installation() {
    print_header "Testing RAGAS Installation"
    
    # Test in Docker container
    print_info "Testing RAGAS in Docker container..."
    
    docker run --rm \
        -v "$(pwd)/../ragas:/app/ragas" \
        rag-eval-pipeline:latest \
        python -c "
import sys
print(f'Python version: {sys.version}')
try:
    import ragas
    print(f'✅ RAGAS version: {ragas.__version__}')
    print(f'✅ RAGAS location: {ragas.__file__}')
    
    # Test basic functionality
    from ragas.evaluation import evaluate
    print('✅ RAGAS evaluate function imported successfully')
    
    # List available metrics
    print('📋 Testing RAGAS components...')
    try:
        from ragas.metrics import answer_relevancy, faithfulness, context_recall
        print('✅ Core metrics imported successfully')
    except ImportError as e:
        print(f'⚠️ Some metrics failed to import: {e}')
    
    print('🎉 RAGAS installation test completed successfully')
    
except ImportError as e:
    print(f'❌ RAGAS import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ RAGAS test failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "RAGAS installation test passed"
    else
        print_error "RAGAS installation test failed"
        return 1
    fi
}

# Function to show local development workflow
show_workflow() {
    print_header "Local RAGAS Development Workflow"
    
    echo "🔧 Development Setup:"
    echo "  1. Run './local-ragas-dev.sh --setup' to initialize"
    echo "  2. Activate local environment: 'source venv-ragas/bin/activate'"
    echo "  3. Make changes to RAGAS code in ./ragas/"
    echo ""
    echo "🐳 Docker Development:"
    echo "  1. Build image: './local-ragas-dev.sh --build'"
    echo "  2. Run dev container: './local-ragas-dev.sh --run-dev'"
    echo "  3. Test installation: './local-ragas-dev.sh --test'"
    echo ""
    echo "📁 Directory Structure:"
    echo "  - ./ragas/              # RAGAS submodule (your local development)"
    echo "  - ./venv-ragas/         # Local Python environment"
    echo "  - ./eval-pipeline/      # Main pipeline code"
    echo ""
    echo "🔀 Workflow Examples:"
    echo "  # Full setup"
    echo "  ./local-ragas-dev.sh --setup --build --test"
    echo ""
    echo "  # Update and rebuild"
    echo "  ./local-ragas-dev.sh --update --build"
    echo ""
    echo "  # Development session"
    echo "  ./local-ragas-dev.sh --run-dev"
}

# Function to update RAGAS submodule
update_ragas() {
    print_header "Updating RAGAS Submodule"
    
    if [ -f "$SCRIPT_DIR/manage-ragas.sh" ]; then
        chmod +x "$SCRIPT_DIR/manage-ragas.sh"
        "$SCRIPT_DIR/manage-ragas.sh" --update
    else
        print_warning "manage-ragas.sh not found, using git directly"
        cd "$PROJECT_ROOT/ragas"
        git fetch origin
        git checkout main
        git pull origin main
        cd "$PROJECT_ROOT"
        git add ragas
    fi
}

# Function to clean up development environment
cleanup() {
    print_header "Cleaning Up Development Environment"
    
    cd "$PROJECT_ROOT"
    
    print_info "Removing virtual environment..."
    if [ -d "venv-ragas" ]; then
        rm -rf venv-ragas
        print_success "Removed venv-ragas"
    fi
    
    print_info "Cleaning Docker resources..."
    docker system prune -f || print_warning "Could not clean Docker resources"
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Local RAGAS Development Helper"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup      Setup local RAGAS development environment"
    echo "  --build      Build Docker image with local RAGAS"
    echo "  --run-dev    Run development container"
    echo "  --test       Test RAGAS installation"
    echo "  --update     Update RAGAS submodule"
    echo "  --workflow   Show development workflow guide"
    echo "  --cleanup    Clean up development environment"
    echo "  --help       Show this help message"
    echo ""
    echo "Combined options:"
    echo "  --setup --build --test    # Full setup and test"
    echo "  --update --build          # Update and rebuild"
    echo ""
    echo "Examples:"
    echo "  $0 --setup              # Initialize development environment"
    echo "  $0 --build              # Build Docker image"
    echo "  $0 --run-dev            # Start development container"
    echo "  $0 --workflow           # Show workflow guide"
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        show_workflow
        return 0
    fi
    
    # Check prerequisites first
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Process arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup)
                setup_local_ragas
                shift
                ;;
            --build)
                build_docker_local
                shift
                ;;
            --run-dev)
                run_dev_container
                shift
                ;;
            --test)
                test_ragas_installation
                shift
                ;;
            --update)
                update_ragas
                shift
                ;;
            --workflow)
                show_workflow
                shift
                ;;
            --cleanup)
                cleanup
                shift
                ;;
            --help)
                show_usage
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Run main function
main "$@"
