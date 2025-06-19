#!/bin/bash

# Robust Docker Build Script with Network Resilience
# Handles proxy settings, network issues, and provides fallback options

set -e

echo "🔧 Robust Docker Build for RAG Evaluation Pipeline"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to test network connectivity
test_network() {
    print_status "Testing network connectivity..."
    
    # Test DNS resolution
    if nslookup pypi.org >/dev/null 2>&1; then
        print_success "DNS resolution working"
        return 0
    elif nslookup google.com >/dev/null 2>&1; then
        print_warning "DNS partially working but PyPI may be blocked"
        return 1
    else
        print_error "DNS resolution failed"
        return 2
    fi
}

# Function to detect proxy settings
detect_proxy() {
    print_status "Detecting proxy settings..."
    
    PROXY_FOUND=false
    
    if [[ -n "$HTTP_PROXY" ]] || [[ -n "$http_proxy" ]]; then
        print_success "HTTP proxy found: ${HTTP_PROXY:-$http_proxy}"
        export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
        PROXY_FOUND=true
    fi
    
    if [[ -n "$HTTPS_PROXY" ]] || [[ -n "$https_proxy" ]]; then
        print_success "HTTPS proxy found: ${HTTPS_PROXY:-$https_proxy}"
        export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
        PROXY_FOUND=true
    fi
    
    if [[ "$PROXY_FOUND" == false ]]; then
        print_warning "No proxy settings found"
        
        # Check if we're in a corporate environment
        if [[ -n "$CORPORATE_PROXY" ]]; then
            print_status "Using corporate proxy setting"
            export HTTP_PROXY="$CORPORATE_PROXY"
            export HTTPS_PROXY="$CORPORATE_PROXY"
            PROXY_FOUND=true
        fi
    fi
    
    return 0
}

# Function to build Docker image with different strategies
build_with_strategy() {
    local strategy=$1
    local image_name="rag-eval-pipeline"
    local build_args=""
    
    print_status "Building with strategy: $strategy"
    
    case $strategy in
        "direct")
            print_status "Building without proxy..."
            ;;
        "proxy")
            if [[ -n "$HTTP_PROXY" ]]; then
                build_args="--build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY"
                print_status "Building with proxy: $HTTP_PROXY"
            else
                print_error "No proxy available for proxy strategy"
                return 1
            fi
            ;;
        "minimal")
            print_status "Building minimal image with essential packages only..."
            # Create a minimal requirements file
            cat > requirements.minimal.txt << 'EOF'
pandas>=1.5.0
numpy>=1.21.0
PyYAML>=6.0
openpyxl>=3.0.10
sentence-transformers>=2.2.2
spacy>=3.4.0
scikit-learn>=1.1.0
keybert>=0.7.0
yake>=0.4.8
requests>=2.28.0
tqdm>=4.64.0
matplotlib>=3.5.0
EOF
            build_args="--build-arg REQUIREMENTS_FILE=requirements.minimal.txt"
            ;;
        "offline")
            print_status "Building offline-first image..."
            build_args="--build-arg OFFLINE_MODE=1"
            ;;
    esac
    
    # Execute build with timeout
    timeout 1800 docker build \
        --no-cache \
        --progress=plain \
        $build_args \
        -t $image_name \
        . 2>&1 | tee build.log
        
    return $?
}

# Function to verify build success
verify_build() {
    local image_name="rag-eval-pipeline"
    
    print_status "Verifying build..."
    
    if docker images | grep -q "$image_name"; then
        print_success "Image built successfully"
        
        # Test basic functionality
        print_status "Testing basic functionality..."
        if docker run --rm $image_name python -c "import pandas, numpy, yaml; print('Basic imports successful')"; then
            print_success "Basic functionality test passed"
            return 0
        else
            print_warning "Basic functionality test failed"
            return 1
        fi
    else
        print_error "Image not found after build"
        return 1
    fi
}

# Function to clean up failed builds
cleanup_failed_build() {
    print_status "Cleaning up failed build artifacts..."
    
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1 || true
    
    # Remove build cache
    docker builder prune -f >/dev/null 2>&1 || true
    
    print_success "Cleanup completed"
}

# Main execution
main() {
    print_status "Starting robust Docker build process..."
    
    # Test network connectivity
    network_status=0
    test_network || network_status=$?
    
    # Detect proxy settings
    detect_proxy
    
    # Determine build strategy based on network status
    if [[ $network_status -eq 0 ]]; then
        strategies=("direct" "proxy" "minimal")
    elif [[ $network_status -eq 1 ]]; then
        strategies=("proxy" "direct" "minimal")
    else
        strategies=("proxy" "minimal" "offline")
    fi
    
    # Try different build strategies
    for strategy in "${strategies[@]}"; do
        print_status "Attempting build with strategy: $strategy"
        
        if build_with_strategy "$strategy"; then
            if verify_build; then
                print_success "Build completed successfully with strategy: $strategy"
                
                # Show final image info
                echo ""
                print_status "Final image information:"
                docker images rag-eval-pipeline
                
                echo ""
                print_success "Build completed! You can now run:"
                echo "  docker run -it rag-eval-pipeline"
                echo "  docker compose up"
                
                exit 0
            else
                print_warning "Build succeeded but verification failed with strategy: $strategy"
            fi
        else
            print_error "Build failed with strategy: $strategy"
            cleanup_failed_build
        fi
        
        print_status "Trying next strategy..."
    done
    
    print_error "All build strategies failed"
    
    # Provide troubleshooting info
    echo ""
    print_error "Troubleshooting suggestions:"
    echo "1. Check your internet connection"
    echo "2. Verify proxy settings if behind corporate firewall"
    echo "3. Try setting proxy manually:"
    echo "   export HTTP_PROXY=http://your-proxy:port"
    echo "   export HTTPS_PROXY=http://your-proxy:port"
    echo "4. Check build.log for detailed error information"
    echo "5. Try building with reduced requirements:"
    echo "   ./build_robust.sh minimal"
    
    exit 1
}

# Handle command line arguments
if [[ $# -gt 0 ]]; then
    case $1 in
        "minimal")
            print_status "Forcing minimal build strategy..."
            build_with_strategy "minimal"
            verify_build
            exit $?
            ;;
        "proxy")
            print_status "Forcing proxy build strategy..."
            build_with_strategy "proxy"
            verify_build
            exit $?
            ;;
        "offline")
            print_status "Forcing offline build strategy..."
            build_with_strategy "offline"
            verify_build
            exit $?
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [strategy]"
            echo "Strategies:"
            echo "  minimal  - Build with minimal dependencies"
            echo "  proxy    - Build with proxy settings"
            echo "  offline  - Build for offline use"
            echo "  help     - Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown strategy: $1"
            exit 1
            ;;
    esac
fi

# Run main function
main
