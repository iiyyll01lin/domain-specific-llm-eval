#!/bin/bash
# Proxy-aware Docker build script

echo "🔧 Proxy-Aware Docker Build"
echo "=========================================="

# Function to test network connectivity
test_network() {
    echo "🌐 Testing network connectivity..."
    
    # Test DNS resolution
    if nslookup pypi.org > /dev/null 2>&1; then
        echo "✅ DNS resolution working"
    else
        echo "❌ DNS resolution failed"
        return 1
    fi
    
    # Test HTTP connectivity
    if curl -s --connect-timeout 10 https://pypi.org/simple/ > /dev/null; then
        echo "✅ HTTP connectivity working"
    else
        echo "❌ HTTP connectivity failed"
        return 1
    fi
    
    return 0
}

# Function to detect proxy settings
detect_proxy() {
    echo "🔍 Detecting proxy settings..."
    
    # Check environment variables
    if [[ -n "$HTTP_PROXY" || -n "$HTTPS_PROXY" ]]; then
        echo "✅ Proxy settings found in environment"
        echo "   HTTP_PROXY: ${HTTP_PROXY:-not set}"
        echo "   HTTPS_PROXY: ${HTTPS_PROXY:-not set}"
        return 0
    fi
    
    # Check common proxy configurations
    if [[ -f /etc/environment ]]; then
        if grep -q "http_proxy" /etc/environment; then
            echo "✅ Proxy settings found in /etc/environment"
            source /etc/environment
            return 0
        fi
    fi
    
    echo "⚠️ No proxy settings detected"
    return 1
}

# Function to build with proxy settings
build_with_proxy() {
    local http_proxy="$1"
    local https_proxy="$2"
    
    echo "🏗️ Building with proxy settings..."
    echo "   HTTP_PROXY: $http_proxy"
    echo "   HTTPS_PROXY: $https_proxy"
    
    # Update RAGAS submodule before building
    echo "📦 Managing RAGAS submodule..."
    
    # Use the manage-ragas script for proper submodule handling
    local script_dir="$(dirname "${BASH_SOURCE[0]}")"
    if [ -f "$script_dir/manage-ragas.sh" ]; then
        chmod +x "$script_dir/manage-ragas.sh"
        
        # First check status
        if ! "$script_dir/manage-ragas.sh" --status; then
            echo "🔄 Initializing RAGAS submodule..."
            "$script_dir/manage-ragas.sh" --init || echo "⚠️ Could not initialize RAGAS submodule"
        else
            echo "🔄 Updating RAGAS submodule..."
            "$script_dir/manage-ragas.sh" --update || echo "⚠️ Could not update RAGAS submodule"
        fi
    else
        # Fallback to manual git commands
        echo "⚠️ manage-ragas.sh not found, using fallback method..."
        cd "$(dirname "${BASH_SOURCE[0]}")/.." # Go to project root
        if [ -d "ragas" ]; then
            cd ragas
            git pull origin main || echo "⚠️ Could not update RAGAS submodule"
            cd ..
        else
            git submodule update --init --recursive || echo "⚠️ Could not initialize RAGAS submodule"
        fi
        cd "$(dirname "${BASH_SOURCE[0]}")" # Return to eval-pipeline directory
    fi
    
    # Prepare RAGAS for Docker build
    if ! prepare_ragas_build; then
        echo "❌ Failed to prepare RAGAS for build"
        return 1
    fi
    
    # Build from project root to include RAGAS submodule
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    echo "🏗️ Building from project root: $(pwd)"
    echo "🔍 RAGAS directory check:"
    if [ -d "ragas" ]; then
        echo "✅ RAGAS directory found: $(ls -la ragas/ | head -3)"
    else
        echo "❌ RAGAS directory not found!"
        return 1
    fi
    
    docker build \
        --build-arg HTTP_PROXY="$http_proxy" \
        --build-arg HTTPS_PROXY="$https_proxy" \
        --build-arg NO_PROXY="localhost,127.0.0.1" \
        -t rag-eval-pipeline:latest \
        -f eval-pipeline/Dockerfile \
        .
}

# Function to build without proxy
build_without_proxy() {
    echo "🏗️ Building without proxy..."
    
    # Build from project root to include RAGAS submodule
    cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    echo "🏗️ Building from project root: $(pwd)"
    echo "🔍 RAGAS directory check:"
    if [ -d "ragas" ]; then
        echo "✅ RAGAS directory found: $(ls -la ragas/ | head -3)"
    else
        echo "❌ RAGAS directory not found!"
        return 1
    fi
    
    docker build \
        -t rag-eval-pipeline:latest \
        -f eval-pipeline/Dockerfile \
        .
}

# Function to prepare RAGAS for build
prepare_ragas_build() {
    echo "📦 Preparing RAGAS for Docker build..."
    
    # Check if RAGAS exists in parent directory
    if [ -d "../ragas" ]; then
        echo "✅ Found RAGAS in parent directory, copying to build context..."
        
        # Remove any existing ragas directory in build context
        if [ -d "./ragas" ]; then
            rm -rf ./ragas
        fi
        
        # Copy RAGAS from parent directory to build context
        cp -r ../ragas ./ragas
        
        echo "✅ RAGAS copied to build context"
    elif [ -d "./ragas" ]; then
        echo "✅ RAGAS already in build context"
    else
        echo "❌ RAGAS not found in parent directory or build context!"
        echo "   Please run: cd .. && git submodule add https://github.com/explodinggradients/ragas.git ragas"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    echo "Starting Docker build process..."
    
    # Test network connectivity first
    if ! test_network; then
        echo "❌ Network connectivity issues detected"
        echo "🔧 Suggestions:"
        echo "   1. Check your internet connection"
        echo "   2. Configure proxy settings if behind corporate firewall"
        echo "   3. Try: export HTTP_PROXY=http://your-proxy:port"
        echo "   4. Try: export HTTPS_PROXY=http://your-proxy:port"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Detect and use proxy settings
    if detect_proxy; then
        build_with_proxy "$HTTP_PROXY" "$HTTPS_PROXY"
    else
        # Try to build without proxy
        build_without_proxy
    fi
    
    # Check build result
    if [[ $? -eq 0 ]]; then
        echo "✅ Docker build completed successfully!"
    else
        echo "❌ Docker build failed"
        echo "🔧 Try running with proxy settings:"
        echo "   export HTTP_PROXY=http://your-proxy:port"
        echo "   export HTTPS_PROXY=http://your-proxy:port"
        echo "   ./build_with_proxy.sh"
        exit 1
    fi
}

# Run main function
main "$@"
