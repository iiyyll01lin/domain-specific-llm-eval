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
    
    # Debug: Show what build args will be passed
    echo "🔍 Build arguments that will be passed:"
    echo "   --build-arg HTTP_PROXY=\"$http_proxy\""
    echo "   --build-arg HTTPS_PROXY=\"$https_proxy\""
    echo "   --build-arg NO_PROXY=\"localhost,127.0.0.1\""
    
    docker build \
        --build-arg HTTP_PROXY="$http_proxy" \
        --build-arg HTTPS_PROXY="$https_proxy" \
        --build-arg NO_PROXY="localhost,127.0.0.1" \
        --progress=plain \
        -t rag-eval-pipeline:latest \
        . 2>&1 | tee build-with-proxy.log
        
    local build_result=$?
    
    if [ $build_result -eq 0 ]; then
        echo "✅ Build completed successfully!"
    else
        echo "❌ Build failed with exit code: $build_result"
        echo "📋 Check build-with-proxy.log for details"
        
        # Show last 20 lines of log for quick debugging
        echo ""
        echo "📋 Last 20 lines of build log:"
        tail -n 20 build-with-proxy.log
    fi
    
    return $build_result
}

# Function to build without proxy
build_without_proxy() {
    echo "🏗️ Building without proxy..."
    
    docker build \
        -t rag-eval-pipeline:latest \
        .
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
