#!/bin/bash
# Test Docker build with proxy settings

echo "🔧 Testing Docker Build with Proxy Settings"
echo "============================================="

# Check environment proxy settings
echo "📋 Current proxy environment:"
echo "  HTTP_PROXY: ${HTTP_PROXY:-not set}"
echo "  HTTPS_PROXY: ${HTTPS_PROXY:-not set}"
echo "  http_proxy: ${http_proxy:-not set}"
echo "  https_proxy: ${https_proxy:-not set}"

# Test proxy connectivity
echo ""
echo "🌐 Testing proxy connectivity..."
if curl -s --connect-timeout 5 --proxy "${HTTP_PROXY}" https://pypi.org/simple/ > /dev/null; then
    echo "✅ Proxy connection to PyPI working"
else
    echo "❌ Proxy connection to PyPI failed"
fi

# Build with explicit proxy settings
echo ""
echo "🏗️ Building Docker image with proxy..."
docker build \
    --build-arg HTTP_PROXY="${HTTP_PROXY}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY}" \
    --build-arg NO_PROXY="localhost,127.0.0.1" \
    --progress=plain \
    --no-cache \
    -t rag-eval-pipeline:proxy-test \
    -f Dockerfile \
    . 2>&1 | tee build-proxy-test.log

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "✅ Docker build with proxy completed successfully!"
    
    # Test the built image
    echo ""
    echo "🧪 Testing built image..."
    docker run --rm rag-eval-pipeline:proxy-test python -c "import pandas; print('✅ Pandas imported successfully')"
else
    echo "❌ Docker build with proxy failed"
    echo "📋 Check build-proxy-test.log for details"
fi
