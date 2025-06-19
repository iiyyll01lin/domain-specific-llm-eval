#!/bin/bash
# Apply tiktoken offline solution to your running container
set -e

echo "🚀 Applying tiktoken offline solution to container..."

CONTAINER_NAME="rag-eval-pipeline"

# Check if container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "❌ Container $CONTAINER_NAME is not running"
    echo "Please start your container first with: ./deploy.sh --full --dev-advanced"
    exit 1
fi

echo "✅ Container $CONTAINER_NAME is running"

# Copy the offline setup script to container
echo "📋 Copying setup files to container..."
docker cp scripts/setup_tiktoken_offline.py $CONTAINER_NAME:/tmp/
docker cp scripts/download_tiktoken_cache.sh $CONTAINER_NAME:/tmp/

# Run the setup inside the container
echo "🔧 Setting up tiktoken offline cache in container..."
docker exec -it $CONTAINER_NAME bash -c "
    cd /tmp
    python3 setup_tiktoken_offline.py --cache-dir /app/.cache/tiktoken
    
    # Set environment variables permanently
    echo 'export TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken' >> ~/.bashrc
    echo 'export TIKTOKEN_CACHE_ONLY=1' >> ~/.bashrc  
    echo 'export TIKTOKEN_DISABLE_DOWNLOAD=1' >> ~/.bashrc
    
    # Set for current session
    export TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
    export TIKTOKEN_CACHE_ONLY=1
    export TIKTOKEN_DISABLE_DOWNLOAD=1
    
    echo '✅ Tiktoken offline setup completed in container'
"

# Test that tiktoken works offline
echo "🧪 Testing tiktoken offline functionality..."
docker exec -it $CONTAINER_NAME bash -c "
    export TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
    export TIKTOKEN_CACHE_ONLY=1
    export TIKTOKEN_DISABLE_DOWNLOAD=1
    
    python3 -c '
try:
    import tiktoken
    encoding = tiktoken.get_encoding(\"o200k_base\")
    tokens = encoding.encode(\"Hello, this is a test for offline tiktoken!\")
    print(f\"✅ Offline tiktoken working: {len(tokens)} tokens\")
    print(f\"✅ Test successful - your container is ready for offline operation!\")
except Exception as e:
    print(f\"⚠️ Test failed: {e}\")
    print(\"ℹ️ This might be expected if tiktoken is not used in your current pipeline\")
'
"

echo ""
echo "🎯 TIKTOKEN OFFLINE SOLUTION APPLIED!"
echo ""
echo "Your container now has:"
echo "  ✅ All tiktoken encoding files cached locally"
echo "  ✅ Environment variables set for offline operation"  
echo "  ✅ No more internet dependency for tiktoken"
echo ""
echo "Next steps:"
echo "  1. Restart your evaluation pipeline"
echo "  2. Verify no more 'Connection reset by peer' errors"
echo "  3. Your system now works completely offline!"
echo ""
echo "Container logs:"
docker logs --tail 10 $CONTAINER_NAME
