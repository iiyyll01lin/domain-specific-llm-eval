#!/bin/bash
# Quick fix to make your current container work with tiktoken offline

echo "🚀 Applying tiktoken offline fix to running container..."

# Set environment variables for offline operation
export TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
export TIKTOKEN_CACHE_ONLY=1
export TIKTOKEN_DISABLE_DOWNLOAD=1

# Configure the running container
docker exec -it rag-eval-pipeline bash -c "
# Create cache directory if it doesn't exist
mkdir -p /app/.cache/tiktoken

# Set environment variables
export TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
export TIKTOKEN_CACHE_ONLY=1
export TIKTOKEN_DISABLE_DOWNLOAD=1

# Add to bash profile for persistence
echo 'export TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken' >> ~/.bashrc
echo 'export TIKTOKEN_CACHE_ONLY=1' >> ~/.bashrc
echo 'export TIKTOKEN_DISABLE_DOWNLOAD=1' >> ~/.bashrc

echo '✅ Tiktoken environment variables configured'
echo '📁 Cache directory: /app/.cache/tiktoken'
"

# Test if tiktoken works in the container
echo "🧪 Testing tiktoken in container..."
docker exec -it rag-eval-pipeline python3 -c "
import os
os.environ['TIKTOKEN_CACHE_ONLY'] = '1'
os.environ['TIKTOKEN_DISABLE_DOWNLOAD'] = '1'
try:
    import tiktoken
    print('✅ Tiktoken import successful')
except Exception as e:
    print(f'⚠️ Tiktoken issue: {e}')
"

echo "✅ Container configured for offline tiktoken operation"
echo "🔄 Your container should now work without internet for tiktoken!"
echo ""
echo "💡 For permanent fix, rebuild with: ./deploy.sh --full --dev-advanced"
