#!/bin/bash
# Pre-download tiktoken encodings for offline Docker operation
set -e

echo "🚀 Pre-downloading tiktoken encodings for offline operation..."

# Create cache directory
mkdir -p /app/.cache/tiktoken

# Download encoding files using Python
python3 << 'EOF'
import sys
import os
import requests
from pathlib import Path

# Create tiktoken cache directory
cache_dir = Path('/app/.cache/tiktoken')
cache_dir.mkdir(parents=True, exist_ok=True)

# Encoding files to download
encodings = {
    'o200k_base': 'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken',
    'cl100k_base': 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken', 
    'p50k_base': 'https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken',
    'r50k_base': 'https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken'
}

success_count = 0
for name, url in encodings.items():
    try:
        print(f'📥 Downloading {name}...')
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        cache_file = cache_dir / f'{name}.tiktoken'
        cache_file.write_bytes(response.content)
        
        print(f'✅ Downloaded {name} ({len(response.content)} bytes)')
        success_count += 1
    except Exception as e:
        print(f'❌ Failed to download {name}: {e}')

print(f'📊 Downloaded {success_count}/{len(encodings)} encodings successfully')
if success_count > 0:
    print('✅ Tiktoken offline cache created')
else:
    print('⚠️ No encodings downloaded - tiktoken will attempt online download at runtime')
EOF

# Set file permissions
if [ -d "/app/.cache/tiktoken" ] && [ "$(ls -A /app/.cache/tiktoken 2>/dev/null)" ]; then
    chmod 644 /app/.cache/tiktoken/*.tiktoken 2>/dev/null || true
    echo "📋 Set file permissions for tiktoken cache"
fi

# Verify the files exist
echo "📁 Tiktoken cache contents:"
ls -la /app/.cache/tiktoken/ 2>/dev/null || echo "   (empty or missing)"

echo "✅ Tiktoken cache setup completed"
