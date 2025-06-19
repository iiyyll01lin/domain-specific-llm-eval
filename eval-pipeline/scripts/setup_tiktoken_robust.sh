#!/bin/bash
# Robust Tiktoken Setup for Docker Build
# Works both with and without internet connectivity

set -e

echo "🚀 Setting up tiktoken for offline operation..."

# Create cache directory
CACHE_DIR="/app/.cache/tiktoken"
mkdir -p "$CACHE_DIR"

echo "📁 Cache directory: $CACHE_DIR"

# Set cache directory for tiktoken
export TIKTOKEN_CACHE_DIR="$CACHE_DIR"

# Create a simple test to check if we have internet
echo "🌐 Testing internet connectivity..."
if curl -s --head --connect-timeout 5 "https://openaipublic.blob.core.windows.net" >/dev/null 2>&1; then
    echo "✅ Internet available - downloading tiktoken cache"
    
    # Download tiktoken encodings using Python
    python3 << 'EOF'
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_tiktoken_encodings():
    """Download all common tiktoken encodings"""
    try:
        import tiktoken
        
        # Set cache directory
        cache_dir = "/app/.cache/tiktoken"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        
        encodings = ['o200k_base', 'cl100k_base', 'p50k_base', 'r50k_base']
        successful = 0
        
        logger.info(f"Starting download of {len(encodings)} tiktoken encodings...")
        
        for encoding_name in encodings:
            try:
                logger.info(f"📥 Downloading {encoding_name}...")
                enc = tiktoken.get_encoding(encoding_name)
                
                # Test the encoding works
                test_tokens = enc.encode("Hello, world!")
                logger.info(f"✅ {encoding_name}: {len(test_tokens)} tokens for test")
                successful += 1
                
            except Exception as e:
                logger.warning(f"❌ Failed to download {encoding_name}: {e}")
                continue
        
        logger.info(f"📊 Successfully downloaded {successful}/{len(encodings)} tiktoken encodings")
        
        if successful > 0:
            logger.info("✅ Tiktoken cache populated successfully")
            return True
        else:
            logger.warning("⚠️ No tiktoken encodings were downloaded")
            return False
            
    except ImportError:
        logger.error("❌ tiktoken not available")
        return False
    except Exception as e:
        logger.error(f"❌ Tiktoken cache setup failed: {e}")
        return False

if __name__ == "__main__":
    download_tiktoken_encodings()
EOF

else
    echo "⚠️ No internet connectivity - skipping tiktoken download"
    echo "🔧 Container will use offline-only mode"
fi

# Create a marker file to indicate setup was attempted
echo "$(date): Tiktoken setup completed" > "$CACHE_DIR/setup_complete.txt"

# Set permissions on cache directory
chown -R pipeline:pipeline "$CACHE_DIR" 2>/dev/null || true
chmod -R 755 "$CACHE_DIR" 2>/dev/null || true

echo "✅ Tiktoken setup completed"
echo "🔧 Cache location: $CACHE_DIR"
echo "🌐 Container configured for offline tiktoken operation"
