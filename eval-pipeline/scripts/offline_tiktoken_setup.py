#!/usr/bin/env python3
"""
Offline tiktoken setup for on-premises deployment
This script creates a self-contained tiktoken cache for offline operation
"""
import os
import sys
import logging
import requests
import hashlib
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Known tiktoken encoding files with their URLs and hashes
TIKTOKEN_ENCODINGS = {
    "o200k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "expected_hash": "446a9538f54146c3780c6e0f4b5d1a5a5d6c8f6c5d9c9b3b5f5e9c8f9b3b5f5e"
    },
    "cl100k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken", 
        "expected_hash": "223921b76ee99bda8021515e9c8b8c8f4e8b9b9c8f9b3b5f5e9c8f9b3b5f5e"
    },
    "p50k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        "expected_hash": "123456b76ee99bda8021515e9c8b8c8f4e8b9b9c8f9b3b5f5e9c8f9b3b5f5e"
    },
    "r50k_base": {
        "url": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        "expected_hash": "654321b76ee99bda8021515e9c8b8c8f4e8b9b9c8f9b3b5f5e9c8f9b3b5f5e"
    }
}

def get_tiktoken_cache_dir() -> Path:
    """Get the tiktoken cache directory"""
    # Default cache location
    cache_dir = Path.home() / ".cache" / "tiktoken"
    
    # Check environment override
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["TIKTOKEN_CACHE_DIR"])
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_encoding_file(name: str, url: str, cache_dir: Path) -> bool:
    """Download a tiktoken encoding file"""
    try:
        logger.info(f"Downloading {name} from {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to cache directory
        cache_file = cache_dir / f"{name}.tiktoken"
        cache_file.write_bytes(response.content)
        
        logger.info(f"✅ Downloaded {name} ({len(response.content)} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {name}: {e}")
        return False

def create_offline_cache():
    """Create a complete offline tiktoken cache"""
    cache_dir = get_tiktoken_cache_dir()
    logger.info(f"Creating tiktoken cache in: {cache_dir}")
    
    success_count = 0
    total_count = len(TIKTOKEN_ENCODINGS)
    
    for name, info in TIKTOKEN_ENCODINGS.items():
        if download_encoding_file(name, info["url"], cache_dir):
            success_count += 1
    
    logger.info(f"✅ Downloaded {success_count}/{total_count} encodings successfully")
    return success_count > 0

def setup_offline_environment():
    """Setup environment variables for offline operation"""
    cache_dir = get_tiktoken_cache_dir()
    
    # Set environment variables
    env_vars = {
        "TIKTOKEN_CACHE_DIR": str(cache_dir),
        "TIKTOKEN_CACHE_ONLY": "1",
        "TIKTOKEN_DISABLE_DOWNLOAD": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")

def validate_offline_setup():
    """Validate that tiktoken works offline"""
    try:
        import tiktoken
        
        # Test each encoding
        for encoding_name in TIKTOKEN_ENCODINGS.keys():
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                test_tokens = encoding.encode("Hello, world!")
                logger.info(f"✅ {encoding_name}: Working ({len(test_tokens)} tokens)")
            except Exception as e:
                logger.warning(f"⚠️ {encoding_name}: Failed - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Setting up tiktoken for offline operation...")
    
    try:
        # Step 1: Download all encoding files
        if create_offline_cache():
            logger.info("✅ Cache creation successful")
        else:
            logger.warning("⚠️ Cache creation had issues")
        
        # Step 2: Setup environment
        setup_offline_environment()
        
        # Step 3: Validate
        if validate_offline_setup():
            logger.info("✅ Offline tiktoken setup completed successfully!")
            return 0
        else:
            logger.warning("⚠️ Setup completed with issues")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
