#!/usr/bin/env python3
"""
Initialize tiktoken data for offline operation
Run this script during container startup to ensure tiktoken data is available
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pre_download_tiktoken_data():
    """Pre-download all common tiktoken encodings"""
    encodings_to_download = [
        "o200k_base",     # Latest GPT-4o
        "cl100k_base",    # GPT-4, GPT-3.5-turbo
        "p50k_base",      # GPT-3
        "r50k_base"       # GPT-3 davinci
    ]
    
    logger.info("Starting tiktoken data pre-download...")
    
    try:
        import tiktoken
        
        for encoding_name in encodings_to_download:
            try:
                logger.info(f"Downloading tiktoken encoding: {encoding_name}")
                encoding = tiktoken.get_encoding(encoding_name)
                
                # Test the encoding works
                test_text = "Hello, world!"
                tokens = encoding.encode(test_text)
                logger.info(f"✅ {encoding_name}: {len(tokens)} tokens for test text")
                
            except Exception as e:
                logger.warning(f"❌ Failed to download {encoding_name}: {e}")
                continue
                
        logger.info("✅ Tiktoken data pre-download completed successfully")
        return True
        
    except ImportError:
        logger.warning("⚠️ tiktoken not available for pre-download")
        return False
    except Exception as e:
        logger.error(f"❌ Tiktoken pre-download failed: {e}")
        return False

def setup_offline_mode():
    """Setup environment for offline tiktoken operation"""
    
    # Get the tiktoken cache directory
    try:
        import tiktoken
        # Use the correct method to get cache directory
        if hasattr(tiktoken, '_tiktoken'):
            # For newer versions
            cache_dir = os.path.expanduser("~/.cache/tiktoken")
        else:
            # Fallback to default location
            cache_dir = os.path.expanduser("~/.cache/tiktoken")
        
        logger.info(f"Tiktoken cache directory: {cache_dir}")
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Also set the environment variable for tiktoken to use this cache
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        
    except Exception as e:
        logger.warning(f"Could not determine tiktoken cache directory: {e}")
        # Set default cache directory
        cache_dir = os.path.expanduser("~/.cache/tiktoken")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    
    # Set environment variables for offline operation
    os.environ["TIKTOKEN_CACHE_ONLY"] = "1"
    os.environ["TIKTOKEN_DISABLE_DOWNLOAD"] = "1"
    logger.info("Set TIKTOKEN_CACHE_ONLY=1 and TIKTOKEN_DISABLE_DOWNLOAD=1 for offline operation")

def validate_setup():
    """Validate that tiktoken is working properly"""
    try:
        import tiktoken
        
        # Test primary encoding
        encoding = tiktoken.get_encoding("o200k_base")
        test_text = "This is a validation test for tiktoken offline operation."
        tokens = encoding.encode(test_text)
        
        logger.info(f"✅ Tiktoken validation successful: {len(tokens)} tokens")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tiktoken validation failed: {e}")
        return False

def main():
    """Main initialization function"""
    logger.info("🚀 Starting tiktoken initialization...")
    
    # Step 1: Pre-download data (if we have internet)
    download_success = pre_download_tiktoken_data()
    
    # Step 2: Setup offline mode
    setup_offline_mode()
    
    # Step 3: Validate setup
    validation_success = validate_setup()
    
    if validation_success:
        logger.info("✅ Tiktoken initialization completed successfully!")
        return 0
    else:
        logger.warning("⚠️ Tiktoken initialization completed with issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
