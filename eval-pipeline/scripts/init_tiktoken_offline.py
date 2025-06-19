#!/usr/bin/env python3
"""
Initialize tiktoken with offline fallback for the RAG evaluation pipeline
This ensures tiktoken works even without internet connectivity
"""
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_tiktoken_offline():
    """Setup tiktoken for offline operation with fallback"""
    
    # Set environment variables for offline mode
    os.environ["TIKTOKEN_CACHE_ONLY"] = "1"
    os.environ["TIKTOKEN_DISABLE_DOWNLOAD"] = "1"
    os.environ["TIKTOKEN_FORCE_OFFLINE"] = "1"
    
    # Add scripts directory to path for fallback
    scripts_dir = "/app/scripts"
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    
    # ALWAYS use fallback in offline mode - don't even attempt real tiktoken
    logger.info("🔄 Activating tiktoken fallback for offline operation...")
    
    try:
        # Import and activate fallback immediately
        from tiktoken_fallback import patch_tiktoken_with_fallback
        patch_tiktoken_with_fallback()
        
        # Test fallback
        import tiktoken
        test_enc = tiktoken.get_encoding("cl100k_base")
        test_tokens = test_enc.encode("test")
        
        logger.info("✅ Tiktoken fallback activated and working")
        return True
        
    except Exception as fallback_error:
        logger.error(f"❌ Tiktoken fallback failed: {fallback_error}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Initializing tiktoken for offline operation...")
    
    success = setup_tiktoken_offline()
    
    if success:
        logger.info("✅ Tiktoken initialization completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Tiktoken initialization failed")
        sys.exit(1)
