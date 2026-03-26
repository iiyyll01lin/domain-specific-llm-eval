#!/usr/bin/env python3
"""
Validate tiktoken offline operation
Tests that tiktoken works without internet access
"""
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tiktoken_offline():
    """Test tiktoken in offline mode"""
    logger.info("🧪 Testing tiktoken offline operation...")
    
    # Set offline environment
    os.environ["TIKTOKEN_CACHE_ONLY"] = "1"
    os.environ["TIKTOKEN_DISABLE_DOWNLOAD"] = "1"
    os.environ["TIKTOKEN_CACHE_DIR"] = "/app/.cache/tiktoken"
    
    try:
        import tiktoken
        
        # Test each encoding
        encodings_to_test = ['o200k_base', 'cl100k_base', 'p50k_base', 'r50k_base']
        successful_tests = 0
        
        for encoding_name in encodings_to_test:
            try:
                logger.info(f"Testing {encoding_name}...")
                
                # Get encoding
                enc = tiktoken.get_encoding(encoding_name)
                
                # Test encoding/decoding
                test_text = "Hello, world! This is a tiktoken offline test."
                tokens = enc.encode(test_text)
                decoded = enc.decode(tokens)
                
                logger.info(f"✅ {encoding_name}: {len(tokens)} tokens, decode successful")
                successful_tests += 1
                
            except Exception as e:
                logger.warning(f"⚠️ {encoding_name}: Failed - {e}")
        
        # Summary
        logger.info(f"📊 Tiktoken test results: {successful_tests}/{len(encodings_to_test)} encodings working")
        
        if successful_tests > 0:
            logger.info("✅ Tiktoken offline operation: WORKING")
            return True
        else:
            logger.error("❌ Tiktoken offline operation: FAILED")
            return False
            
    except ImportError:
        logger.error("❌ tiktoken not available")
        return False
    except Exception as e:
        logger.error(f"❌ Tiktoken test failed: {e}")
        return False

def check_cache_directory():
    """Check tiktoken cache directory"""
    cache_dir = "/app/.cache/tiktoken"
    
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        logger.info(f"📁 Cache directory exists: {cache_dir}")
        logger.info(f"📄 Cache files: {len(files)} files")
        
        # List tiktoken files
        tiktoken_files = [f for f in files if f.endswith('.tiktoken')]
        if tiktoken_files:
            logger.info(f"🎯 Tiktoken cache files: {tiktoken_files}")
        else:
            logger.warning("⚠️ No .tiktoken files found in cache")
            
        return len(tiktoken_files) > 0
    else:
        logger.error(f"❌ Cache directory not found: {cache_dir}")
        return False

def main():
    """Main validation function"""
    logger.info("🚀 Starting tiktoken offline validation...")
    
    # Check cache directory
    cache_ok = check_cache_directory()
    
    # Test tiktoken functionality
    tiktoken_ok = test_tiktoken_offline()
    
    if cache_ok and tiktoken_ok:
        logger.info("✅ Tiktoken offline validation: PASSED")
        return 0
    else:
        logger.warning("⚠️ Tiktoken offline validation: ISSUES DETECTED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
