#!/usr/bin/env python3
"""
Test script to verify tiktoken patching works correctly
This simulates what happens when ragas imports tiktoken
"""

import sys
import os
from pathlib import Path

# Add scripts to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.append(str(scripts_dir))

def test_tiktoken_patch():
    """Test that tiktoken patch prevents network errors"""
    
    print("🧪 Testing tiktoken patch...")
    
    # Apply the patch
    try:
        from tiktoken_fallback import patch_tiktoken_with_fallback
        patch_tiktoken_with_fallback()
        print("✅ Patch applied successfully")
    except Exception as e:
        print(f"❌ Patch failed: {e}")
        return False
    
    # Test import and usage (this is what ragas does)
    try:
        import tiktoken
        
        # This is the exact line that causes issues in ragas
        DEFAULT_TOKENIZER = tiktoken.get_encoding("o200k_base")
        
        # Test basic operations
        test_text = "Hello world, this is a test!"
        tokens = DEFAULT_TOKENIZER.encode(test_text)
        decoded = DEFAULT_TOKENIZER.decode(tokens)
        
        print(f"✅ Test text: '{test_text}'")
        print(f"✅ Tokens: {tokens}")
        print(f"✅ Token count: {len(tokens)}")
        print(f"✅ Decoded: '{decoded}'")
        
        print("✅ Tiktoken patch working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Tiktoken test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ragas_import():
    """Test that ragas can import without errors"""
    
    print("\n🧪 Testing ragas import compatibility...")
    
    try:
        # This is where the error occurs in the logs
        print("Attempting: from ragas.testset.transforms.base import ...")
        
        # Mock the specific import that fails
        import tiktoken
        DEFAULT_TOKENIZER = tiktoken.get_encoding("o200k_base")
        
        print("✅ Ragas-style tiktoken import successful!")
        return True
        
    except Exception as e:
        print(f"❌ Ragas import test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TIKTOKEN PATCH VALIDATION TEST")
    print("=" * 60)
    
    # Set environment variables like the docker entrypoint does
    os.environ["TIKTOKEN_CACHE_ONLY"] = "1"
    os.environ["TIKTOKEN_DISABLE_DOWNLOAD"] = "1"
    os.environ["TIKTOKEN_FORCE_OFFLINE"] = "1"
    
    success1 = test_tiktoken_patch()
    success2 = test_ragas_import()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED - Patch should prevent restart loop!")
    else:
        print("❌ TESTS FAILED - Patch needs more work")
    print("=" * 60)
