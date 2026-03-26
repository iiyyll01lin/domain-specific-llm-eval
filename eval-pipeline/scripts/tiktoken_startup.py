#!/usr/bin/env python3
"""
Tiktoken startup module - automatically applies tiktoken patches on import
This ensures tiktoken fallback is available before any RAGAS imports
"""

def apply_tiktoken_patch():
    """Apply tiktoken fallback patch during startup only if needed"""
    try:
        # First test if tiktoken works normally
        import tiktoken
        from tiktoken.core import Encoding
        test_encoding = tiktoken.get_encoding("o200k_base")
        test_tokens = test_encoding.encode("test")
        
        print("✅ Tiktoken working normally - no patch needed")
        return True
        
    except Exception as e:
        print(f"⚠️ Tiktoken issue detected: {e}")
        print("🔧 Applying tiktoken fallback patch...")
        
        try:
            from tiktoken_fallback import patch_tiktoken_with_fallback
            patch_tiktoken_with_fallback()
            print("✅ Tiktoken fallback startup patch applied")
            return True
        except Exception as e2:
            print(f"❌ Startup tiktoken patch failed: {e2}")
            return False

# Auto-apply patch on module import
apply_tiktoken_patch()
