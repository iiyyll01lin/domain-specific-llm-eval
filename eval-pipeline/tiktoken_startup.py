#!/usr/bin/env python3
"""
Startup hook to patch tiktoken before any imports
This should be imported first in any Python process
"""

import sys
import os
from pathlib import Path

def ensure_tiktoken_patch():
    """Ensure tiktoken is patched before any imports can fail"""
    
    # Check if already patched
    if 'tiktoken' in sys.modules:
        tiktoken_module = sys.modules['tiktoken']
        if hasattr(tiktoken_module, '__class__') and 'Offline' in str(type(tiktoken_module)):
            # Already patched
            return True
    
    # Add scripts to path
    current_dir = Path(__file__).parent
    scripts_dir = current_dir / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
    
    # Also try relative path for when called from subdirectories
    scripts_dir_alt = current_dir.parent / "scripts"
    if scripts_dir_alt.exists():
        sys.path.insert(0, str(scripts_dir_alt))
    
    # Set environment variables
    os.environ.setdefault("TIKTOKEN_CACHE_ONLY", "1")
    os.environ.setdefault("TIKTOKEN_DISABLE_DOWNLOAD", "1") 
    os.environ.setdefault("TIKTOKEN_FORCE_OFFLINE", "1")
    
    try:
        from tiktoken_fallback import patch_tiktoken_with_fallback
        patch_tiktoken_with_fallback()
        return True
    except Exception as e:
        print(f"Warning: Could not apply tiktoken patch: {e}")
        return False

# Apply patch immediately on import
ensure_tiktoken_patch()
