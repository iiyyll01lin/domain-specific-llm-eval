#!/usr/bin/env python3
"""
Quick patch for tiktoken issues in the RAG evaluation pipeline
This script patches the tiktoken imports to be more resilient to network failures
"""
import os
import sys
from pathlib import Path

def patch_tiktoken_imports():
    """Patch tiktoken imports to handle network failures gracefully"""
    
    # List of files that need tiktoken patches
    files_to_patch = [
        "ragas/ragas/src/ragas/testset/transforms/base.py",
        "ragas/ragas/src/ragas/utils.py", 
        "rr-data/sft-dataset.py"
    ]
    
    workspace_root = Path("/mnt/d/workspace/domain-specific-llm-eval")
    
    for file_path in files_to_patch:
        full_path = workspace_root / file_path
        
        if not full_path.exists():
            print(f"⚠️ File not found: {full_path}")
            continue
            
        print(f"🔧 Patching: {full_path}")
        
        try:
            # Read the file
            content = full_path.read_text()
            
            # Create a backup
            backup_path = full_path.with_suffix(full_path.suffix + ".bak")
            backup_path.write_text(content)
            print(f"   📋 Backup created: {backup_path}")
            
            # Apply patches based on file type
            if "base.py" in str(full_path):
                content = patch_base_py(content)
            elif "utils.py" in str(full_path):
                content = patch_utils_py(content)
            elif "sft-dataset.py" in str(full_path):
                content = patch_sft_dataset_py(content)
            
            # Write the patched content
            full_path.write_text(content)
            print(f"   ✅ Patched successfully")
            
        except Exception as e:
            print(f"   ❌ Failed to patch: {e}")

def patch_base_py(content):
    """Patch ragas/testset/transforms/base.py"""
    
    # Replace tiktoken import with safe version
    if "import tiktoken" in content and "try:" not in content[:200]:
        content = content.replace(
            "import tiktoken\nfrom tiktoken.core import Encoding",
            """try:
    import tiktoken
    from tiktoken.core import Encoding
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("⚠️ tiktoken not available, using fallback")
    TIKTOKEN_AVAILABLE = False
    tiktoken = None
    Encoding = None"""
        )
        
        # Replace DEFAULT_TOKENIZER assignment
        if "DEFAULT_TOKENIZER = tiktoken.get_encoding" in content:
            content = content.replace(
                'DEFAULT_TOKENIZER = tiktoken.get_encoding("o200k_base")',
                '''try:
    DEFAULT_TOKENIZER = tiktoken.get_encoding("o200k_base") if TIKTOKEN_AVAILABLE else None
except Exception:
    print("⚠️ Failed to load tiktoken encoding, using None")
    DEFAULT_TOKENIZER = None'''
            )
    
    return content

def patch_utils_py(content):
    """Patch ragas/utils.py"""
    
    if "import tiktoken" in content and "try:" not in content[:200]:
        content = content.replace(
            "import tiktoken",
            """try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("⚠️ tiktoken not available in utils")
    tiktoken = None
    TIKTOKEN_AVAILABLE = False"""
        )
    
    return content

def patch_sft_dataset_py(content):
    """Patch rr-data/sft-dataset.py"""
    
    if "import tiktoken" in content and "try:" not in content[:500]:
        content = content.replace(
            "import tiktoken",
            """try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("⚠️ tiktoken not available in sft-dataset")
    tiktoken = None
    TIKTOKEN_AVAILABLE = False"""
        )
    
    return content

def main():
    print("🚀 Starting tiktoken patch process...")
    
    # Check if we're in the right directory
    workspace = Path("/mnt/d/workspace/domain-specific-llm-eval")
    if not workspace.exists():
        print(f"❌ Workspace not found: {workspace}")
        return 1
    
    print(f"📁 Working in: {workspace}")
    
    # Apply patches
    patch_tiktoken_imports()
    
    print("✅ Tiktoken patching completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
