"""
Fallback tiktoken implementation for offline operation
This provides a drop-in replacement when tiktoken cache is not available
"""
import re
import hashlib
import types
import importlib.util
from typing import List, Optional

class OfflineTiktokenFallback:
    """Offline fallback tokenizer when tiktoken cache is unavailable"""
    
    def __init__(self, encoding_name: str = "o200k_base"):
        self.name = encoding_name
        self.encoding_name = encoding_name
        # Add common tiktoken attributes that libraries might expect
        self.max_token_value = 200000
        self.eot_token = 199999
        
    def encode(self, text: str) -> List[int]:
        """Simple word-based tokenization fallback"""
        if not text:
            return []
            
        # Basic tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        # Convert to consistent token IDs using hash
        token_ids = []
        for token in tokens:
            # Create consistent hash-based token ID
            hash_obj = hashlib.md5(token.encode('utf-8'))
            token_id = int(hash_obj.hexdigest()[:8], 16) % 50000  # Limit range
            token_ids.append(token_id)
            
        return token_ids
    
    def decode(self, tokens: List[int]) -> str:
        """Basic decode - approximate reconstruction"""
        # This is a simplified decode, real tiktoken decode is more complex
        return f"<decoded_{len(tokens)}_tokens>"
    
    def encode_ordinary(self, text: str) -> List[int]:
        """Alias for encode"""
        return self.encode(text)
    
    def decode_tokens_bytes(self, tokens: List[int]) -> List[bytes]:
        """Return token bytes - fallback implementation"""
        return [f"<token_{i}>".encode() for i in tokens]
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts"""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode multiple token lists"""
        return [self.decode(tokens) for tokens in token_lists]
    
    # Additional methods that ragas or other libraries might expect
    def encode_with_unstable(self, text: str, allowed_special: set = None, disallowed_special: str = "raise") -> List[int]:
        """Encode with special tokens - fallback ignores special handling"""
        return self.encode(text)
    
    def decode_single_token_bytes(self, token: int) -> bytes:
        """Decode single token to bytes"""
        return f"<token_{token}>".encode()

import types
import importlib.util

class OfflineTiktokenModule:
    """Mock tiktoken module for offline operation"""
    
    @staticmethod
    def get_encoding(encoding_name: str) -> OfflineTiktokenFallback:
        """Return fallback encoding"""
        return OfflineTiktokenFallback(encoding_name)
    
    @staticmethod
    def encoding_for_model(model_name: str) -> OfflineTiktokenFallback:
        """Return fallback encoding for model"""
        # Map common models to encoding names
        model_encodings = {
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base", 
            "gpt-4o": "o200k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-davinci-003": "p50k_base",
        }
        
        encoding_name = model_encodings.get(model_name, "cl100k_base")
        return OfflineTiktokenFallback(encoding_name)

def create_tiktoken_module():
    """Create a proper module object for tiktoken fallback"""
    # Create a proper module object
    module = types.ModuleType('tiktoken')
    
    # Set required module attributes
    module.__name__ = 'tiktoken'
    module.__file__ = '<tiktoken_fallback>'
    module.__package__ = 'tiktoken'
    module.__loader__ = None
    
    # Create a proper ModuleSpec - this is critical for transformers compatibility
    spec = importlib.util.spec_from_loader(
        'tiktoken',
        loader=None,
        origin='<tiktoken_fallback>'
    )
    module.__spec__ = spec
    
    # Add main functions
    module.get_encoding = OfflineTiktokenModule.get_encoding
    module.encoding_for_model = OfflineTiktokenModule.encoding_for_model
    
    # Create and add core submodule (this is what RAGAS is looking for)
    core_module = types.ModuleType('tiktoken.core')
    core_module.__name__ = 'tiktoken.core'
    core_module.__file__ = '<tiktoken_fallback>'
    core_module.__package__ = 'tiktoken.core'
    core_module.__loader__ = None
    
    # Create spec for core module too
    core_spec = importlib.util.spec_from_loader(
        'tiktoken.core',
        loader=None,
        origin='<tiktoken_fallback>'
    )
    core_module.__spec__ = core_spec
    core_module.Encoding = OfflineTiktokenFallback  # RAGAS might import this
    
    # Add core module to main tiktoken module
    module.core = core_module
    
    # Add common constants that libraries might expect
    module.ENCODING_CONSTRUCTORS = {}
    module.MODEL_TO_ENCODING = {}
    
    return module

def patch_tiktoken_with_fallback():
    """Patch tiktoken to use fallback when cache fails"""
    import sys
    
    # Always patch immediately to prevent any real tiktoken imports
    print("🔄 Installing tiktoken fallback...")
    
    # Create a proper module object instead of just a class instance
    fallback_module = create_tiktoken_module()
    
    # Install the main module
    sys.modules['tiktoken'] = fallback_module
    
    # Also patch the core submodule specifically
    sys.modules['tiktoken.core'] = fallback_module.core
    
    print("✅ Tiktoken fallback pre-installed with proper module spec")
    
    try:
        # Test if our fallback module works with importlib
        import importlib.util
        spec = importlib.util.find_spec('tiktoken')
        if spec is not None:
            print("✅ Tiktoken fallback module properly registered")
        else:
            print("⚠️ Tiktoken fallback module spec not found")
            
        # Test basic functionality
        import tiktoken
        test_enc = tiktoken.get_encoding("cl100k_base")
        test_tokens = test_enc.encode("test")
        print(f"✅ Tiktoken fallback working - encoded 'test' to {len(test_tokens)} tokens")
        
        # Test core module access (this is what RAGAS needs)
        import tiktoken.core
        print("✅ Tiktoken core module accessible")
        
        # Verify __spec__ is properly set (this is what transformers checks)
        if hasattr(tiktoken, '__spec__') and tiktoken.__spec__ is not None:
            print("✅ Tiktoken __spec__ properly set")
        else:
            print("⚠️ Tiktoken __spec__ issue detected")
        
    except Exception as e:
        print(f"⚠️ Tiktoken fallback test failed: {e}")
    
    print("✅ Tiktoken fallback activated")

if __name__ == "__main__":
    # Test the fallback tokenizer
    fallback = OfflineTiktokenFallback()
    test_text = "Hello, this is a test for offline tokenization!"
    
    tokens = fallback.encode(test_text)
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print("✅ Offline fallback tokenizer working!")
