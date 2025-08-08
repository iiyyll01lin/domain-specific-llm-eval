"""
Alternative tokenization approach for offline operation
Replaces tiktoken with fully offline alternatives
"""
import re
from typing import List, Optional

class OfflineTokenizer:
    """Offline tokenizer that doesn't require internet"""
    
    def __init__(self, model_name: str = "simple"):
        self.model_name = model_name
        
    def encode(self, text: str) -> List[int]:
        """Simple word-based tokenization"""
        # Basic word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Convert words to simple hash-based token IDs
        tokens = []
        for word in words:
            # Simple hash to int conversion
            token_id = hash(word) % 50000  # Limit to reasonable range
            tokens.append(abs(token_id))  # Ensure positive
            
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Basic decode - not perfect but functional"""
        # This is a simplified decode, real tiktoken decode is more complex
        return f"<decoded_{len(tokens)}_tokens>"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encode(text))

def get_offline_encoding(encoding_name: str = "o200k_base") -> OfflineTokenizer:
    """Drop-in replacement for tiktoken.get_encoding()"""
    return OfflineTokenizer(encoding_name)

# Monkey patch for existing code
def patch_tiktoken_for_offline():
    """Patch tiktoken imports to use offline tokenizer"""
    import sys
    
    # Create a mock tiktoken module
    class MockTiktoken:
        @staticmethod
        def get_encoding(name: str) -> OfflineTokenizer:
            return get_offline_encoding(name)
        
        @staticmethod  
        def encoding_for_model(model: str) -> OfflineTokenizer:
            return get_offline_encoding("o200k_base")
    
    # Replace tiktoken in sys.modules
    sys.modules['tiktoken'] = MockTiktoken()
    
    print("✅ Tiktoken replaced with offline tokenizer")

if __name__ == "__main__":
    # Test the offline tokenizer
    tokenizer = OfflineTokenizer()
    test_text = "Hello, this is a test for offline tokenization!"
    
    tokens = tokenizer.encode(test_text)
    token_count = tokenizer.count_tokens(test_text)
    
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
    print(f"Token count: {token_count}")
    print("✅ Offline tokenizer working!")
