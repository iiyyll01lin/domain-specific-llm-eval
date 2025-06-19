"""
Offline tiktoken utility for handling network connectivity issues
Provides fallback mechanisms when tiktoken can't access the internet
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_tiktoken_encoding_safe(encoding_name: str = "o200k_base"):
    """
    Safely get tiktoken encoding with fallback mechanisms
    
    Args:
        encoding_name: Name of the encoding to load
        
    Returns:
        tiktoken.Encoding or None if failed
    """
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load tiktoken encoding '{encoding_name}': {e}")
        logger.info("Attempting fallback encoding methods...")
        
        # Try alternative encodings
        fallback_encodings = ["cl100k_base", "p50k_base", "r50k_base"]
        if encoding_name in fallback_encodings:
            fallback_encodings.remove(encoding_name)
            
        for fallback in fallback_encodings:
            try:
                import tiktoken
                logger.info(f"Trying fallback encoding: {fallback}")
                return tiktoken.get_encoding(fallback)
            except Exception as fallback_error:
                logger.debug(f"Fallback encoding '{fallback}' also failed: {fallback_error}")
                continue
        
        logger.error("All tiktoken encoding methods failed. Running without tiktoken support.")
        return None

def count_tokens_safe(text: str, encoding_name: str = "o200k_base") -> int:
    """
    Safely count tokens with fallback to character-based approximation
    
    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding to use
        
    Returns:
        Token count (approximate if tiktoken fails)
    """
    encoding = get_tiktoken_encoding_safe(encoding_name)
    
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed with tiktoken: {e}")
    
    # Fallback: approximate token count using character count
    # Rough approximation: ~4 characters per token for English text
    approx_tokens = len(text) // 4
    logger.info(f"Using character-based token approximation: {approx_tokens} tokens")
    return approx_tokens

def setup_tiktoken_offline_mode():
    """
    Setup tiktoken for offline operation by setting environment variables
    """
    # Set tiktoken to use local cache only
    os.environ["TIKTOKEN_CACHE_ONLY"] = "1"
    
    # Disable tiktoken's automatic downloading
    os.environ["TIKTOKEN_DISABLE_DOWNLOAD"] = "1"
    
    logger.info("Configured tiktoken for offline operation")

def validate_tiktoken_cache() -> bool:
    """
    Validate that tiktoken cache is properly set up
    
    Returns:
        True if tiktoken can work offline, False otherwise
    """
    try:
        encoding = get_tiktoken_encoding_safe("o200k_base")
        if encoding is not None:
            # Test encoding some text
            test_text = "This is a test sentence for tiktoken validation."
            tokens = encoding.encode(test_text)
            logger.info(f"Tiktoken validation successful. Test text encoded to {len(tokens)} tokens.")
            return True
    except Exception as e:
        logger.warning(f"Tiktoken validation failed: {e}")
    
    return False

# Export the safe functions for use in other modules
__all__ = [
    'get_tiktoken_encoding_safe',
    'count_tokens_safe', 
    'setup_tiktoken_offline_mode',
    'validate_tiktoken_cache'
]
