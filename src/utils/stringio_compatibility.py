#!/usr/bin/env python3
"""
StringIO Compatibility Layer for RAGAS
======================================

This module provides compatibility fixes for StringIO validation errors
in RAGAS TestsetGenerator.
"""

import logging
from typing import Any, Dict, Union, List
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class StringIOCompatible(BaseModel):
    """Compatible StringIO model that handles various input formats."""
    text: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StringIOCompatible":
        """Create StringIO from dictionary, handling various formats."""
        if isinstance(data, dict):
            # Handle query/answer dict format
            if "query" in data and "answer" in data:
                text = f"Query: {data['query']}\nAnswer: {data['answer']}"
                return cls(text=text)
            # Handle text field directly
            elif "text" in data:
                return cls(text=str(data["text"]))
            # Handle content field
            elif "content" in data:
                return cls(text=str(data["content"]))
            # Convert entire dict to text
            else:
                text = str(data)
                return cls(text=text)
        elif isinstance(data, str):
            return cls(text=data)
        else:
            return cls(text=str(data))
    
    @classmethod
    def safe_create(cls, data: Any) -> "StringIOCompatible":
        """Safely create StringIO from any input, with fallbacks."""
        try:
            if data is None:
                return cls(text="")
            elif isinstance(data, cls):
                return data
            elif isinstance(data, dict):
                return cls.from_dict(data)
            elif isinstance(data, str):
                return cls(text=data)
            else:
                return cls(text=str(data))
        except Exception as e:
            logger.warning(f"Failed to create StringIO from {type(data)}: {e}")
            return cls(text="")

def patch_stringio_validation():
    """Patch RAGAS to use compatible StringIO validation."""
    try:
        # Import RAGAS modules that use StringIO
        from ragas.prompt.base import StringIO
        
        # Create a monkey patch for StringIO validation
        original_parse_obj = StringIO.parse_obj if hasattr(StringIO, 'parse_obj') else None
        original_validate = StringIO.validate if hasattr(StringIO, 'validate') else None
        
        def safe_parse_obj(cls, obj):
            """Safe parse_obj that handles validation errors."""
            try:
                if hasattr(StringIO, '_original_parse_obj'):
                    return StringIO._original_parse_obj(obj)
                elif original_parse_obj:
                    return original_parse_obj(obj)
                else:
                    # Fallback: create from compatible layer
                    compatible = StringIOCompatible.safe_create(obj)
                    return StringIO(text=compatible.text)
            except ValidationError as e:
                logger.warning(f"StringIO validation error, using fallback: {e}")
                compatible = StringIOCompatible.safe_create(obj)
                return StringIO(text=compatible.text)
            except Exception as e:
                logger.error(f"StringIO parse error: {e}")
                return StringIO(text=str(obj) if obj is not None else "")
        
        def safe_validate(cls, value):
            """Safe validation that handles various input types."""
            try:
                if hasattr(StringIO, '_original_validate'):
                    return StringIO._original_validate(value)
                elif original_validate:
                    return original_validate(value)
                else:
                    compatible = StringIOCompatible.safe_create(value)
                    return StringIO(text=compatible.text)
            except ValidationError as e:
                logger.warning(f"StringIO validation error, using fallback: {e}")
                compatible = StringIOCompatible.safe_create(value)
                return StringIO(text=compatible.text)
            except Exception as e:
                logger.error(f"StringIO validation error: {e}")
                return StringIO(text=str(value) if value is not None else "")
        
        # Apply patches
        if original_parse_obj and not hasattr(StringIO, '_original_parse_obj'):
            StringIO._original_parse_obj = original_parse_obj
            StringIO.parse_obj = classmethod(safe_parse_obj)
            
        if original_validate and not hasattr(StringIO, '_original_validate'):
            StringIO._original_validate = original_validate
            StringIO.validate = classmethod(safe_validate)
            
        logger.info("✅ StringIO validation patches applied")
        return True
        
    except ImportError as e:
        logger.warning(f"Could not import RAGAS StringIO for patching: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to patch StringIO validation: {e}")
        return False

# Auto-apply patches when module is imported
if __name__ == "__main__":
    patch_stringio_validation()
