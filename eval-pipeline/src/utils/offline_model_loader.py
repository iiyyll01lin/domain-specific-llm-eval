#!/usr/bin/env python3
"""
Offline Model Loader for Sentence Transformers
Forces all sentence-transformers models to use cached versions only
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class OfflineModelLoader:
    """
    Utility class to ensure all sentence-transformers models are loaded offline
    """
    
    @staticmethod
    @contextmanager
    def offline_context():
        """
        Context manager that forces offline mode for HuggingFace models
        """
        # Store original environment
        original_env = {}
        offline_vars = {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_CACHE': os.path.expanduser('~/.cache/huggingface/transformers'),
            'HF_HOME': os.path.expanduser('~/.cache/huggingface'),
        }
        
        # Set offline environment variables
        for key, value in offline_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            logger.info("🔒 Forcing offline mode for HuggingFace models")
            yield
        finally:
            # Restore original environment
            for key in offline_vars:
                if original_env[key] is not None:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)
            logger.info("🔓 Restored online mode for HuggingFace models")
    
    @staticmethod
    def load_sentence_transformer_offline(model_name: str, **kwargs):
        """
        Load a sentence transformer model in offline mode
        """
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Clean model name (remove sentence-transformers/ prefix if present)
        clean_model_name = model_name.replace('sentence-transformers/', '')
        
        try:
            # First, try to load with cache_folder specified
            cache_folder = os.path.expanduser('~/.cache/huggingface/hub')
            
            # Set offline environment variables
            original_offline = os.environ.get('HF_HUB_OFFLINE')
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading sentence transformer '{clean_model_name}' from local cache")
                
                # Try different variations of the model name
                model_variations = [
                    clean_model_name,
                    f"sentence-transformers/{clean_model_name}",
                    model_name
                ]
                
                for model_variant in model_variations:
                    try:
                        model = SentenceTransformer(model_variant, cache_folder=cache_folder, **kwargs)
                        logger.info(f"✅ Successfully loaded '{model_variant}' from local cache")
                        return model
                    except Exception as variant_error:
                        logger.debug(f"Failed to load variant '{model_variant}': {variant_error}")
                        continue
                
                # If no variant worked, raise the last error
                raise Exception(f"Could not load any variant of {model_name}")
                
            finally:
                # Restore original offline setting
                if original_offline is not None:
                    os.environ['HF_HUB_OFFLINE'] = original_offline
                else:
                    os.environ.pop('HF_HUB_OFFLINE', None)
                
        except Exception as e:
            logger.error(f"❌ Failed to load '{model_name}' from local cache: {e}")
            
            # Try simple fallback loading without offline restriction
            logger.warning("🔄 Trying fallback loading with online access (limited)")
            try:
                from sentence_transformers import SentenceTransformer
                
                # Allow brief online access with timeout
                import socket
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(10)  # 10 second timeout
                
                try:
                    model = SentenceTransformer(clean_model_name, **kwargs)
                    logger.info(f"✅ Successfully loaded '{clean_model_name}' with fallback method")
                    return model
                finally:
                    socket.setdefaulttimeout(original_timeout)
                    
            except Exception as fallback_error:
                logger.error(f"❌ Fallback loading failed: {fallback_error}")
                raise e
    
    @staticmethod
    def check_model_availability(model_name: str) -> bool:
        """
        Check if a model is available in local cache
        """
        try:
            # Clean model name
            clean_model_name = model_name.replace('sentence-transformers/', '')
            
            # Check if model directory exists in cache
            cache_folder = os.path.expanduser('~/.cache/huggingface/hub')
            model_dir_name = f"models--sentence-transformers--{clean_model_name}"
            model_path = os.path.join(cache_folder, model_dir_name)
            
            if os.path.exists(model_path):
                logger.debug(f"✅ Found model directory for {model_name}: {model_path}")
                return True
            else:
                logger.debug(f"❌ No model directory found for {model_name}")
                return False
        except Exception as e:
            logger.debug(f"Error checking availability for {model_name}: {e}")
            return False
    
    @staticmethod
    def get_available_models() -> Dict[str, bool]:
        """
        Get a list of available models in local cache
        """
        common_models = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'paraphrase-multilingual-mpnet-base-v2',
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        ]
        
        availability = {}
        for model_name in common_models:
            availability[model_name] = OfflineModelLoader.check_model_availability(model_name)
        
        return availability

# Global function for easy import
def load_sentence_transformer_offline(model_name: str, **kwargs):
    """
    Global function to load sentence transformer models offline
    """
    return OfflineModelLoader.load_sentence_transformer_offline(model_name, **kwargs)

# Test function
def test_offline_loading():
    """
    Test offline loading functionality
    """
    print("🧪 Testing offline model loading...")
    
    # Check available models
    available = OfflineModelLoader.get_available_models()
    print("\n📋 Model availability:")
    for model, is_available in available.items():
        status = "✅" if is_available else "❌"
        print(f"  {status} {model}")
    
    # Test loading a model
    test_models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
    for model_name in test_models:
        try:
            model = load_sentence_transformer_offline(model_name)
            print(f"✅ Successfully loaded {model_name}")
            print(f"   Device: {model.device}")
            print(f"   Max sequence length: {model.max_seq_length}")
            break
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")

if __name__ == "__main__":
    test_offline_loading()
