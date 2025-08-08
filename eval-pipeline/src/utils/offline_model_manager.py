#!/usr/bin/env python3
"""
Comprehensive Offline Model Manager for Domain-Specific LLM Eval Pipeline

This module ensures ALL models (sentence transformers, spaCy, HuggingFace) work offline
without any internet connection. It handles model loading, caching, and fallback strategies.
"""

import os
import sys
import logging
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class OfflineModelManager:
    """
    Centralized manager for all offline model operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with pipeline configuration"""
        self.config = config
        self.cache_dir = Path(config.get('advanced', {}).get('caching', {}).get('cache_dir', './cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model cache paths
        self.sentence_transformer_cache = self.cache_dir / 'sentence_transformers'
        self.spacy_cache = self.cache_dir / 'spacy_models'
        self.huggingface_cache = self.cache_dir / 'huggingface'
        
        # Create cache directories
        self.sentence_transformer_cache.mkdir(parents=True, exist_ok=True)
        self.spacy_cache.mkdir(parents=True, exist_ok=True)
        self.huggingface_cache.mkdir(parents=True, exist_ok=True)
        
        # Set up offline environment
        self._setup_offline_environment()
        
        logger.info(f"🔧 Offline Model Manager initialized")
        logger.info(f"   Cache directory: {self.cache_dir}")
        logger.info(f"   Sentence Transformers cache: {self.sentence_transformer_cache}")
        logger.info(f"   spaCy cache: {self.spacy_cache}")
        logger.info(f"   HuggingFace cache: {self.huggingface_cache}")
    
    def _setup_offline_environment(self):
        """Set up environment variables for offline operation"""
        offline_env = {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1',
            'TRANSFORMERS_CACHE': str(self.huggingface_cache),
            'HF_HOME': str(self.huggingface_cache),
            'HF_HUB_CACHE': str(self.huggingface_cache),
            'SENTENCE_TRANSFORMERS_HOME': str(self.sentence_transformer_cache),
            'SPACY_WARNING_IGNORE': 'W008',
            'TOKENIZERS_PARALLELISM': 'false'
        }
        
        for key, value in offline_env.items():
            os.environ[key] = value
            
        logger.info("🔒 Offline environment configured")
    
    @contextmanager
    def offline_context(self):
        """Context manager for guaranteed offline operations"""
        # Store original environment
        original_env = {}
        offline_vars = {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1',
            'TRANSFORMERS_CACHE': str(self.huggingface_cache),
            'HF_HOME': str(self.huggingface_cache),
            'SENTENCE_TRANSFORMERS_HOME': str(self.sentence_transformer_cache),
            'TOKENIZERS_PARALLELISM': 'false'
        }
        
        # Set offline environment
        for key, value in offline_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            yield
        finally:
            # Restore original environment
            for key in offline_vars:
                if original_env[key] is not None:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)
    
    def load_sentence_transformer(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        Load sentence transformer model in offline mode with fallback strategies
        """
        logger.info(f"🔄 Loading sentence transformer: {model_name}")
        
        # Clean model name
        clean_model_name = model_name.replace('sentence-transformers/', '')
        
        with self.offline_context():
            try:
                from sentence_transformers import SentenceTransformer
                
                # Strategy 1: Load from custom cache
                try:
                    model = SentenceTransformer(
                        clean_model_name,
                        cache_folder=str(self.sentence_transformer_cache),
                        local_files_only=True,
                        **kwargs
                    )
                    logger.info(f"✅ Loaded {clean_model_name} from custom cache")
                    return model
                except Exception as e:
                    logger.debug(f"Custom cache failed: {e}")
                
                # Strategy 2: Load from default HuggingFace cache
                try:
                    model = SentenceTransformer(
                        clean_model_name,
                        local_files_only=True,
                        **kwargs
                    )
                    logger.info(f"✅ Loaded {clean_model_name} from HF cache")
                    return model
                except Exception as e:
                    logger.debug(f"HF cache failed: {e}")
                
                # Strategy 3: Try with sentence-transformers prefix
                try:
                    full_model_name = f"sentence-transformers/{clean_model_name}"
                    model = SentenceTransformer(
                        full_model_name,
                        local_files_only=True,
                        **kwargs
                    )
                    logger.info(f"✅ Loaded {full_model_name} from cache")
                    return model
                except Exception as e:
                    logger.debug(f"Full name failed: {e}")
                
                # Strategy 4: Check HuggingFace cache format (models--sentence-transformers--<model>)
                hf_cache_dir = self.sentence_transformer_cache / f"models--sentence-transformers--{clean_model_name}"
                if hf_cache_dir.exists():
                    try:
                        # Find the latest snapshot directory
                        snapshots_dir = hf_cache_dir / "snapshots"
                        if snapshots_dir.exists():
                            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                            if snapshot_dirs:
                                latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                                model = SentenceTransformer(str(latest_snapshot), **kwargs)
                                logger.info(f"✅ Loaded {clean_model_name} from HF cache snapshot")
                                return model
                    except Exception as e:
                        logger.debug(f"HF cache snapshot failed: {e}")
                
                # Strategy 5: Check local model directory
                local_model_path = self.sentence_transformer_cache / clean_model_name
                if local_model_path.exists():
                    try:
                        model = SentenceTransformer(str(local_model_path), **kwargs)
                        logger.info(f"✅ Loaded {clean_model_name} from local path")
                        return model
                    except Exception as e:
                        logger.debug(f"Local path failed: {e}")
                
                raise Exception(f"All offline strategies failed for {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name} offline: {e}")
                
                # Last resort: Use alternative model if available
                alternative_models = {
                    'all-MiniLM-L6-v2': ['all-mpnet-base-v2', 'paraphrase-MiniLM-L6-v2'],
                    'all-mpnet-base-v2': ['all-MiniLM-L6-v2', 'paraphrase-multilingual-mpnet-base-v2'],
                    'paraphrase-multilingual-mpnet-base-v2': ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']
                }
                
                if clean_model_name in alternative_models:
                    logger.warning(f"🔄 Trying alternative models for {clean_model_name}")
                    for alt_model in alternative_models[clean_model_name]:
                        try:
                            # IMPORTANT: Prevent infinite recursion by calling SentenceTransformer directly
                            from sentence_transformers import SentenceTransformer
                            model = SentenceTransformer(alt_model, local_files_only=True, **kwargs)
                            logger.info(f"✅ Using alternative model: {alt_model}")
                            return model
                        except Exception:
                            continue
                
                logger.error(f"❌ All alternatives failed for {model_name}")
                return None
    
    def load_spacy_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        Load spaCy model in offline mode
        """
        logger.info(f"🔄 Loading spaCy model: {model_name}")
        
        try:
            import spacy
            
            # Strategy 1: Load directly
            try:
                nlp = spacy.load(model_name, **kwargs)
                logger.info(f"✅ Loaded spaCy model: {model_name}")
                return nlp
            except Exception as e:
                logger.debug(f"Direct load failed: {e}")
            
            # Strategy 2: Try from local cache
            local_model_path = self.spacy_cache / model_name
            if local_model_path.exists():
                try:
                    nlp = spacy.load(str(local_model_path), **kwargs)
                    logger.info(f"✅ Loaded spaCy model from cache: {model_name}")
                    return nlp
                except Exception as e:
                    logger.debug(f"Cache load failed: {e}")
            
            # Strategy 3: Use alternative models
            alternative_models = {
                'en_core_web_trf': ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'],
                'zh_core_web_trf': ['zh_core_web_lg', 'zh_core_web_md', 'zh_core_web_sm'],
                'en_core_web_lg': ['en_core_web_md', 'en_core_web_sm'],
                'zh_core_web_lg': ['zh_core_web_md', 'zh_core_web_sm']
            }
            
            if model_name in alternative_models:
                logger.warning(f"🔄 Trying alternative spaCy models for {model_name}")
                for alt_model in alternative_models[model_name]:
                    try:
                        nlp = spacy.load(alt_model, **kwargs)
                        logger.info(f"✅ Using alternative spaCy model: {alt_model}")
                        return nlp
                    except Exception:
                        continue
            
            logger.error(f"❌ Could not load any spaCy model for {model_name}")
            return None
            
        except ImportError:
            logger.error("❌ spaCy not installed")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy model {model_name}: {e}")
            return None
    
    def check_model_availability(self) -> Dict[str, Dict[str, bool]]:
        """
        Check availability of all models used in the pipeline
        """
        logger.info("🔍 Checking model availability...")
        
        # Get model names from config
        sentence_models = []
        spacy_models = []
        
        # Extract from contextual keywords config
        contextual_config = self.config.get('evaluation', {}).get('contextual_keywords', {})
        if contextual_config.get('sentence_model'):
            sentence_models.append(contextual_config['sentence_model'])
        if contextual_config.get('spacy_model_en'):
            spacy_models.append(contextual_config['spacy_model_en'])
        if contextual_config.get('spacy_model_zh'):
            spacy_models.append(contextual_config['spacy_model_zh'])
        
        # Extract from RAGAS config
        ragas_config = self.config.get('evaluation', {}).get('ragas_metrics', {})
        if ragas_config.get('embeddings', {}).get('model_name'):
            sentence_models.append(ragas_config['embeddings']['model_name'])
        
        # Extract from testset generation config
        testset_config = self.config.get('testset_generation', {})
        if testset_config.get('embeddings_model'):
            sentence_models.append(testset_config['embeddings_model'])
        
        # Remove duplicates
        sentence_models = list(set(sentence_models))
        spacy_models = list(set(spacy_models))
        
        availability = {
            'sentence_transformers': {},
            'spacy': {},
            'status': 'checking'
        }
        
        # Check sentence transformers
        for model_name in sentence_models:
            if model_name:
                try:
                    model = self.load_sentence_transformer(model_name)
                    availability['sentence_transformers'][model_name] = model is not None
                except Exception:
                    availability['sentence_transformers'][model_name] = False
        
        # Check spaCy models
        for model_name in spacy_models:
            if model_name:
                try:
                    nlp = self.load_spacy_model(model_name)
                    availability['spacy'][model_name] = nlp is not None
                except Exception:
                    availability['spacy'][model_name] = False
        
        # Determine overall status
        sentence_transformers_available = all(availability['sentence_transformers'].values()) if availability['sentence_transformers'] else True
        spacy_available = all(availability['spacy'].values()) if availability['spacy'] else True
        all_available = sentence_transformers_available and spacy_available
        
        availability['status'] = 'available' if all_available else 'missing_models'
        
        return availability
    
    def get_model_fallback_config(self) -> Dict[str, Any]:
        """
        Get fallback configuration for missing models
        """
        return {
            'sentence_transformers': {
                'fallback_enabled': True,
                'fallback_models': {
                    'all-MiniLM-L6-v2': ['all-mpnet-base-v2', 'paraphrase-MiniLM-L6-v2'],
                    'all-mpnet-base-v2': ['all-MiniLM-L6-v2', 'paraphrase-multilingual-mpnet-base-v2']
                }
            },
            'spacy': {
                'fallback_enabled': True,
                'fallback_models': {
                    'en_core_web_trf': ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'],
                    'zh_core_web_trf': ['zh_core_web_lg', 'zh_core_web_md', 'zh_core_web_sm']
                }
            }
        }
    
    def validate_offline_setup(self) -> Dict[str, Any]:
        """
        Validate that offline setup is working correctly
        """
        logger.info("🧪 Validating offline setup...")
        
        validation = {
            'environment': {},
            'models': {},
            'overall_status': 'unknown'
        }
        
        # Check environment variables
        required_env = ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']
        for env_var in required_env:
            validation['environment'][env_var] = os.environ.get(env_var) == '1'
        
        # Check model availability
        validation['models'] = self.check_model_availability()
        
        # Overall status
        env_ok = all(validation['environment'].values())
        models_ok = validation['models']['status'] == 'available'
        
        validation['overall_status'] = 'ready' if (env_ok and models_ok) else 'needs_setup'
        
        return validation


# Global instance for easy access
_offline_manager = None

def get_offline_manager(config: Dict[str, Any]) -> OfflineModelManager:
    """Get or create global offline manager instance"""
    global _offline_manager
    if _offline_manager is None:
        _offline_manager = OfflineModelManager(config)
    return _offline_manager

def load_sentence_transformer_offline(model_name: str, config: Dict[str, Any], **kwargs):
    """Global function to load sentence transformer offline"""
    manager = get_offline_manager(config)
    return manager.load_sentence_transformer(model_name, **kwargs)

def load_spacy_offline(model_name: str, config: Dict[str, Any], **kwargs):
    """Global function to load spaCy model offline"""
    manager = get_offline_manager(config)
    return manager.load_spacy_model(model_name, **kwargs)

def validate_offline_setup(config: Dict[str, Any]) -> Dict[str, Any]:
    """Global function to validate offline setup"""
    manager = get_offline_manager(config)
    return manager.validate_offline_setup()


if __name__ == "__main__":
    # Test configuration
    test_config = {
        'advanced': {
            'caching': {
                'cache_dir': './cache'
            }
        },
        'evaluation': {
            'contextual_keywords': {
                'sentence_model': 'all-MiniLM-L6-v2',
                'spacy_model_en': 'en_core_web_sm',
                'spacy_model_zh': 'zh_core_web_sm'
            },
            'ragas_metrics': {
                'embeddings': {
                    'model_name': 'all-MiniLM-L6-v2'
                }
            }
        }
    }
    
    # Test offline manager
    manager = OfflineModelManager(test_config)
    validation = manager.validate_offline_setup()
    
    print("🧪 Offline Setup Validation:")
    print(f"Environment: {validation['environment']}")
    print(f"Models: {validation['models']}")
    print(f"Status: {validation['overall_status']}")
