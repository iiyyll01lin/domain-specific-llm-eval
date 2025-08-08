#!/usr/bin/env python3
"""
GPU Configuration Module for Domain-Specific LLM Evaluation Pipeline

This module provides utilities to configure GPU acceleration for all models
in the evaluation pipeline including sentence transformers, spaCy models,
and other neural network components.
"""

import os
import torch
from typing import Dict, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GPUManager:
    """Manages GPU configuration and device selection for all models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize GPU manager with configuration.
        
        Args:
            config: Configuration dictionary with GPU settings
        """
        self.config = config or {}
        self.device = self._determine_device()
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        
        # Log GPU status
        self._log_gpu_status()
    
    def _determine_device(self) -> str:
        """Determine the best device to use."""
        # Check if GPU is available
        if torch.cuda.is_available():
            # Check if user has specified a device preference
            device_pref = self.config.get('device', 'auto')
            
            if device_pref == 'auto':
                return 'cuda:0'  # Use first GPU
            elif device_pref.startswith('cuda'):
                return device_pref
            else:
                return 'cpu'
        else:
            logger.warning("CUDA not available, using CPU")
            return 'cpu'
    
    def _log_gpu_status(self):
        """Log current GPU configuration status."""
        if self.cuda_available:
            logger.info(f"🎮 GPU acceleration enabled")
            logger.info(f"📱 Device: {self.device}")
            logger.info(f"🔢 Available GPUs: {self.device_count}")
            
            # Log GPU memory info
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"💾 GPU Memory: {gpu_memory / 1024**3:.1f} GB")
        else:
            logger.info("💻 Using CPU (GPU not available)")
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration for model loading."""
        return {
            'device': self.device,
            'cuda_available': self.cuda_available,
            'device_count': self.device_count,
            'torch_dtype': torch.float16 if self.cuda_available else torch.float32
        }
    
    def get_sentence_transformer_config(self) -> Dict[str, Any]:
        """Get configuration for sentence transformer models."""
        config = {
            'device': self.device,
            'trust_remote_code': True,
            'local_files_only': True,
            'cache_folder': './cache/sentence_transformers'
        }
        return config
    
    def get_huggingface_embeddings_config(self) -> Dict[str, Any]:
        """Get configuration for HuggingFace embeddings."""
        return {
            'model_kwargs': {
                'device': self.device,
                'trust_remote_code': True
            },
            'encode_kwargs': {
                'batch_size': 32 if self.cuda_available else 16,
                'show_progress_bar': True
            }
        }
    
    def get_transformers_config(self) -> Dict[str, Any]:
        """Get configuration for transformers models."""
        config = {
            'device_map': 'auto' if self.cuda_available else None,
            'torch_dtype': torch.float16 if self.cuda_available else torch.float32,
            'trust_remote_code': True,
            'local_files_only': True
        }
        
        if self.cuda_available:
            config['low_cpu_mem_usage'] = True
            
        return config
    
    def get_ragas_config(self) -> Dict[str, Any]:
        """Get configuration for RAGAS metrics."""
        return {
            'device': self.device,
            'batch_size': 8 if self.cuda_available else 2,
            'embeddings': {
                'device': self.device,
                'model_kwargs': {
                    'device': self.device
                }
            }
        }
    
    def optimize_for_gpu(self):
        """Apply GPU-specific optimizations."""
        if self.cuda_available:
            # Enable cuDNN benchmark for better performance
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable TensorFloat-32 for better performance on A100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("🚀 GPU optimizations applied")
    
    def get_spacy_config(self) -> Dict[str, Any]:
        """Get configuration for spaCy models."""
        # spaCy GPU support depends on the model and installation
        return {
            'gpu_id': 0 if self.cuda_available else -1,
            'prefer_gpu': self.cuda_available
        }

def configure_gpu_for_pipeline(config: Dict[str, Any]) -> GPUManager:
    """
    Configure GPU settings for the entire pipeline.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        GPUManager instance with configured settings
    """
    gpu_manager = GPUManager(config)
    
    # Apply optimizations
    gpu_manager.optimize_for_gpu()
    
    # Update config with GPU settings
    device_config = gpu_manager.get_device_config()
    
    # Update sentence transformer configs
    if 'evaluation' in config:
        if 'contextual_keywords' in config['evaluation']:
            config['evaluation']['contextual_keywords']['sentence_model_config'].update({
                'device': device_config['device']
            })
        
        if 'ragas_metrics' in config['evaluation']:
            config['evaluation']['ragas_metrics']['embeddings'].update({
                'device': device_config['device']
            })
    
    # Update testset generation configs
    if 'testset_generation' in config:
        if 'ragas_config' in config['testset_generation']:
            config['testset_generation']['ragas_config']['embeddings'].update({
                'device': device_config['device']
            })
    
    return gpu_manager

def check_gpu_compatibility() -> Dict[str, Any]:
    """
    Check GPU compatibility and available resources.
    
    Returns:
        Dictionary with GPU compatibility information
    """
    results = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'gpu_devices': []
    }
    
    if results['cuda_available']:
        for i in range(results['device_count']):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': props.name,
                'total_memory': props.total_memory,
                'memory_gb': props.total_memory / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}"
            }
            results['gpu_devices'].append(device_info)
    
    return results

def print_gpu_status():
    """Print detailed GPU status information."""
    print("\n🎮 GPU Configuration Status")
    print("=" * 50)
    
    status = check_gpu_compatibility()
    
    print(f"CUDA Available: {'✅ Yes' if status['cuda_available'] else '❌ No'}")
    print(f"PyTorch Version: {status['pytorch_version']}")
    
    if status['cuda_available']:
        print(f"CUDA Version: {status['cuda_version']}")
        print(f"GPU Device Count: {status['device_count']}")
        
        for device in status['gpu_devices']:
            print(f"\n📱 GPU {device['id']}: {device['name']}")
            print(f"   Memory: {device['memory_gb']:.1f} GB")
            print(f"   Compute Capability: {device['compute_capability']}")
    else:
        print("ℹ️ No GPU detected - using CPU")
    
    print("\n🔧 Model Device Settings:")
    gpu_manager = GPUManager()
    
    print(f"   Primary Device: {gpu_manager.device}")
    print(f"   Sentence Transformers: {gpu_manager.device}")
    print(f"   HuggingFace Embeddings: {gpu_manager.device}")
    print(f"   RAGAS Metrics: {gpu_manager.device}")

if __name__ == "__main__":
    print_gpu_status()
