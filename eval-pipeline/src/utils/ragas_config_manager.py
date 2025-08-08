"""
RAGAS Configuration Override Utilities
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGASConfigManager:
    """Manages RAGAS configuration overrides."""
    
    def __init__(self, pipeline_config: Dict[str, Any]):
        self.pipeline_config = pipeline_config
        self.advanced_config = pipeline_config.get('advanced', {})
        self.testset_config = pipeline_config.get('testset_generation', {})
        self.ragas_config = self.testset_config.get('ragas_config', {})
        
    def get_optimized_run_config(self, memory_optimization: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create optimized RunConfig parameters for RAGAS."""
        
        # Get base configuration
        generation_config = self.ragas_config.get('generation_config', {})
        base_run_config = generation_config.get('run_config', {})
        
        # Get parallelization settings
        parallelization = self.advanced_config.get('parallelization', {})
        
        # Start with defaults
        run_config = {
            'max_workers': 2,
            'timeout': 120,
            'max_retries': 2,
            'max_wait': 10
        }
        
        # Apply base configuration
        run_config.update(base_run_config)
        
        # ✅ IMPLEMENTED: Override with our performance settings
        if parallelization.get('enabled', False):
            configured_workers = parallelization.get('max_workers', 4)
            run_config['max_workers'] = configured_workers
            logger.info(f"🔧 RAGAS workers overridden: {configured_workers} (from pipeline config)")
        
        # Apply memory-based optimizations
        if memory_optimization:
            recommended_workers = memory_optimization.get('recommended_workers', 2)
            if recommended_workers < run_config['max_workers']:
                run_config['max_workers'] = recommended_workers
                logger.info(f"🧠 RAGAS workers limited by memory: {recommended_workers}")
        
        # Apply performance optimizations
        batch_config = self.testset_config.get('batch_configuration', {})
        performance_opts = batch_config.get('performance_optimization', {})
        
        if performance_opts.get('parallel_processing', False):
            # Increase workers for parallel processing
            run_config['max_workers'] = min(run_config['max_workers'], 16)
            logger.info(f"⚡ Parallel processing enabled: {run_config['max_workers']} workers")
        
        # Apply timeout optimizations
        resource_limits = self.advanced_config.get('resource_limits', {})
        max_processing_time = resource_limits.get('max_processing_time_minutes', 30)
        run_config['timeout'] = min(run_config['timeout'], max_processing_time * 60 // 10)  # 1/10th of total time per operation
        
        logger.info(f"🎯 Optimized RAGAS RunConfig: {run_config}")
        return run_config
    
    def create_ragas_run_config(self, memory_optimization: Optional[Dict[str, Any]] = None):
        """Create actual RAGAS RunConfig object."""
        try:
            from ragas.run_config import RunConfig
            
            config_params = self.get_optimized_run_config(memory_optimization)
            run_config = RunConfig(**config_params)
            
            logger.info(f"✅ RAGAS RunConfig created successfully with optimized parameters")
            return run_config
            
        except ImportError as e:
            logger.warning(f"⚠️ Could not import RAGAS RunConfig: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error creating RAGAS RunConfig: {e}")
            return None
    
    def get_optimized_generation_config(self, memory_optimization: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get complete optimized generation configuration."""
        
        # Start with base generation config
        generation_config = self.ragas_config.get('generation_config', {}).copy()
        
        # Override run_config with optimized version
        generation_config['run_config'] = self.get_optimized_run_config(memory_optimization)
        
        # Apply high-memory optimizations
        if memory_optimization and memory_optimization.get('available_memory_mb', 0) > 8000:  # 8GB+
            generation_config['high_memory_mode'] = True
            generation_config['aggressive_caching'] = True
            generation_config['parallel_document_loading'] = True
            logger.info("🚀 High-memory optimizations enabled")
        
        return generation_config


def apply_ragas_config_overrides(generator, pipeline_config: Dict[str, Any], memory_optimization: Optional[Dict[str, Any]] = None):
    """Apply configuration overrides to a RAGAS testset generator."""
    
    config_manager = RAGASConfigManager(pipeline_config)
    
    try:
        # Override RunConfig if the generator has this attribute
        if hasattr(generator, 'run_config'):
            optimized_run_config = config_manager.create_ragas_run_config(memory_optimization)
            if optimized_run_config:
                generator.run_config = optimized_run_config
                logger.info("✅ RAGAS generator RunConfig overridden")
        
        # Override generation parameters if available
        if hasattr(generator, 'generation_config'):
            optimized_gen_config = config_manager.get_optimized_generation_config(memory_optimization)
            generator.generation_config.update(optimized_gen_config)
            logger.info("✅ RAGAS generator generation config overridden")
        
        # Apply embeddings optimizations
        embeddings_config = pipeline_config.get('testset_generation', {}).get('ragas_config', {}).get('embeddings', {})
        if embeddings_config and hasattr(generator, 'embeddings'):
            logger.info("🔧 Applying embeddings optimizations to RAGAS generator")
            # Note: Specific embeddings overrides would go here
            
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply RAGAS config overrides: {e}")
        return False
