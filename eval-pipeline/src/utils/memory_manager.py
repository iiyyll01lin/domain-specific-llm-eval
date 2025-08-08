"""
Memory Management Utilities for Performance Optimization
"""
import psutil
import gc
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory usage and optimization for the pipeline."""
    
    def __init__(self, max_memory_mb: int = 4096, aggressive_mode: bool = False):
        self.max_memory_mb = max_memory_mb
        self.aggressive_mode = aggressive_mode
        self.process = psutil.Process()
        self.baseline_memory = self.get_current_memory_mb()
        
        logger.info(f"🧠 Memory Manager initialized: {max_memory_mb}MB limit, aggressive={aggressive_mode}")
        logger.info(f"📊 Baseline memory usage: {self.baseline_memory:.1f}MB")
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_available_memory_mb(self) -> float:
        """Get available memory for allocation."""
        current = self.get_current_memory_mb()
        return max(0, self.max_memory_mb - current)
    
    def check_memory_limit(self, operation_name: str = "operation") -> bool:
        """Check if we're within memory limits."""
        current = self.get_current_memory_mb()
        if current > self.max_memory_mb:
            logger.warning(f"⚠️ Memory limit exceeded during {operation_name}: {current:.1f}MB > {self.max_memory_mb}MB")
            if self.aggressive_mode:
                self.force_cleanup()
            return False
        return True
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        logger.info("🧹 Forcing memory cleanup...")
        before = self.get_current_memory_mb()
        gc.collect()
        after = self.get_current_memory_mb()
        freed = before - after
        logger.info(f"✅ Memory cleanup: freed {freed:.1f}MB ({before:.1f}MB → {after:.1f}MB)")
    
    def optimize_for_operation(self, operation_name: str, estimated_memory_mb: float = 0) -> Dict[str, Any]:
        """Optimize memory settings for a specific operation."""
        available = self.get_available_memory_mb()
        current = self.get_current_memory_mb()
        
        # Calculate optimal settings
        if estimated_memory_mb > available:
            logger.warning(f"⚠️ {operation_name} needs {estimated_memory_mb:.1f}MB but only {available:.1f}MB available")
            if self.aggressive_mode:
                self.force_cleanup()
                available = self.get_available_memory_mb()
        
        # Return optimization recommendations
        optimization = {
            'current_memory_mb': current,
            'available_memory_mb': available,
            'estimated_need_mb': estimated_memory_mb,
            'can_proceed': available >= estimated_memory_mb,
            'recommended_batch_size': self._calculate_optimal_batch_size(available),
            'recommended_workers': self._calculate_optimal_workers(available)
        }
        
        logger.info(f"🎯 Memory optimization for {operation_name}: {optimization}")
        return optimization
    
    def _calculate_optimal_batch_size(self, available_memory_mb: float) -> int:
        """Calculate optimal batch size based on available memory."""
        # Estimate ~10MB per 1000 samples (conservative)
        if available_memory_mb < 100:
            return 100
        elif available_memory_mb < 1000:
            return int(available_memory_mb * 10)
        else:
            return min(5000, int(available_memory_mb * 5))
    
    def _calculate_optimal_workers(self, available_memory_mb: float) -> int:
        """Calculate optimal number of workers based on available memory."""
        # Estimate ~500MB per worker (conservative)
        max_workers_by_memory = max(1, int(available_memory_mb / 500))
        max_workers_by_cpu = psutil.cpu_count()
        return min(max_workers_by_memory, max_workers_by_cpu, 16)
    
    def log_memory_status(self, context: str = ""):
        """Log current memory status."""
        current = self.get_current_memory_mb()
        available = self.get_available_memory_mb()
        percentage = (current / self.max_memory_mb) * 100
        
        logger.info(f"📊 Memory Status {context}: {current:.1f}MB/{self.max_memory_mb}MB ({percentage:.1f}%) - {available:.1f}MB available")


def get_memory_manager(config: Dict[str, Any]) -> MemoryManager:
    """Factory function to create memory manager from config."""
    advanced_config = config.get('advanced', {})
    resource_limits = advanced_config.get('resource_limits', {})
    
    max_memory_mb = resource_limits.get('max_memory_mb', 4096)
    
    # Check for memory optimization settings
    batch_config = config.get('testset_generation', {}).get('batch_configuration', {})
    performance_opts = batch_config.get('performance_optimization', {})
    aggressive_mode = performance_opts.get('memory_aggressive', False)
    
    return MemoryManager(max_memory_mb=max_memory_mb, aggressive_mode=aggressive_mode)
