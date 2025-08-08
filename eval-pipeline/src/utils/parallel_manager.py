"""
Parallel Processing Manager for Performance Optimization
"""
import logging
import threading
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Callable, Optional, Tuple
from functools import wraps
import time

logger = logging.getLogger(__name__)

class ParallelProcessingManager:
    """Manages parallel processing with configurable workers and optimizations."""
    
    def __init__(self, config: Dict[str, Any], memory_manager=None):
        self.config = config
        self.memory_manager = memory_manager
        
        # Get parallelization configuration
        advanced_config = config.get('advanced', {})
        parallelization = advanced_config.get('parallelization', {})
        
        self.enabled = parallelization.get('enabled', False)
        self.max_workers = parallelization.get('max_workers', 4)
        
        # Apply memory constraints
        if memory_manager:
            memory_optimization = memory_manager.optimize_for_operation("parallel_processing", 0)
            recommended_workers = memory_optimization.get('recommended_workers', self.max_workers)
            if recommended_workers < self.max_workers:
                logger.info(f"🧠 Reducing workers due to memory constraints: {self.max_workers} → {recommended_workers}")
                self.max_workers = recommended_workers
        
        # Thread pools for different types of work
        self.thread_pools = {}
        
        logger.info(f"⚡ Parallel Processing Manager initialized: {self.max_workers} workers, enabled={self.enabled}")
    
    def get_thread_pool(self, pool_name: str, max_workers: Optional[int] = None) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create a thread pool for specific operations."""
        if pool_name not in self.thread_pools:
            workers = min(max_workers or self.max_workers, self.max_workers)
            self.thread_pools[pool_name] = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"pipeline-{pool_name}"
            )
            logger.info(f"🔧 Created thread pool '{pool_name}': {workers} workers")
        
        return self.thread_pools[pool_name]
    
    def parallel_map(self, func: Callable, items: List[Any], pool_name: str = "default", 
                    chunk_size: Optional[int] = None, timeout: Optional[float] = None) -> List[Any]:
        """Execute function in parallel over list of items."""
        if not self.enabled or len(items) <= 1:
            # Sequential processing
            logger.debug(f"🔄 Sequential processing: {len(items)} items")
            return [func(item) for item in items]
        
        # Parallel processing
        pool = self.get_thread_pool(pool_name)
        results = []
        
        logger.info(f"⚡ Parallel processing: {len(items)} items with {pool._max_workers} workers")
        
        try:
            # Submit all tasks
            futures = [pool.submit(func, item) for item in items]
            
            # Collect results with timeout
            for i, future in enumerate(concurrent.futures.as_completed(futures, timeout=timeout)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Memory check during processing
                    if self.memory_manager and i % 10 == 0:  # Check every 10 items
                        self.memory_manager.check_memory_limit(f"parallel_map_{pool_name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Parallel task failed: {e}")
                    results.append(None)
            
            logger.info(f"✅ Parallel processing completed: {len(results)} results")
            return results
            
        except concurrent.futures.TimeoutError:
            logger.error(f"❌ Parallel processing timeout after {timeout}s")
            return [None] * len(items)
        except Exception as e:
            logger.error(f"❌ Parallel processing failed: {e}")
            return [None] * len(items)
    
    def parallel_batch_process(self, func: Callable, items: List[Any], batch_size: int,
                              pool_name: str = "batch", timeout: Optional[float] = None) -> List[Any]:
        """Process items in parallel batches."""
        if not self.enabled:
            logger.debug(f"🔄 Sequential batch processing: {len(items)} items")
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = func(batch)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
            return results
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        logger.info(f"⚡ Parallel batch processing: {len(batches)} batches of ~{batch_size} items")
        
        # Process batches in parallel
        batch_results = self.parallel_map(func, batches, pool_name, timeout=timeout)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if batch_result:
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
        
        return results
    
    def async_parallel_process(self, async_func: Callable, items: List[Any],
                              max_concurrent: Optional[int] = None) -> List[Any]:
        """Process items using async/await parallelism."""
        if not self.enabled:
            return asyncio.run(self._sequential_async(async_func, items))
        
        max_concurrent = min(max_concurrent or self.max_workers, self.max_workers)
        return asyncio.run(self._parallel_async(async_func, items, max_concurrent))
    
    async def _sequential_async(self, async_func: Callable, items: List[Any]) -> List[Any]:
        """Sequential async processing."""
        results = []
        for item in items:
            result = await async_func(item)
            results.append(result)
        return results
    
    async def _parallel_async(self, async_func: Callable, items: List[Any], max_concurrent: int) -> List[Any]:
        """Parallel async processing with semaphore."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_func(item):
            async with semaphore:
                return await async_func(item)
        
        logger.info(f"⚡ Async parallel processing: {len(items)} items with {max_concurrent} concurrent tasks")
        
        tasks = [limited_func(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        clean_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Async task failed: {result}")
                clean_results.append(None)
            else:
                clean_results.append(result)
        
        return clean_results
    
    def shutdown(self):
        """Shutdown all thread pools."""
        for pool_name, pool in self.thread_pools.items():
            try:
                pool.shutdown(wait=True, cancel_futures=True)
                logger.info(f"🔚 Shutdown thread pool: {pool_name}")
            except Exception as e:
                logger.warning(f"⚠️ Error shutting down pool {pool_name}: {e}")
        
        self.thread_pools.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        stats = {
            'enabled': self.enabled,
            'max_workers': self.max_workers,
            'active_pools': len(self.thread_pools),
            'pools': {}
        }
        
        for pool_name, pool in self.thread_pools.items():
            try:
                stats['pools'][pool_name] = {
                    'max_workers': pool._max_workers,
                    'threads': len(pool._threads) if hasattr(pool, '_threads') else 0
                }
            except:
                stats['pools'][pool_name] = {'status': 'unknown'}
        
        return stats


def parallel_decorator(pool_name: str = "default", enabled_config_path: str = "advanced.parallelization.enabled"):
    """Decorator to make functions run in parallel when enabled."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get config from first argument (usually self with config)
            if args and hasattr(args[0], 'config'):
                config = args[0].config
                
                # Check if parallelization is enabled
                keys = enabled_config_path.split('.')
                enabled = config
                for key in keys:
                    enabled = enabled.get(key, False)
                    if not isinstance(enabled, (dict, bool)):
                        enabled = False
                        break
                
                if enabled and hasattr(args[0], '_parallel_manager'):
                    # Use parallel processing
                    parallel_manager = args[0]._parallel_manager
                    pool = parallel_manager.get_thread_pool(pool_name)
                    
                    # Submit to thread pool
                    future = pool.submit(func, *args, **kwargs)
                    return future.result()
            
            # Fall back to normal execution
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_parallel_manager(config: Dict[str, Any], memory_manager=None) -> ParallelProcessingManager:
    """Factory function to create parallel processing manager."""
    return ParallelProcessingManager(config, memory_manager)
