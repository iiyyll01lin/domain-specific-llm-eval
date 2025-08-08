"""
Intelligent Caching System for Performance Optimization
"""
import pickle
import hashlib
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import OrderedDict
import time

logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU cache with size limits."""
    
    def __init__(self, max_size_mb: float):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache = OrderedDict()
        self.size_bytes = 0
        self.lock = threading.RLock()
        
    def _get_size(self, obj) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value['data']
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache, return True if successful."""
        with self.lock:
            # Calculate size
            item_size = self._get_size(value)
            
            # Check if item is too large
            if item_size > self.max_size_bytes:
                logger.warning(f"⚠️ Cache item too large: {item_size} bytes > {self.max_size_bytes} bytes")
                return False
            
            # Remove existing item if present
            if key in self.cache:
                old_size = self.cache[key]['size']
                self.size_bytes -= old_size
                del self.cache[key]
            
            # Make room for new item
            while self.size_bytes + item_size > self.max_size_bytes and self.cache:
                # Remove least recently used item
                lru_key, lru_value = self.cache.popitem(last=False)
                self.size_bytes -= lru_value['size']
                logger.debug(f"🗑️ Evicted cache item: {lru_key}")
            
            # Add new item
            self.cache[key] = {
                'data': value,
                'size': item_size,
                'timestamp': time.time()
            }
            self.size_bytes += item_size
            
            logger.debug(f"💾 Cached item: {key} ({item_size} bytes) - total: {self.size_bytes}/{self.max_size_bytes}")
            return True
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.size_bytes = 0
            logger.info("🧹 Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'items': len(self.cache),
                'size_mb': self.size_bytes / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'utilization': self.size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }


class IntelligentCacheManager:
    """Manages multiple cache types with intelligent strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.caches = {}
        
        # Get cache configuration
        advanced_config = config.get('advanced', {})
        caching_config = advanced_config.get('caching', {})
        
        total_cache_mb = caching_config.get('cache_size_mb', 1000)
        
        # Allocate cache sizes based on usage patterns
        self.cache_allocations = {
            'embeddings': total_cache_mb * 0.4,    # 40% for embeddings
            'documents': total_cache_mb * 0.3,     # 30% for documents
            'kg_cache': total_cache_mb * 0.15,     # 15% for knowledge graphs
            'testsets': total_cache_mb * 0.1,      # 10% for testsets
            'misc': total_cache_mb * 0.05          # 5% for miscellaneous
        }
        
        # Initialize caches
        for cache_type, size_mb in self.cache_allocations.items():
            self.caches[cache_type] = LRUCache(size_mb)
            logger.info(f"🗄️ Initialized {cache_type} cache: {size_mb:.1f}MB")
        
        # Cache directories
        self.cache_base_dir = Path(caching_config.get('cache_dir', './cache'))
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Intelligent Cache Manager initialized: {total_cache_mb}MB total")
    
    def _get_cache_key(self, category: str, identifier: str, params: Dict[str, Any] = None) -> str:
        """Generate unique cache key."""
        key_data = f"{category}:{identifier}"
        if params:
            # Sort params for consistent hashing
            sorted_params = sorted(params.items())
            params_str = str(sorted_params)
            key_data += f":{params_str}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache_embeddings(self, text: str, model_name: str, embedding: Any) -> bool:
        """Cache embeddings with text and model as key."""
        key = self._get_cache_key("embeddings", text, {"model": model_name})
        return self.caches['embeddings'].put(key, embedding)
    
    def get_cached_embeddings(self, text: str, model_name: str) -> Optional[Any]:
        """Get cached embeddings."""
        key = self._get_cache_key("embeddings", text, {"model": model_name})
        return self.caches['embeddings'].get(key)
    
    def cache_document(self, doc_id: str, document: Any) -> bool:
        """Cache processed documents."""
        key = self._get_cache_key("documents", doc_id)
        return self.caches['documents'].put(key, document)
    
    def get_cached_document(self, doc_id: str) -> Optional[Any]:
        """Get cached document."""
        key = self._get_cache_key("documents", doc_id)
        return self.caches['documents'].get(key)
    
    def cache_knowledge_graph(self, kg_id: str, kg_data: Any) -> bool:
        """Cache knowledge graph data."""
        key = self._get_cache_key("kg_cache", kg_id)
        return self.caches['kg_cache'].put(key, kg_data)
    
    def get_cached_knowledge_graph(self, kg_id: str) -> Optional[Any]:
        """Get cached knowledge graph."""
        key = self._get_cache_key("kg_cache", kg_id)
        return self.caches['kg_cache'].get(key)
    
    def cache_testset(self, testset_id: str, testset_data: Any) -> bool:
        """Cache testset data."""
        key = self._get_cache_key("testsets", testset_id)
        return self.caches['testsets'].put(key, testset_data)
    
    def get_cached_testset(self, testset_id: str) -> Optional[Any]:
        """Get cached testset."""
        key = self._get_cache_key("testsets", testset_id)
        return self.caches['testsets'].get(key)
    
    def cache_misc(self, cache_id: str, data: Any, params: Dict[str, Any] = None) -> bool:
        """Cache miscellaneous data."""
        key = self._get_cache_key("misc", cache_id, params)
        return self.caches['misc'].put(key, data)
    
    def get_cached_misc(self, cache_id: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached miscellaneous data."""
        key = self._get_cache_key("misc", cache_id, params)
        return self.caches['misc'].get(key)
    
    def persistent_cache_embeddings(self, text: str, model_name: str, embedding: Any):
        """Cache embeddings to disk for persistence."""
        cache_dir = self.cache_base_dir / "embeddings" / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        filename = hashlib.md5(text.encode()).hexdigest() + ".pkl"
        cache_file = cache_dir / filename
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            logger.debug(f"💾 Persisted embedding: {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist embedding: {e}")
    
    def get_persistent_cached_embeddings(self, text: str, model_name: str) -> Optional[Any]:
        """Get embeddings from persistent disk cache."""
        cache_dir = self.cache_base_dir / "embeddings" / model_name
        filename = hashlib.md5(text.encode()).hexdigest() + ".pkl"
        cache_file = cache_dir / filename
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                logger.debug(f"🎯 Loaded persistent embedding: {cache_file}")
                return embedding
            except Exception as e:
                logger.warning(f"⚠️ Failed to load persistent embedding: {e}")
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {}
        total_items = 0
        total_size_mb = 0
        
        for cache_type, cache in self.caches.items():
            cache_stats = cache.stats()
            stats[cache_type] = cache_stats
            total_items += cache_stats['items']
            total_size_mb += cache_stats['size_mb']
        
        stats['total'] = {
            'items': total_items,
            'size_mb': total_size_mb,
            'max_size_mb': sum(self.cache_allocations.values()),
            'utilization': total_size_mb / sum(self.cache_allocations.values()) if sum(self.cache_allocations.values()) > 0 else 0
        }
        
        return stats
    
    def log_cache_stats(self):
        """Log current cache statistics."""
        stats = self.get_cache_stats()
        total_stats = stats['total']
        
        logger.info(f"📊 Cache Statistics:")
        logger.info(f"   Total: {total_stats['items']} items, {total_stats['size_mb']:.1f}MB/{total_stats['max_size_mb']:.1f}MB ({total_stats['utilization']:.1%})")
        
        for cache_type, cache_stats in stats.items():
            if cache_type != 'total':
                logger.info(f"   {cache_type}: {cache_stats['items']} items, {cache_stats['size_mb']:.1f}MB ({cache_stats['utilization']:.1%})")
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("🧹 All caches cleared")


def get_cache_manager(config: Dict[str, Any]) -> IntelligentCacheManager:
    """Factory function to create cache manager from config."""
    return IntelligentCacheManager(config)
