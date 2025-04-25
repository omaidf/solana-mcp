"""
Cache service for Solana MCP.

This module provides a caching service with TTL support for improved performance.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from solana_mcp.services.base_service import BaseService, handle_errors
from solana_mcp.utils.errors import ResourceNotFoundError
from solana_mcp.utils.api_response import DataNotFoundError

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CacheEntry(Generic[T]):
    """Cache entry with value and expiration time."""
    
    def __init__(self, value: T, ttl: float):
        """
        Initialize a cache entry.
        
        Args:
            value: The value to cache
            ttl: Time to live in seconds
        """
        self.value = value
        self.expires_at = time.time() + ttl
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        return time.time() > self.expires_at


class LRUCache(Generic[K, V]):
    """A Least Recently Used cache with TTL support."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300,
        cleanup_interval: float = 60
    ):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of entries in the cache
            default_ttl: Default time to live in seconds
            cleanup_interval: Interval for cache cleanup in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.logger = logging.getLogger("LRUCache")
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def get(self, key: K) -> Optional[V]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                self.expirations += 1
                self.cache.pop(key)
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.update_access()
            self.hits += 1
            return entry.value
        
        self.misses += 1
        return None
    
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default_ttl if None)
        """
        # Ensure we don't exceed max size
        while len(self.cache) >= self.max_size:
            # Remove the least recently used item (front of OrderedDict)
            self.cache.popitem(last=False)
            self.evictions += 1
        
        # Add/update the entry
        self.cache[key] = CacheEntry(value, ttl or self.default_ttl)
        # Move to end (most recently used)
        self.cache.move_to_end(key)
    
    def delete(self, key: K) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        if key in self.cache:
            self.cache.pop(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        count = 0
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        
        for key in expired_keys:
            self.cache.pop(key)
            count += 1
        
        self.expirations += count
        return count
    
    async def start_cleanup_task(self) -> None:
        """Start the automatic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_task(self) -> None:
        """Stop the automatic cleanup task."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_loop(self) -> None:
        """Run the cleanup loop periodically."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                removed = self.cleanup()
                if removed > 0:
                    self.logger.debug(f"Removed {removed} expired cache entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "evictions": self.evictions,
            "expirations": self.expirations
        }


class CacheService(BaseService):
    """
    Cache service with TTL support.
    
    Features:
    - In-memory caching with configurable TTL
    - Automatic pruning of expired entries
    - Size limits to prevent memory issues
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,
        prune_interval: float = 60.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the cache service.
        
        Args:
            max_size: Maximum number of items to keep in cache
            default_ttl: Default time-to-live for cache entries in seconds
            prune_interval: Interval between cache pruning in seconds
            logger: Optional logger instance
        """
        super().__init__(logger=logger)
        self.cache: Dict[str, CacheEntry[Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.prune_interval = prune_interval
        self._last_prune = time.time()
        self.logger = logger or logging.getLogger(__name__)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    @handle_errors()
    async def get(self, key: str) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value
            
        Raises:
            DataNotFoundError: If key is not in cache or expired
        """
        self._maybe_prune()
        
        if key not in self.cache:
            raise DataNotFoundError(f"Key '{key}' not found in cache")
        
        entry = self.cache[key]
        if entry.is_expired():
            del self.cache[key]
            raise DataNotFoundError(f"Cache entry for '{key}' has expired")
        
        return entry.value

    @handle_errors()
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default_ttl if None)
        """
        self._maybe_prune()
        
        # Enforce size limit - remove oldest entries if needed
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._prune_oldest()
        
        self.cache[key] = CacheEntry(value, ttl or self.default_ttl)

    @handle_errors()
    async def delete(self, key: str) -> None:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key to delete
        """
        if key in self.cache:
            del self.cache[key]

    @handle_errors()
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        self.cache.clear()

    def _maybe_prune(self) -> None:
        """Prune expired cache entries if it's time to do so."""
        now = time.time()
        if now - self._last_prune > self.prune_interval:
            self._prune_expired()
            self._last_prune = now

    def _prune_expired(self) -> None:
        """Remove all expired entries from the cache."""
        expired_keys = [
            key for key, entry in self.cache.items() if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.logger.debug(f"Pruned {len(expired_keys)} expired cache entries")

    def _prune_oldest(self) -> None:
        """Remove the oldest entries to maintain the size limit."""
        if not self.cache:
            return
        
        # Find oldest entry
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].expires_at - time.time()
        )
        
        # Remove it
        del self.cache[oldest_key]
        self.logger.debug(f"Pruned oldest cache entry: {oldest_key}")

    @handle_errors()
    async def get_or_set(self, key: str, getter_func: callable, ttl: Optional[float] = None) -> Any:
        """
        Get a value from cache or set it using the getter function.
        
        Args:
            key: Cache key
            getter_func: Async function to call if cache miss
            ttl: Optional TTL override
            
        Returns:
            The cached or retrieved value
        """
        try:
            return await self.get(key)
        except DataNotFoundError:
            # Call the getter function
            value = await getter_func()
            # Cache the result
            await self.set(key, value, ttl)
            return value

    async def start(self) -> None:
        """Start the cache service."""
        await self.cache.start_cleanup_task()
        self.logger.info("Cache service started")
    
    async def stop(self) -> None:
        """Stop the cache service."""
        await self.cache.stop_cleanup_task()
        self.logger.info("Cache service stopped")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    async def get_multi(self, keys: list[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values (only for keys found in cache)
        """
        result = {}
        for key in keys:
            value = self.cache.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def set_multi(self, items: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """
        Set multiple values in the cache.
        
        Args:
            items: Dictionary mapping keys to values
            ttl: Time to live in seconds (uses default_ttl if None)
        """
        for key, value in items.items():
            self.cache.set(key, value, ttl)
    
    async def delete_multi(self, keys: list[str]) -> None:
        """
        Delete multiple values from the cache.
        
        Args:
            keys: List of cache keys
        """
        for key in keys:
            self.cache.delete(key)


# Singleton instance
_cache_service: Optional[CacheService] = None


def get_cache_service(
    default_ttl: float = 300.0, 
    max_size: int = 1000
) -> CacheService:
    """
    Get the singleton cache service instance.
    
    Args:
        default_ttl: Default TTL in seconds
        max_size: Maximum cache size
        
    Returns:
        The cache service instance
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService(default_ttl=default_ttl, max_size=max_size)
    return _cache_service 