"""Caching functionality for Solana MCP server."""

import asyncio
import functools
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

# Type variable for function return type
T = TypeVar('T')

@dataclass
class CacheConfig:
    """Configuration for the cache mechanism."""
    
    # Whether caching is enabled
    enabled: bool = True
    
    # Default TTL in seconds
    default_ttl: int = 30
    
    # TTLs for specific data types (in seconds)
    ttls: Dict[str, int] = None
    
    # Maximum memory cache size (item count)
    max_items: int = 10000
    
    def __post_init__(self):
        """Initialize default TTLs if not provided."""
        if self.ttls is None:
            self.ttls = {
                # Account data generally doesn't change that often
                "account": 60,
                # NFT metadata rarely changes
                "metadata": 3600,
                # Balance changes frequently
                "balance": 10,
                # Transaction data is immutable
                "transaction": 3600,
                # Token supply doesn't change often
                "token_supply": 300,
                # Network data changes frequently
                "network": 10,
                # Program data rarely changes
                "program": 600,
                # Block data is immutable
                "block": 3600
            }


class MemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, config: CacheConfig = None):
        """Initialize the cache.
        
        Args:
            config: Cache configuration. Defaults to default config.
        """
        self.config = config or CacheConfig()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
    
    def start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Periodically clean up expired cache entries."""
        while True:
            try:
                self._cleanup()
                # Run cleanup every 30 seconds
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log the error but don't crash the task
                await asyncio.sleep(60)  # Longer delay on error
    
    def _cleanup(self):
        """Remove expired cache entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self._cache.items():
            if entry["expires_at"] < current_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._cache.pop(key, None)
        
        # If we're over the max items limit, remove oldest entries
        if len(self._cache) > self.config.max_items:
            # Sort by expiration time (oldest first)
            sorted_keys = sorted(
                self._cache.keys(), 
                key=lambda k: self._cache[k]["expires_at"]
            )
            # Remove oldest entries until we're under the limit
            for key in sorted_keys[:len(self._cache) - self.config.max_items]:
                self._cache.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        if not self.config.enabled:
            return None
            
        entry = self._cache.get(key)
        if entry is None:
            return None
            
        # Check if expired
        if entry["expires_at"] < time.time():
            self._cache.pop(key, None)
            return None
            
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: int = None, category: str = None):
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time to live in seconds. Defaults to category or default TTL.
            category: Data category for determining TTL. Defaults to None.
        """
        if not self.config.enabled:
            return
            
        # Determine TTL
        if ttl is None:
            if category and category in self.config.ttls:
                ttl = self.config.ttls[category]
            else:
                ttl = self.config.default_ttl
                
        # Set the cache entry
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "category": category
        }
    
    def invalidate(self, key: str):
        """Invalidate a specific cache entry.
        
        Args:
            key: The cache key to invalidate
        """
        self._cache.pop(key, None)
    
    def invalidate_by_prefix(self, prefix: str):
        """Invalidate all cache entries with keys starting with the prefix.
        
        Args:
            prefix: The key prefix
        """
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            self._cache.pop(key, None)
    
    def invalidate_by_category(self, category: str):
        """Invalidate all cache entries of a specific category.
        
        Args:
            category: The category to invalidate
        """
        keys_to_remove = [
            k for k, v in self._cache.items() 
            if v.get("category") == category
        ]
        for key in keys_to_remove:
            self._cache.pop(key, None)
    
    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()
    
    def stop(self):
        """Stop the cache cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None


# Global memory cache instance
_memory_cache = MemoryCache()


def get_cache() -> MemoryCache:
    """Get the global memory cache instance.
    
    Returns:
        The memory cache instance
    """
    return _memory_cache


def cache(ttl: int = None, category: str = None, key_builder: Callable = None):
    """Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds. Defaults to category default.
        category: Data category for TTL. Defaults to None.
        key_builder: Function to build cache key. Defaults to args-based key.
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not _memory_cache.config.enabled:
                return await func(*args, **kwargs)
                
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key is function name + args + kwargs
                key_parts = [func.__module__, func.__name__]
                
                # Add args (excluding self/cls for methods)
                if args:
                    if hasattr(args[0], '__dict__'):
                        # Skip self/cls
                        arg_part = json.dumps([str(a) for a in args[1:]])
                    else:
                        arg_part = json.dumps([str(a) for a in args])
                    key_parts.append(arg_part)
                
                # Add kwargs
                if kwargs:
                    kwargs_part = json.dumps(
                        {k: str(v) for k, v in sorted(kwargs.items())}
                    )
                    key_parts.append(kwargs_part)
                
                cache_key = ":".join(key_parts)
            
            # Check cache
            cached_value = _memory_cache.get(cache_key)
            if cached_value is not None:
                return cached_value
                
            # Call function and cache result
            result = await func(*args, **kwargs)
            _memory_cache.set(cache_key, result, ttl=ttl, category=category)
            return result
            
        return wrapper
    return decorator


def init_cache(config: CacheConfig = None):
    """Initialize the cache with provided configuration.
    
    Args:
        config: Cache configuration. Defaults to default config.
    """
    global _memory_cache
    _memory_cache = MemoryCache(config or CacheConfig())
    _memory_cache.start_cleanup_task()


def shutdown_cache():
    """Shutdown the cache and cleanup tasks."""
    _memory_cache.stop() 