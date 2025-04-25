"""
Caching system for Solana RPC responses to reduce load and improve performance.
"""

import time
import logging
import asyncio
from typing import Any, Dict, Callable, Optional
from functools import wraps

# Setup logging
logger = logging.getLogger(__name__)

class RPCCache:
    """In-memory cache for RPC call results with time-based expiration."""
    
    def __init__(self, default_ttl: int = 300):
        """Initialize the cache with a default time-to-live.
        
        Args:
            default_ttl: Default cache TTL in seconds (default: 5 minutes)
        """
        self.cache = {}  # type: Dict[str, Dict[str, Any]]
        self.default_ttl = default_ttl
        self.locks = {}  # Per-key locks to prevent thundering herd
    
    async def get_or_fetch(self, cache_key: str, fetch_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get data from cache or fetch using the provided function.
        
        Args:
            cache_key: Unique key for identifying the cached data
            fetch_func: Async function to call if cache miss
            ttl: Time-to-live in seconds (defaults to self.default_ttl)
            
        Returns:
            Cached data or freshly fetched data
        """
        ttl = ttl or self.default_ttl
        now = time.time()
        
        # Check if we have a valid cached entry
        if cache_key in self.cache and (now - self.cache[cache_key]['timestamp'] < ttl):
            logger.debug(f"Cache hit for key: {cache_key}")
            return self.cache[cache_key]['data']
        
        # Acquire a lock for this key to prevent multiple fetches
        if cache_key not in self.locks:
            self.locks[cache_key] = asyncio.Lock()
        
        async with self.locks[cache_key]:
            # Check again in case another request fetched while we were waiting
            if cache_key in self.cache and (now - self.cache[cache_key]['timestamp'] < ttl):
                logger.debug(f"Cache hit after lock for key: {cache_key}")
                return self.cache[cache_key]['data']
            
            # Fetch fresh data
            logger.debug(f"Cache miss for key: {cache_key}, fetching fresh data")
            try:
                data = await fetch_func()
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                return data
            except Exception as e:
                logger.error(f"Error fetching data for key {cache_key}: {str(e)}")
                # If we have stale data, return it rather than failing
                if cache_key in self.cache:
                    logger.warning(f"Returning stale data for key: {cache_key}")
                    self.cache[cache_key]['stale'] = True
                    return self.cache[cache_key]['data']
                raise
    
    def invalidate(self, cache_key: str) -> None:
        """Invalidate a specific cache entry.
        
        Args:
            cache_key: The key to invalidate
        """
        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.debug(f"Invalidated cache key: {cache_key}")
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        logger.debug("Cleared entire cache")

    def cleanup(self) -> None:
        """Remove expired entries from the cache."""
        now = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if now - value['timestamp'] > self.default_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


def cached(ttl: int = None):
    """Decorator to cache results of async functions.
    
    Args:
        ttl: Optional cache TTL in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Create a cache instance specific to this function
        func_cache = RPCCache(default_ttl=ttl or 300)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            # Exclude 'self' from the key if it's a method
            arg_key = []
            for i, arg in enumerate(args):
                if i == 0 and hasattr(args[0], '__dict__'):
                    # Skip 'self' for methods
                    continue
                arg_key.append(str(arg))
            
            # Add kwargs to the cache key
            for k, v in sorted(kwargs.items()):
                arg_key.append(f"{k}={v}")
            
            cache_key = f"{func.__module__}.{func.__name__}:{':'.join(arg_key)}"
            
            # Get or fetch the result
            return await func_cache.get_or_fetch(
                cache_key,
                lambda: func(*args, **kwargs)
            )
        
        # Add a reference to the cache instance
        wrapper.cache = func_cache
        return wrapper
    
    return decorator

# Create a global cache instance
global_cache = RPCCache() 