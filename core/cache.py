"""
Caching Layer - In-Memory Cache with TTL
Reduces API calls and speeds up repeated calculations.

Phase 7: Performance Optimization
"""
import time
import logging
from typing import Any, Optional, Callable, Dict
from functools import wraps
from collections import OrderedDict
import threading
import hashlib
import json

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Thread-safe in-memory cache with TTL (Time To Live).
    Uses LRU eviction when max size is reached.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Stats
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, key: Any) -> str:
        """Convert key to string"""
        if isinstance(key, str):
            return key
        return hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired"""
        if key not in self._expiry:
            return True
        return time.time() > self._expiry[key]
    
    def _evict_expired(self):
        """Remove expired items"""
        now = time.time()
        expired = [k for k, exp in self._expiry.items() if now > exp]
        for key in expired:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used items if over max size"""
        while len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest)
            self._expiry.pop(oldest, None)
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache.
        
        Returns:
            Cached value or None if not found/expired
        """
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key in self._cache:
                if not self._is_expired(str_key):
                    # Move to end (most recently used)
                    self._cache.move_to_end(str_key)
                    self._hits += 1
                    return self._cache[str_key]
                else:
                    # Remove expired
                    self._cache.pop(str_key, None)
                    self._expiry.pop(str_key, None)
            
            self._misses += 1
            return None
    
    def set(self, key: Any, value: Any, ttl: int = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
        """
        str_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Evict if needed
            self._evict_expired()
            if str_key not in self._cache:
                self._evict_lru()
            
            self._cache[str_key] = value
            self._expiry[str_key] = time.time() + ttl
            
            # Move to end
            self._cache.move_to_end(str_key)
    
    def delete(self, key: Any) -> bool:
        """Delete key from cache"""
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key in self._cache:
                self._cache.pop(str_key)
                self._expiry.pop(str_key, None)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': round(hit_rate, 1)
            }


# Global cache instances
_market_data_cache = TTLCache(max_size=500, default_ttl=60)  # 1 min for market data
_indicator_cache = TTLCache(max_size=1000, default_ttl=300)  # 5 min for indicators
_ml_cache = TTLCache(max_size=100, default_ttl=300)  # 5 min for ML predictions
_general_cache = TTLCache(max_size=500, default_ttl=600)  # 10 min general


def cached(
    cache_type: str = "general",
    ttl: int = None,
    key_prefix: str = ""
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        cache_type: Type of cache to use (market, indicator, ml, general)
        ttl: TTL in seconds (uses cache default if None)
        key_prefix: Prefix for cache key
        
    Usage:
        @cached(cache_type="indicator", ttl=300)
        def calculate_rsi(prices):
            ...
    """
    caches = {
        "market": _market_data_cache,
        "indicator": _indicator_cache,
        "ml": _ml_cache,
        "general": _general_cache
    }
    cache = caches.get(cache_type, _general_cache)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            
            # Add args to key (serialize safely)
            for arg in args:
                if hasattr(arg, 'shape'):  # numpy array or DataFrame
                    key_parts.append(f"shape:{arg.shape}")
                elif hasattr(arg, '__len__') and len(arg) > 10:
                    key_parts.append(f"len:{len(arg)}")
                else:
                    key_parts.append(str(arg)[:100])
            
            # Add kwargs to key
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{str(v)[:50]}")
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            if result is not None:
                cache.set(cache_key, result, ttl)
            
            return result
        
        # Add cache control methods
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_stats = lambda: cache.get_stats()
        
        return wrapper
    return decorator


def get_market_cache() -> TTLCache:
    """Get market data cache"""
    return _market_data_cache


def get_indicator_cache() -> TTLCache:
    """Get indicator cache"""
    return _indicator_cache


def get_ml_cache() -> TTLCache:
    """Get ML prediction cache"""
    return _ml_cache


def get_general_cache() -> TTLCache:
    """Get general cache"""
    return _general_cache


def get_all_cache_stats() -> Dict:
    """Get statistics for all caches"""
    return {
        'market': _market_data_cache.get_stats(),
        'indicator': _indicator_cache.get_stats(),
        'ml': _ml_cache.get_stats(),
        'general': _general_cache.get_stats()
    }


def clear_all_caches():
    """Clear all caches"""
    _market_data_cache.clear()
    _indicator_cache.clear()
    _ml_cache.clear()
    _general_cache.clear()
    logger.info("All caches cleared")


# Memoization decorator for expensive calculations
def memoize(maxsize: int = 128):
    """
    Simple memoization decorator (no TTL, just LRU).
    For pure functions with no side effects.
    
    Usage:
        @memoize(maxsize=100)
        def fibonacci(n):
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache = OrderedDict()
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create hashable key
            key = (args, tuple(sorted(kwargs.items())))
            
            with lock:
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
            
            result = func(*args, **kwargs)
            
            with lock:
                cache[key] = result
                cache.move_to_end(key)
                
                # Evict oldest if over size
                while len(cache) > maxsize:
                    cache.popitem(last=False)
            
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache), 'maxsize': maxsize}
        
        return wrapper
    return decorator
