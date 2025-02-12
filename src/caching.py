from collections import OrderedDict
from functools import wraps


def cache_last_n_files(num_cached_files=2):
    """
    A decorator to cache the results of a function for the last N unique argument
    combinations. Each decorated function will have its own independent cache.

    Args:
        num_cached_files (int): Number of unique argument combinations to keep in
        the cache.
    """

    def decorator(func):
        # Create a unique cache for this function
        cache = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on the function arguments
            cache_key = (args, frozenset(kwargs.items()))

            # If the result is cached, return it
            if cache_key in cache:
                # Move the accessed key to the end to mark it as recently used
                cache.move_to_end(cache_key)
                return cache[cache_key]

            # Otherwise, call the function and cache the result
            result = func(*args, **kwargs)
            cache[cache_key] = result

            # If the cache exceeds the allowed size, remove the oldest entry
            if len(cache) > num_cached_files:
                cache.popitem(last=False)

            return result

        # Attach the cache to the wrapper function for debugging or inspection
        wrapper.cache = cache

        return wrapper

    return decorator
