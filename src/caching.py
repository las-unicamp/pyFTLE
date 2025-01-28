from collections import OrderedDict
from functools import wraps


def cache_last_n_files(num_cached_files=2):
    """
    A decorator to cache the results of a function for the last N file paths.
    Args:
        num_cached_files (int): Number of files to keep in the cache.
    """

    def decorator(func):
        cache = OrderedDict()  # OrderedDict to maintain insertion order

        @wraps(func)
        def wrapper(file_path, *args, **kwargs):
            # If the result is cached, return it
            if file_path in cache:
                return cache[file_path]

            # Otherwise, call the function and cache the result
            result = func(file_path, *args, **kwargs)
            cache[file_path] = result

            # If the cache exceeds the allowed size, remove the oldest entry
            if len(cache) > num_cached_files:
                cache.popitem(last=False)

            return result

        return wrapper

    return decorator
