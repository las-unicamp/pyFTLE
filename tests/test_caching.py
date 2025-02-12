# ruff: noqa: F841

from collections import OrderedDict

from src.caching import cache_last_n_files


# Define some dummy functions to test the decorator
@cache_last_n_files(num_cached_files=2)
def function_a(file_path):
    return f"Result from function_a: {file_path}"


@cache_last_n_files(num_cached_files=2)
def function_b(file_path):
    return f"Result from function_b: {file_path}"


def test_cache_decorator_creates_separate_caches():
    """
    Test that the cache_last_n_files decorator creates separate caches for
    different functions.
    """
    # Call function_a twice with the same argument
    result_a1 = function_a("file1.txt")
    result_a2 = function_a("file1.txt")

    # Call function_b twice with the same argument
    result_b1 = function_b("file1.txt")
    result_b2 = function_b("file1.txt")

    # Assert that the results are cached correctly for each function
    assert id(result_a1) == id(
        result_a2
    ), "function_a should return cached result on second call"
    assert id(result_b1) == id(
        result_b2
    ), "function_b should return cached result on second call"

    # Check that the caches are independent
    assert (
        result_a1 != result_b1
    ), "function_a and function_b should have different results"

    # Inspect the caches (optional, for debugging)
    assert isinstance(
        function_a.cache, OrderedDict
    ), "function_a should have its own cache"
    assert isinstance(
        function_b.cache, OrderedDict
    ), "function_b should have its own cache"

    assert len(function_a.cache) == 1, "function_a's cache should have 1 entry"
    assert len(function_b.cache) == 1, "function_b's cache should have 1 entry"


def test_cache_size_limit():
    """
    Test that the cache size is limited to num_cached_files.
    """
    # Call function_a with 3 different arguments
    result_a1 = function_a("file1.txt")
    result_a2 = function_a("file2.txt")
    result_a3 = function_a("file3.txt")

    # Check the cache size
    assert (
        len(function_a.cache) == 2
    ), "function_a's cache should be limited to 2 entries"

    # The oldest entry ("file1.txt") should have been evicted
    cache_key_file1 = (("file1.txt",), frozenset())
    cache_key_file2 = (("file2.txt",), frozenset())
    cache_key_file3 = (("file3.txt",), frozenset())

    assert (
        cache_key_file1 not in function_a.cache
    ), "Oldest entry should be evicted from function_a's cache"
    assert (
        cache_key_file2 in function_a.cache
    ), "file2.txt should still be in function_a's cache"
    assert (
        cache_key_file3 in function_a.cache
    ), "file3.txt should still be in function_a's cache"


def test_cache_independence():
    """
    Test that the caches for function_a and function_b are independent.
    """
    # Call function_a and function_b with the same argument
    result_a = function_a("file1.txt")
    result_b = function_b("file1.txt")

    # Assert that the results are different
    assert (
        result_a != result_b
    ), "function_a and function_b should have independent caches"

    # Check that the caches are populated correctly
    cache_key_file1 = (("file1.txt",), frozenset())
    assert (
        cache_key_file1 in function_a.cache
    ), "file1.txt should be in function_a's cache"
    assert (
        cache_key_file1 in function_b.cache
    ), "file1.txt should be in function_b's cache"
