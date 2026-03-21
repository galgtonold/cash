"""Tests for global cache functionality."""
from cash import cache, Cash


@cache
def global_add(a, b):
    '''Function using global cache decorator.'''
    return a + b


def test_global_cache():
    '''Test that global @cache decorator works.'''
    result1 = global_add(1, 2)
    assert result1 == 3
    
    result2 = global_add(1, 2)
    assert result2 == 3, 'Should return cached result'


def test_global_vs_local_cache():
    '''Test that global cache is independent from local Cash instances.'''
    # Global cache
    global_result = global_add(5, 10)
    assert global_result == 15
    
    # Local instance
    local_app = Cash(register_magic=False)
    
    @local_app.cache
    def local_add(a, b):
        return a + b
    
    local_result = local_add(5, 10)
    assert local_result == 15
    
    # Both work independently
    assert global_result == local_result
