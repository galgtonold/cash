"""Tests for TTL (Time To Live) functionality."""
import time


def test_ttl_expiration(cash_instance):
    '''Test that cached values expire after TTL.'''
    side_effect = {'count': 0}
    
    @cash_instance.cache(ttl=0.1)
    def func_with_ttl(x):
        side_effect['count'] += 1
        return x * 2
    
    # First call - compute
    result1 = func_with_ttl(10)
    assert result1 == 20
    assert side_effect['count'] == 1
    
    # Second call - from cache
    result2 = func_with_ttl(10)
    assert result2 == 20
    assert side_effect['count'] == 1, 'Should use cached value'
    
    # Wait for expiration
    time.sleep(0.15)
    
    # Third call - recompute after expiration
    result3 = func_with_ttl(10)
    assert result3 == 20
    assert side_effect['count'] == 2, 'Should recompute after TTL expires'


def test_cleanup(cash_instance):
    '''Test cleanup of expired cache entries.'''
    
    @cash_instance.cache(ttl=0.1)
    def short_ttl(x):
        return x
    
    @cash_instance.cache(ttl=10)
    def long_ttl(x):
        return x
    
    # Cache both
    short_ttl(1)
    long_ttl(2)
    
    # Wait for short TTL to expire
    time.sleep(0.15)
    
    # Cleanup expired items
    deleted = cash_instance.cleanup()
    assert deleted == 1, 'Should delete one expired entry'
    
    # Verify long_ttl entry still exists
    assert len(cash_instance.backend._store) == 1
    
    # Force cleanup with max_age
    deleted = cash_instance.cleanup(max_age=0.1)
    assert deleted == 1, 'Should delete the remaining entry'
    assert len(cash_instance.backend._store) == 0, 'All entries should be cleaned'
