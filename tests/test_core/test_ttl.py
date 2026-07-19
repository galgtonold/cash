"""Tests for TTL (Time To Live) functionality."""
import time

from cash.backends import FileBackend
from cash.core import Cash


def test_default_ttl_applies_without_explicit_ttl(temp_cache_dir):
    """A decorator cache with no per-call ttl inherits the backend's default_ttl.

    Regression guard for the metadata-dataclass migration: the producer used
    to stamp ``ttl=None`` into the wire dict, which left the key *present* and
    so suppressed ``FileBackend(default_ttl=...)``. ``CacheMetadata.to_dict()``
    now omits ``None`` fields, so an unset ttl falls through to the backend
    default and the entry expires.
    """
    # 2s, not 0.1s. The "cache hit" assertion below has to land inside the TTL
    # window, and between the two calls sits an async write plus the get() that
    # waits for it. 100ms is comfortable on an idle dev box and marginal on a
    # contended 2-vCPU CI runner, where this failed as `assert 2 == 1` — the
    # entry expired before it could be hit. The window only has to be long
    # enough that expiry cannot happen by accident; the sleep below still
    # exercises real expiry.
    backend = FileBackend(cache_dir=temp_cache_dir, default_ttl=2.0)
    cash = Cash(backend=backend, register_magic=False)
    side_effect = {'count': 0}

    @cash.cache
    def compute(x):
        side_effect['count'] += 1
        return x * 2

    assert compute(10) == 20
    assert side_effect['count'] == 1
    assert compute(10) == 20  # cache hit — no recompute
    assert side_effect['count'] == 1

    time.sleep(2.1)  # past default_ttl

    assert compute(10) == 20  # default_ttl expired — recompute
    assert side_effect['count'] == 2

    backend.clear()


def test_ttl_expiration(cash_instance):
    '''Test that cached values expire after TTL.'''
    side_effect = {'count': 0}
    
    # 2s for the same reason as test_default_ttl_applies_without_explicit_ttl:
    # the "from cache" assertion below must land inside the TTL window, and an
    # async write plus the get() that waits for it sit in between. A 100ms
    # budget is marginal on a contended CI runner.
    @cash_instance.cache(ttl=2.0)
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
    time.sleep(2.1)

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
