"""Tests for TTL (Time To Live) functionality."""
import time

import pytest

from cash.backends import FileBackend
from cash.core import Cash


@pytest.fixture
def clock(monkeypatch):
    """A wall clock the test moves by hand.

    A ttl window measured with real sleeps is a race against the machine: the
    "still inside the ttl" call has an async write and the first write into a
    fresh directory ahead of it, and a Windows runner once took longer than a
    2s window over that (`assert 2 == 1`, having passed 40 runs before).
    Moving the clock separates "inside" from "past" by construction.
    """
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


def test_default_ttl_applies_without_explicit_ttl(temp_cache_dir, clock):
    """A decorator cache with no per-call ttl inherits the backend's default_ttl.

    Regression guard for the metadata-dataclass migration: the producer used
    to stamp ``ttl=None`` into the wire dict, which left the key *present* and
    so suppressed ``FileBackend(default_ttl=...)``. ``CacheMetadata.to_dict()``
    now omits ``None`` fields, so an unset ttl falls through to the backend
    default and the entry expires.
    """
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

    clock[0] += 2.1  # past default_ttl

    assert compute(10) == 20  # default_ttl expired — recompute
    assert side_effect['count'] == 2

    backend.clear()


def test_ttl_expiration(cash_instance, clock):
    '''Test that cached values expire after TTL.'''
    side_effect = {'count': 0}
    
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
    
    clock[0] += 2.1  # past the ttl

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


def test_a_file_tier_s_default_ttl_reaches_its_entries(temp_cache_dir, monkeypatch):
    """`default_ttl` on a declared file tier -- `[[tool.cash.tiers]]` or
    `CASH_TIER_<N>_DEFAULT_TTL` -- was accepted, shown by `cash info`, and
    dropped when the tier was built, so entries never expired. It is the
    only way to give every decorated function a default ttl from config."""
    monkeypatch.setenv("CASH_TIER_0_TYPE", "file")
    monkeypatch.setenv("CASH_TIER_0_DEFAULT_TTL", "7")
    monkeypatch.setenv("CASH_CACHE_DIR", temp_cache_dir)
    c = Cash(register_magic=False)
    tiers = getattr(c.backend, "backends", [c.backend])
    assert [getattr(t, "_default_ttl", None) for t in tiers if isinstance(t, FileBackend)] == [7]


def test_a_tier_default_ttl_expires_the_ram_copy_too(temp_cache_dir, monkeypatch, clock):
    """The ttl was stamped into the file tier's copy only, and the decorator
    checked its own ttl alone: in the process that wrote the entry, the RAM
    tier kept serving it after the disk copy had expired."""
    monkeypatch.setenv("CASH_TIER_0_TYPE", "memory")
    monkeypatch.setenv("CASH_TIER_1_TYPE", "file")
    monkeypatch.setenv("CASH_TIER_1_DEFAULT_TTL", "1")
    monkeypatch.setenv("CASH_CACHE_DIR", temp_cache_dir)
    c = Cash(register_magic=False)
    runs = []

    @c.cache(assume_safe=True)
    def f(x):
        runs.append(x)
        return x

    f(1)
    f(1)
    assert runs == [1], "control: inside the ttl it hits"
    clock[0] += 1.3
    assert f.explain(1).reason == "ttl_expired"
    f(1)
    assert runs == [1, 1], "the RAM tier served an entry past its tier's default_ttl"


def _tiered(monkeypatch, cache_dir, default_ttl):
    monkeypatch.setenv("CASH_TIER_0_TYPE", "memory")
    monkeypatch.setenv("CASH_TIER_1_TYPE", "file")
    monkeypatch.setenv("CASH_TIER_1_DEFAULT_TTL", str(default_ttl))
    monkeypatch.setenv("CASH_CACHE_DIR", cache_dir)
    return Cash(register_magic=False)


def test_lowering_a_tier_default_ttl_shortens_entries_already_written(temp_cache_dir, monkeypatch, clock):
    """Round 19: `default_ttl` 86400 -> 5 in the project config, and an entry
    written under the day was still served 8 seconds later (3 of 3). A
    decorator's ttl= lowered the same way took effect at once."""
    runs = []

    def make(c):
        @c.cache(assume_safe=True)
        def f(x):
            runs.append(x)
            time.sleep(0.15)        # past the persistence floor: the next run reads disk
            return x
        return f

    first = _tiered(monkeypatch, temp_cache_dir, 86400)
    make(first)(1)
    first.backend.backends[-1]._writes.wait_all()
    later = _tiered(monkeypatch, temp_cache_dir, 5)     # the next run, config lowered
    f = make(later)
    clock[0] += 3
    f(1)
    assert runs == [1], "control: inside the new ttl it hits"
    clock[0] += 5
    assert f.explain(1).reason == "ttl_expired"
    f(1)
    assert runs == [1, 1], "an entry written under the old default outlived the new one"


def test_an_entry_expired_under_the_tier_default_says_so(temp_cache_dir, monkeypatch, clock):
    """The file tier drops an expired entry on read, so the miss looked like an
    eviction -- "entry gone: evicted or cleared" -- in the reason and in
    explain(). It is recorded with the ttl it was written with now."""
    def body(x):
        time.sleep(0.15)            # past the persistence floor: the next run reads disk
        return x

    c = _tiered(monkeypatch, temp_cache_dir, 5)
    c.cache(assume_safe=True)(body)(1)
    c.backend.backends[-1]._writes.wait_all()
    fresh = _tiered(monkeypatch, temp_cache_dir, 5)    # a new process: no RAM copy
    f = fresh.cache(assume_safe=True)(body)
    clock[0] += 9
    explanation = f.explain(1)
    assert explanation.reason == "ttl_expired", explanation
    assert "ttl=5" in str(explanation.details)


def test_cash_info_shows_a_tier_s_default_ttl(monkeypatch, capsys):
    from types import SimpleNamespace

    from cash.__main__ import cmd_info
    monkeypatch.setenv("CASH_TIER_0_TYPE", "memory")
    monkeypatch.setenv("CASH_TIER_1_TYPE", "file")
    monkeypatch.setenv("CASH_TIER_1_DEFAULT_TTL", "5")
    cmd_info(SimpleNamespace())
    assert "file (default_ttl=5s)" in capsys.readouterr().out
