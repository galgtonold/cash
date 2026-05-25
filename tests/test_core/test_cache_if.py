"""@cash.cache(cache_if=...) — conditional caching based on a result predicate."""
from __future__ import annotations

import asyncio
import warnings

import pytest

from cash import Cash


def test_cache_if_true_caches_normally(tmp_path):
    """When the predicate returns True, behavior matches today's @c.cache."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(cache_if=lambda r: True)
    def f(x):
        n["calls"] += 1
        return x * 2

    assert f(5) == 10
    assert f(5) == 10
    assert n["calls"] == 1


def test_cache_if_false_skips_store(tmp_path):
    """When the predicate returns False, the result is returned but not cached."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(cache_if=lambda r: r is not None)
    def lookup(key):
        n["calls"] += 1
        return None  # always returns None — should never be cached

    assert lookup("k") is None
    assert lookup("k") is None
    assert n["calls"] == 2, (
        f"None returns should not be cached when cache_if=lambda r: r is not None "
        f"(calls={n['calls']})"
    )


def test_cache_if_mixed_outcomes(tmp_path):
    """A predicate that gates on the result value: some inputs cache, others don't."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}
    # Sentinel: when the input is "skip", return a value the predicate rejects.
    @c.cache(cache_if=lambda r: r != "no")
    def f(x):
        n["calls"] += 1
        return "no" if x == "skip" else f"value-{x}"

    # First call: cacheable result
    assert f("a") == "value-a"
    assert f("a") == "value-a"  # hit
    assert n["calls"] == 1

    # Second arg: cacheable result
    assert f("b") == "value-b"
    assert f("b") == "value-b"  # hit
    assert n["calls"] == 2

    # Third arg: predicate rejects → no caching
    assert f("skip") == "no"
    assert f("skip") == "no"  # recomputes
    assert n["calls"] == 4

    # Back to "a": still hits the original cache entry
    assert f("a") == "value-a"
    assert n["calls"] == 4


def test_cache_if_predicate_exception_skips_store(tmp_path):
    """A buggy predicate that raises must not fail the user's call.
    The result is returned; nothing is cached; the user gets a one-shot
    CashCacheIneffectiveWarning so the silent disablement is visible."""
    from cash import CashCacheIneffectiveWarning

    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    def bad_predicate(result):
        raise RuntimeError("predicate is broken")

    @c.cache(cache_if=bad_predicate)
    def f(x):
        n["calls"] += 1
        return x * 3

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # Function call must succeed even though predicate raises.
        assert f(7) == 21
        # Predicate raises again → still no cache, but no second warning.
        assert f(7) == 21
    assert n["calls"] == 2, (
        f"buggy predicate must not allow caching; expected 2 computes, "
        f"got {n['calls']}"
    )
    # Exact category match — CashImpurityWarning subclasses
    # CashCacheIneffectiveWarning and would also be emitted because
    # `n["calls"] += 1` is a scope mutation. We're asserting on the
    # cache_if-raised warning specifically.
    ineffective = [w for w in captured if w.category is CashCacheIneffectiveWarning]
    assert len(ineffective) == 1, [str(w.message) for w in captured]
    msg = str(ineffective[0].message)
    assert "cache_if" in msg
    assert "RuntimeError" in msg
    assert "predicate is broken" in msg


async def test_cache_if_async_caches_normally(tmp_path):
    """On async functions, cache_if works the same way."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(cache_if=lambda r: r is not None)
    async def fetch(url):
        n["calls"] += 1
        await asyncio.sleep(0)
        return f"data-{url}"

    assert await fetch("u") == "data-u"
    assert await fetch("u") == "data-u"
    assert n["calls"] == 1


async def test_cache_if_async_false_skips_store(tmp_path):
    """On async, a False predicate also skips the store."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(cache_if=lambda r: r is not None)
    async def fetch_missing(key):
        n["calls"] += 1
        await asyncio.sleep(0)
        return None  # simulated miss

    assert await fetch_missing("k") is None
    assert await fetch_missing("k") is None
    assert n["calls"] == 2


def test_cache_if_function_exception_unaffected(tmp_path):
    """If the function itself raises, cache_if is not consulted and the
    exception propagates (existing behavior preserved)."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    predicate_called = {"yes": False}

    def predicate(result):
        predicate_called["yes"] = True
        return True

    @c.cache(cache_if=predicate)
    def f(x):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        f(1)

    assert predicate_called["yes"] is False, (
        "predicate must not be called when the function raises"
    )
