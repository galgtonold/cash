"""Tests that previously-silent failures now surface as CashWarnings
and are also discoverable via ``f.cache_info()['warnings']``.

Prior behavior used ``logger.debug`` or ``logger.warning`` which got
drowned out for anyone not actively listening. These tests pin the
new behavior: visible to ``warnings``-aware users AND inspectable
after the fact via the per-function rolling log.
"""
from __future__ import annotations

import asyncio
import warnings

import pytest

from cash import Cash, CashCacheIneffectiveWarning


def test_cache_if_raise_appears_in_cache_info(tmp_path):
    """The CashWarning that fires when cache_if raises is also stored
    so a user who didn't enable warnings can still see it."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def bad(_result):
        raise RuntimeError("nope")

    @c.cache(cache_if=bad)
    def f(x):
        return x * 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # user is ignoring warnings entirely
        f(5)

    info = f.cache_info()
    assert "warnings" in info
    assert len(info["warnings"]) >= 1
    rec = info["warnings"][0]
    assert rec["category"] == "CashCacheIneffectiveWarning"
    assert "cache_if" in rec["message"]
    assert "timestamp" in rec
    # The code is recorded as its own field, not only inside the text: this log
    # is where people look once the stderr line has scrolled away, and a reader
    # of it should be able to branch on the code the way a warning handler
    # branches on ``w.message.code``.
    assert rec["code"] == "CACHE-IF-RAISED"
    assert rec["message"].startswith("[CACHE-IF-RAISED] ")


def test_cache_if_raise_async_appears_in_cache_info(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def bad(_result):
        raise RuntimeError("async-nope")

    @c.cache(cache_if=bad)
    async def f(x):
        await asyncio.sleep(0)
        return x * 2

    async def go():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await f(5)

    asyncio.run(go())

    info = f.cache_info()
    msgs = [w["message"] for w in info["warnings"]]
    assert any("cache_if" in m and "async-nope" in m for m in msgs), msgs


def test_cache_info_warnings_deduped_per_func(tmp_path):
    """_warn_once dedup means only one entry per (category, func, type) lands."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def bad(_r):
        raise RuntimeError("dup")

    @c.cache(cache_if=bad)
    def f(x):
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(10):
            f(i)

    info = f.cache_info()
    # Each call with new args triggers cache_if; dedup means one log entry.
    assert len(info["warnings"]) == 1


def test_cache_clear_drops_warnings_and_dedup(tmp_path):
    """After cache_clear(), warnings log is empty and re-warnings can fire."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def bad(_r):
        raise RuntimeError("clear-me")

    @c.cache(cache_if=bad)
    def f(x):
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f(1)

    assert len(f.cache_info()["warnings"]) == 1

    f.cache_clear()
    assert f.cache_info()["warnings"] == []

    # New misbehavior must fire a fresh warning, not be swallowed by dedup.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        f(1)
    ineffective = [w for w in captured if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert len(ineffective) == 1


def test_cache_info_warnings_independent_across_funcs(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    def bad(_r):
        raise RuntimeError("x")

    @c.cache(cache_if=bad)
    def f(x):
        return x

    @c.cache
    def g(x):
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f(1)
        g(1)

    assert len(f.cache_info()["warnings"]) == 1
    assert g.cache_info()["warnings"] == []


def test_cache_info_includes_warnings_key_even_when_empty(tmp_path):
    """cache_info() always has a 'warnings' key; clean callers see []."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x

    f(1)
    f(1)
    info = f.cache_info()
    assert "warnings" in info
    assert info["warnings"] == []
    # Other stats survived the addition.
    assert info["hits"] == 1
    assert info["misses"] == 1
