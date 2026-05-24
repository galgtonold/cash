"""Basic async support for @cash.cache."""
from __future__ import annotations

import asyncio

import pytest

from cash import Cash


async def test_async_function_caches(tmp_path):
    """Two awaits of the same async function must hit cache on the second."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n_calls = {"n": 0}

    @c.cache
    async def fetch(x):
        n_calls["n"] += 1
        await asyncio.sleep(0)  # ensure we're actually running async
        return x * 11

    r1 = await fetch(3)
    r2 = await fetch(3)
    assert r1 == 33
    assert r2 == 33
    assert n_calls["n"] == 1, f"expected 1 compute, got {n_calls['n']}"


async def test_async_different_args_miss(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    async def f(x):
        return x + 1

    assert await f(1) == 2
    assert await f(2) == 3
    assert await f(1) == 2  # hit


async def test_async_kwargs_in_key(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    async def f(x, *, mult=2):
        return x * mult

    assert await f(5) == 10
    assert await f(5, mult=3) == 15
    assert await f(5, mult=2) == 10  # cache hit


async def test_async_cache_info(tmp_path):
    """cache_info() reflects awaited outcomes, not coroutine creation."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    async def f(x):
        return x

    await f(1)  # miss
    await f(1)  # hit
    info = f.cache_info()
    assert info["hits"] == 1
    assert info["misses"] == 1


async def test_async_returns_none_caches(tmp_path):
    """A None-returning async function caches just like sync."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache
    async def f(x):
        n["calls"] += 1
        return None

    assert await f(1) is None
    assert await f(1) is None
    assert n["calls"] == 1
