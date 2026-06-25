"""Concurrent awaits of the same key should compute once (async single-flight).

`asyncio.gather` of N calls to the same cached coroutine used to compute N
times - the sync `use_locking` path never applied to async functions. With
`use_locking=True` Cash now coalesces concurrent same-key awaits in-process:
one coroutine (the leader) computes and stores, the rest wait and read the
stored result. Without `use_locking` the old fan-out behavior is preserved.
"""
from __future__ import annotations

import asyncio
import tempfile

from cash import Cash, FileBackend, InMemoryBackend


def test_async_single_flight_collapses_concurrent_awaits():
    c = Cash(backend=InMemoryBackend(), use_locking=True)
    runs = {"n": 0}

    @c.cache
    async def fetch(x):
        runs["n"] += 1
        await asyncio.sleep(0.05)
        return x * x

    async def main():
        return await asyncio.gather(*[fetch(7) for _ in range(8)])

    results = asyncio.run(main())
    assert all(r == 49 for r in results)
    assert runs["n"] == 1, "leader computes once; followers read the stored value"


def test_async_single_flight_distinct_keys_each_compute_once():
    c = Cash(backend=InMemoryBackend(), use_locking=True)
    runs = {"n": 0}

    @c.cache
    async def fetch(x):
        runs["n"] += 1
        await asyncio.sleep(0.05)
        return x * x

    async def main():
        # two of each key, concurrently
        return await asyncio.gather(fetch(2), fetch(3), fetch(2), fetch(3))

    results = asyncio.run(main())
    assert results == [4, 9, 4, 9]
    assert runs["n"] == 2, "one compute per distinct key"


def test_async_without_locking_still_fans_out():
    c = Cash(backend=InMemoryBackend(), use_locking=False)
    runs = {"n": 0}

    @c.cache
    async def fetch(x):
        runs["n"] += 1
        await asyncio.sleep(0.05)
        return x * x

    async def main():
        return await asyncio.gather(*[fetch(7) for _ in range(5)])

    results = asyncio.run(main())
    assert all(r == 49 for r in results)
    assert runs["n"] == 5, "no coalescing without use_locking (documented default)"


def test_async_single_flight_followers_get_result_when_leader_rejects_cache():
    """If the leader's cache_if rejects the value (nothing stored), followers
    must still return a correct result - they fall through and compute."""
    c = Cash(backend=InMemoryBackend(), use_locking=True)
    runs = {"n": 0}

    @c.cache(cache_if=lambda v: False)   # never store
    async def fetch(x):
        runs["n"] += 1
        await asyncio.sleep(0.02)
        return x * x

    async def main():
        return await asyncio.gather(*[fetch(6) for _ in range(4)])

    results = asyncio.run(main())
    assert all(r == 36 for r in results)
    assert runs["n"] >= 1            # correctness preserved; coalescing may not apply


def test_async_single_flight_propagates_exception():
    c = Cash(backend=InMemoryBackend(), use_locking=True)

    @c.cache
    async def boom(x):
        await asyncio.sleep(0.01)
        raise ValueError("nope")

    async def main():
        results = await asyncio.gather(
            *[boom(1) for _ in range(3)], return_exceptions=True
        )
        return results

    results = asyncio.run(main())
    assert all(isinstance(r, ValueError) for r in results)


def test_async_single_flight_sequential_calls_hit_cache():
    """Single-flight must not break the ordinary sequential hit path."""
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()), use_locking=True)
    runs = {"n": 0}

    @c.cache
    async def fetch(x):
        runs["n"] += 1
        return x + 1

    async def main():
        a = await fetch(10)
        b = await fetch(10)     # sequential - normal cache hit
        return a, b

    a, b = asyncio.run(main())
    assert a == b == 11
    assert runs["n"] == 1
