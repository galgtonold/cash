"""@cash.cache on async functions tracks file deps under concurrent gather."""
from __future__ import annotations

import asyncio
import time

import pytest

from cash import Cash


async def test_async_auto_track_open(tmp_path):
    """An async function that reads a file should auto-track that file."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    path = tmp_path / "data.txt"
    path.write_text("v1")

    n = {"calls": 0}

    @c.cache
    async def load():
        n["calls"] += 1
        with open(path) as f:
            return f.read()

    assert await load() == "v1"
    assert await load() == "v1"
    assert n["calls"] == 1  # hit

    # Modify file → next call should miss
    time.sleep(0.05)  # ensure mtime ticks
    path.write_text("v2")
    import os
    future = time.time() + 60
    os.utime(path, (future, future))

    assert await load() == "v2"
    assert n["calls"] == 2


async def test_async_gather_isolated_file_deps(tmp_path):
    """Two parallel async cached funcs each reading distinct files
    must each only see their own file as a dep."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    path_a = tmp_path / "a.txt"; path_a.write_text("a")
    path_b = tmp_path / "b.txt"; path_b.write_text("b")

    @c.cache
    async def load_a():
        await asyncio.sleep(0)  # yield to let task B start
        with open(path_a) as f:
            return f.read()

    @c.cache
    async def load_b():
        await asyncio.sleep(0)
        with open(path_b) as f:
            return f.read()

    a, b = await asyncio.gather(load_a(), load_b())
    assert a == "a" and b == "b"

    # Now mutate only path_a. load_a should miss; load_b should hit.
    time.sleep(0.05)
    path_a.write_text("aa")
    import os
    future = time.time() + 60
    os.utime(path_a, (future, future))

    a2 = await load_a()
    b2 = await load_b()
    assert a2 == "aa", "load_a should pick up the changed file"
    assert b2 == "b", "load_b should still hit (its file did not change)"


async def test_async_source_change_invalidates(tmp_path):
    """When the body of an async @cash.cache function changes, the cache misses.
    Mirrors the sync version in the validation harness."""
    src1 = (
        "import asyncio\n"
        "async def f(x):\n"
        "    await asyncio.sleep(0)\n"
        "    return x * 2\n"
    )
    src2 = (
        "import asyncio\n"
        "async def f(x):\n"
        "    await asyncio.sleep(0)\n"
        "    return x * 3\n"
    )

    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    ns1: dict = {}
    exec(src1, ns1)
    cached1 = c.cache(ns1["f"])
    assert await cached1(7) == 14

    ns2: dict = {}
    exec(src2, ns2)
    cached2 = c.cache(ns2["f"])
    # State hash should differ because the function source differs.
    r2 = await cached2(7)
    assert r2 == 21, (
        f"async source-change did not invalidate the cache: got {r2}, expected 21"
    )


async def test_async_gather_cache_hit_across_tasks(tmp_path):
    """A second `asyncio.gather` of the same cached async function should
    HIT the cache populated by the first gather — proven by a call counter,
    not just a value-equality assertion.

    The first gather has two parallel calls with the same args. Without
    single-flight locking (out of scope for this release), both may miss
    and both compute. After the first gather finishes the cache is
    populated. The second, separately-awaited gather should produce zero
    new computes.
    """
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    path = tmp_path / "shared.txt"
    path.write_text("v1")
    n = {"calls": 0}

    @c.cache
    async def load():
        await asyncio.sleep(0)
        n["calls"] += 1
        with open(path) as f:
            return f.read()

    # First gather populates the cache. With no async single-flight,
    # both concurrent calls can both miss and both compute. Calls count
    # after this is 1 or 2 depending on scheduling; we don't pin a
    # specific value, we just record the baseline.
    r1, r2 = await asyncio.gather(load(), load())
    assert r1 == r2 == "v1"
    initial_calls = n["calls"]
    assert initial_calls >= 1, "expected at least one compute in the first gather"

    # Second gather — same args, cache populated. Both task contexts
    # should HIT. Zero new computes.
    r3, r4 = await asyncio.gather(load(), load())
    assert r3 == r4 == "v1"
    assert n["calls"] == initial_calls, (
        f"second gather recomputed: calls went from {initial_calls} to {n['calls']}. "
        f"Cross-task cache hit is broken."
    )


async def test_async_invalidation_visible_to_separate_task(tmp_path):
    """A file mutation made after one task populates the cache must be
    observed by a SEPARATE async task (not just a sequential follow-up
    await on the same task).

    Concretely: gather task A → populate → mutate file → gather task B →
    must miss + recompute → value reflects mutation. Proven via call
    counter.
    """
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    path = tmp_path / "shared.txt"
    path.write_text("v1")
    n = {"calls": 0}

    @c.cache
    async def load():
        await asyncio.sleep(0)
        n["calls"] += 1
        with open(path) as f:
            return f.read()

    # Task A: populate
    (a_val,) = await asyncio.gather(load())
    assert a_val == "v1"
    initial_calls = n["calls"]

    # Mutate file
    time.sleep(0.05)  # ensure mtime ticks
    path.write_text("v2")
    import os
    future = time.time() + 60
    os.utime(path, (future, future))

    # Task B (fresh gather, not just a follow-up await): must observe
    # the mutation and recompute.
    (b_val,) = await asyncio.gather(load())
    assert b_val == "v2", (
        f"fresh task did not pick up the file change: got {b_val!r}"
    )
    assert n["calls"] > initial_calls, (
        f"fresh task did not recompute after file mutation: "
        f"calls stayed at {n['calls']} (was {initial_calls})"
    )
