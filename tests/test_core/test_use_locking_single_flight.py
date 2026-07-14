"""Sync single-flight: ``use_locking=True`` must collapse concurrent same-key
computes to one on the DEFAULT (non-Redis) backends (CAS-112).

``use_locking=True`` promises single-flight — one concurrent computation per
key — but the base backend ``lock()`` used to be a ``nullcontext`` that only
Redis overrode with a real lock. So a sync cached function on the default
(Tiered / in-memory / file) backend saw every concurrent caller compute, with
no warning. These tests pin the fixed behavior: the body runs exactly once and
both threads get the value. The async path already had single-flight (see
``test_async_single_flight.py``) — this covers the sync path.
"""
from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cash import Cash, FileBackend, InMemoryBackend


def _concurrent_calls(c: Cash, n_threads: int = 8):
    """Return (results, run_count) for *n_threads* concurrent same-arg calls."""
    runs = {"n": 0}
    guard = threading.Lock()
    barrier = threading.Barrier(n_threads)

    @c.cache
    def expensive(x):
        with guard:
            runs["n"] += 1
        time.sleep(0.2)
        return x * x

    def call(_):
        barrier.wait()          # release all threads at once → maximal overlap
        return expensive(7)

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(call, range(n_threads)))
    return results, runs["n"]


@pytest.mark.timeout(30)
def test_default_tiered_backend_single_flight(tmp_path):
    """The DEFAULT backend type (TieredBackend, via cache_dir=) computes once."""
    c = Cash(cache_dir=str(tmp_path / "cache"), use_locking=True, register_magic=False)
    results, runs = _concurrent_calls(c)
    assert all(r == 49 for r in results)
    assert runs == 1, "use_locking must collapse concurrent same-key computes to one"


@pytest.mark.timeout(30)
def test_in_memory_backend_single_flight():
    c = Cash(backend=InMemoryBackend(), use_locking=True, register_magic=False)
    results, runs = _concurrent_calls(c)
    assert all(r == 49 for r in results)
    assert runs == 1


@pytest.mark.timeout(30)
def test_file_backend_single_flight():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()), use_locking=True,
             register_magic=False)
    results, runs = _concurrent_calls(c)
    assert all(r == 49 for r in results)
    assert runs == 1


@pytest.mark.timeout(30)
def test_without_locking_still_fans_out():
    """Control: use_locking=False keeps the old behavior — concurrent callers
    each compute, and there is no new lock contention/deadlock."""
    c = Cash(backend=InMemoryBackend(), use_locking=False, register_magic=False)
    results, runs = _concurrent_calls(c)
    assert all(r == 49 for r in results)
    assert runs > 1, "no coalescing without use_locking (documented default)"


@pytest.mark.timeout(30)
def test_distinct_keys_each_compute_once():
    """Single-flight is per-key: different args must not block each other."""
    c = Cash(backend=InMemoryBackend(), use_locking=True, register_magic=False)
    runs = {"n": 0}
    guard = threading.Lock()

    @c.cache
    def expensive(x):
        with guard:
            runs["n"] += 1
        time.sleep(0.1)
        return x * x

    with ThreadPoolExecutor(max_workers=8) as ex:
        # two of each of four distinct keys, concurrently
        args = [2, 3, 4, 5, 2, 3, 4, 5]
        results = list(ex.map(expensive, args))
    assert results == [a * a for a in args]
    assert runs["n"] == 4, "one compute per distinct key"


@pytest.mark.timeout(30)
def test_recursive_same_key_does_not_deadlock():
    """Self-re-entrancy guard: a cached fn that re-enters itself with the SAME
    key on the same thread must not deadlock (per-key RLock, not Lock)."""
    c = Cash(backend=InMemoryBackend(), use_locking=True, register_magic=False)
    depth = {"d": 0}

    @c.cache
    def f(x):
        depth["d"] += 1
        if depth["d"] == 1:
            # re-enter with the same argument → same cache key → same lock
            f(x)
        return x * 10

    assert f(3) == 30            # would hang under a plain threading.Lock


@pytest.mark.timeout(30)
def test_nested_distinct_key_locking_no_deadlock():
    """Memoized recursion (each level a distinct key) under use_locking must
    not deadlock and each distinct key computes exactly once."""
    c = Cash(backend=InMemoryBackend(), use_locking=True, register_magic=False)
    runs = {"n": 0}

    @c.cache
    def fib(n):
        runs["n"] += 1
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    assert fib(10) == 55
    assert runs["n"] == 11, "one compute per distinct n (0..10)"
