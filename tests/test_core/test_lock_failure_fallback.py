"""use_locking must degrade gracefully: if acquiring the backend lock raises
(a Redis LockError on contention, a dropped connection, an OSError on a file
lock), the call falls back to an unlocked compute instead of crashing. A working
lock must still collapse concurrent same-key computes to one.
"""
from __future__ import annotations

import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

from cash import Cash, CashCacheIneffectiveWarning, InMemoryBackend


class _RaisingLockBackend(InMemoryBackend):
    """Backend whose lock() acquisition always raises (simulates a Redis
    LockError / connection failure)."""

    def lock(self, key):
        class _BadLock:
            def __enter__(self_):
                raise RuntimeError("could not acquire lock")

            def __exit__(self_, *a):
                return False
        return _BadLock()


class _ReleaseFailLockBackend(InMemoryBackend):
    """Lock acquires fine but release raises - must not re-run the compute."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._lk = threading.Lock()

    def lock(self, key):
        outer = self

        class _CM:
            def __enter__(self_):
                outer._lk.acquire()
                return self_

            def __exit__(self_, *a):
                outer._lk.release()
                raise RuntimeError("release failed")
        return _CM()


class _WorkingLockBackend(InMemoryBackend):
    """Real per-key threading lock."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._locks: dict = {}
        self._reg = threading.Lock()

    def lock(self, key):
        with self._reg:
            lk = self._locks.get(key)
            if lk is None:
                lk = self._locks[key] = threading.Lock()
        return lk


def test_lock_acquire_failure_falls_back_unlocked():
    c = Cash(backend=_RaisingLockBackend(), use_locking=True)
    calls = {"n": 0}

    @c.cache
    def f(x):
        calls["n"] += 1
        return x * x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert f(7) == 49                     # no crash
        warned = any(issubclass(x.category, CashCacheIneffectiveWarning)
                     and "lock" in str(x.message).lower() for x in w)
    assert warned
    assert calls["n"] == 1                    # computed unlocked


def test_lock_release_failure_does_not_recompute():
    c = Cash(backend=_ReleaseFailLockBackend(), use_locking=True)
    calls = {"n": 0}

    @c.cache
    def f(x):
        calls["n"] += 1
        return x * 2

    assert f(5) == 10                         # compute ran inside lock; release raised
    assert calls["n"] == 1, "compute must not re-run on release failure"


def test_working_lock_collapses_concurrent_computes():
    c = Cash(backend=_WorkingLockBackend(), use_locking=True)
    runs = {"n": 0}
    guard = threading.Lock()

    @c.cache
    def expensive(x):
        with guard:
            runs["n"] += 1
        time.sleep(0.15)
        return x * x

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda _: expensive(7), range(8)))
    assert all(r == 49 for r in res)
    assert runs["n"] == 1
