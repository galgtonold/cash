"""A stored entry that cannot be read back is a miss, never an exception in
the caller's program.

A damaged entry, or one that names a class this process no longer has, is
recomputed in every backend. A tier that cannot read at all (a server down)
is skipped by the tiered backend, and a cached function whose only backend
fails to read computes its result, just as a failed store does not fail it.
"""

from __future__ import annotations

import pickle
import time

import pytest

import cash
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.sqlite_backend import SQLiteBackend
from cash.backends.tiered_backend import TieredBackend
from cash.exceptions import CacheBackendError

from .remote_doubles import FakeRedisClient


class _Gone:
    """Pickled, then removed from the module, as a renamed class would be."""


def _pickle_of_a_missing_class() -> bytes:
    blob = pickle.dumps(_Gone())
    return blob.replace(b"_Gone", b"_Lost")


def test_a_damaged_sqlite_entry_is_recomputed_by_a_cached_function(tmp_path):
    backend = SQLiteBackend(db_path=str(tmp_path / "c.db"))
    c = cash.Cash(backend=backend)
    runs = []

    @c.cache
    def f(x):
        runs.append(x)
        return x * 2

    assert f(3) == 6
    backend._writes.wait_all()
    with backend._lock:
        assert backend._conn.execute("UPDATE cache_entries SET data = x'00010203'").rowcount == 1
        backend._conn.commit()

    assert f(3) == 6
    assert runs == [3, 3]
    backend.shutdown()


def test_a_sqlite_entry_naming_a_missing_class_is_a_miss(tmp_path):
    backend = SQLiteBackend(db_path=str(tmp_path / "c.db"))
    backend.set("k", 1, {})
    backend._writes.wait_all()
    with backend._lock:
        backend._conn.execute("UPDATE cache_entries SET data = ?", (_pickle_of_a_missing_class(),))
        backend._conn.commit()

    assert backend.get("k") == (None, None)
    backend.shutdown()


def test_a_redis_entry_naming_a_missing_class_is_a_miss():
    pytest.importorskip("redis")
    from unittest.mock import patch

    from cash.backends.redis_backend import RedisBackend

    with patch("redis.Redis"):
        backend = RedisBackend(prefix="p:")
    backend.client = FakeRedisClient()
    backend.set("k", 1, {})
    backend._writes.wait_all()
    backend.client.set("p:k:data", _pickle_of_a_missing_class())

    assert backend.get("k") == (None, None)


class _Unreachable(InMemoryBackend):
    """A tier whose every read fails, as a remote one does when its server is down."""

    source_label = "REMOTE"

    def get(self, key):
        raise CacheBackendError("connection refused")


def test_the_tiered_backend_reads_past_a_tier_that_cannot_read():
    slow = InMemoryBackend()
    slow.set("k", 42, {})
    tiered = TieredBackend([_Unreachable(), slow])

    metadata, value = tiered.get("k")
    assert value == 42


def test_a_cached_function_computes_when_its_backend_cannot_read():
    c = cash.Cash(backend=_Unreachable())
    runs = []

    @c.cache
    def f(x):
        runs.append(x)
        time.sleep(0.001)
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert runs == [1, 1]
