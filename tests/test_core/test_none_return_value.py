"""Returning ``None`` from a cached function must produce a cache hit on
the next call, not a miss-and-recompute.

The bug was that cache-hit detection used ``if cached_data is not None``
as the existence check, conflating "key absent" with "key present but
the stored value is None". The contract from ``CacheBackend.get()`` is
that ``(None, None)`` is the only "key absent" signal — a ``None`` data
value alongside a non-None metadata dict means "we cached None for you,
hand it back".
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from cash import Cash
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.sqlite_backend import SQLiteBackend


def _all_simple_backends(tmp_path):
    yield "memory", InMemoryBackend()
    yield "file", FileBackend(str(tmp_path / "fb"), flush_interval=0)
    yield "sqlite", SQLiteBackend(str(tmp_path / "c.db"))
    try:
        import fakeredis
        import redis as _redis
    except ImportError:
        return
    with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
        from cash.backends.redis_backend import RedisBackend
        yield "redis", RedisBackend(prefix="cash:nrt:")


class TestNoneReturnValueCaches:
    """A None-returning cached function must hit cache on the second call."""

    def test_memory(self, tmp_path):
        self._roundtrip(InMemoryBackend())

    def test_file(self, tmp_path):
        b = FileBackend(str(tmp_path / "fb"), flush_interval=0)
        try:
            self._roundtrip(b)
        finally:
            b.shutdown()

    def test_sqlite(self, tmp_path):
        b = SQLiteBackend(str(tmp_path / "c.db"))
        try:
            self._roundtrip(b)
        finally:
            b.shutdown()

    def test_tiered_ram_disk(self, tmp_path):
        """Tiered: a stored None must propagate through TieredBackend.get."""
        from cash.backends.file_backend import FileBackend
        from cash.backends.tiered_backend import TieredBackend
        b = TieredBackend(
            [InMemoryBackend(), FileBackend(str(tmp_path / "fb"), flush_interval=0)],
            promotion_policy=lambda exec_t, size: True,
        )
        try:
            self._roundtrip(b)
        finally:
            b.shutdown()

    def test_redis_via_fakeredis(self):
        fakeredis = pytest.importorskip("fakeredis")
        _redis = pytest.importorskip("redis")
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            from cash.backends.redis_backend import RedisBackend
            b = RedisBackend(prefix="cash:nrt:")
            try:
                self._roundtrip(b)
            finally:
                b.shutdown()

    @staticmethod
    def _roundtrip(backend):
        c = Cash(backend=backend, register_magic=False)
        calls = [0]

        @c.cache
        def f(x):
            calls[0] += 1
            return None  # the value under test

        assert f(1) is None
        assert f(1) is None
        assert calls[0] == 1, (
            f"expected one compute, got {calls[0]} — None return value treated as miss"
        )

        # Verify the metadata is actually populated (not a phantom hit).
        # The second call's args_hash matches the first; if the backend
        # genuinely held None, the cache_hit_log entry should reflect that.
        # Spot-check via show_stats which iterates the decorator_call_log.
        hits = [e for e in c._decorator_call_log if e["cache_hit"]]
        assert len(hits) == 1, f"expected 1 hit log entry, got {len(hits)}"
