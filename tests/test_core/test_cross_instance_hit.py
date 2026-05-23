"""When two Cash instances open the same backing store and the
same function is defined identically in both, the second instance
must HIT the cache rather than recompute.

This pins down a real-world script scenario:

    # Run 1
    c = Cash(backend=FileBackend("/cache"))
    @c.cache
    def f(x): return expensive(x)
    f(7)            # computes
    # process exits → atexit drains writes

    # Run 2 (same script, new Python process — or new Cash() in tests)
    c = Cash(backend=FileBackend("/cache"))
    @c.cache
    def f(x): return expensive(x)
    f(7)            # should HIT — same store, same source

The earlier decorator-validation harness reported a "limitation" here,
but that was a harness artefact: it defined ``f`` in the first Cash and
``f2`` in the second, which have different ``__qualname__`` and
legitimately produce different cache keys. The tests below use the
same function source on both sides.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from cash import Cash


def _make_cached_fn(c: Cash, calls: dict):
    """Identical helper used by both Cash instances — both call sites
    produce a function with the same ``__module__``, ``__qualname__``,
    and source bytes, so cache_key must match."""

    @c.cache
    def f(x):
        time.sleep(0.12)  # clear smart-persistence floor
        calls["n"] += 1
        return x * 3

    return f


class TestSameSourceHitsAcrossInstances:
    def test_file_backend(self, tmp_path):
        from cash.backends.file_backend import FileBackend
        calls = {"n": 0}

        c1 = Cash(backend=FileBackend(str(tmp_path / "store"), flush_interval=0), register_magic=False)
        f1 = _make_cached_fn(c1, calls)
        assert f1(11) == 33
        c1.shutdown()

        c2 = Cash(backend=FileBackend(str(tmp_path / "store"), flush_interval=0), register_magic=False)
        f2 = _make_cached_fn(c2, calls)
        assert f2(11) == 33
        c2.shutdown()
        assert calls["n"] == 1, (
            f"second instance recomputed (calls={calls['n']}) — same source "
            "should produce same cache key"
        )

    def test_sqlite_backend(self, tmp_path):
        from cash.backends.sqlite_backend import SQLiteBackend
        calls = {"n": 0}
        db = str(tmp_path / "c.db")

        c1 = Cash(backend=SQLiteBackend(db), register_magic=False)
        f1 = _make_cached_fn(c1, calls)
        assert f1(11) == 33
        c1.shutdown()

        c2 = Cash(backend=SQLiteBackend(db), register_magic=False)
        f2 = _make_cached_fn(c2, calls)
        assert f2(11) == 33
        c2.shutdown()
        assert calls["n"] == 1

    def test_redis_backend_via_fakeredis(self):
        import fakeredis
        import redis as _redis
        calls = {"n": 0}
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            from cash.backends.redis_backend import RedisBackend
            prefix = "cash:cit:"

            c1 = Cash(backend=RedisBackend(prefix=prefix), register_magic=False)
            f1 = _make_cached_fn(c1, calls)
            assert f1(11) == 33
            c1.shutdown()

            c2 = Cash(backend=RedisBackend(prefix=prefix), register_magic=False)
            f2 = _make_cached_fn(c2, calls)
            assert f2(11) == 33
            c2.shutdown()
            assert calls["n"] == 1
