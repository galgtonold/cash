"""Per-tier ``max_size_bytes`` constraints.

Each backend declares the upper-bound object size that tier is willing
to accept. ``TieredBackend.set()`` asks each tier individually: an
oversized object skips the constrained tier but still reaches the
unconstrained ones below it.

The universal 1s compute floor stays in place — the size constraint is
*on top* of it, not a replacement for it.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from cash.backends._base import CacheBackend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


class TestBackendMaxSize:
    def test_base_default_is_unbounded(self):
        # CacheBackend's default class attr — no subclass override means no cap.
        assert CacheBackend.max_size_bytes is None

    def test_ram_unbounded(self):
        # In-memory eviction handles RAM pressure; we don't put a hard cap.
        assert InMemoryBackend.max_size_bytes is None

    def test_disk_unbounded(self):
        assert FileBackend.max_size_bytes is None

    def test_redis_caps_at_10_MB(self):
        from cash.backends.redis_backend import RedisBackend
        assert RedisBackend.max_size_bytes == 10 * 1024 * 1024

    def test_sqlite_caps_at_100_MB(self):
        from cash.backends.sqlite_backend import SQLiteBackend
        assert SQLiteBackend.max_size_bytes == 100 * 1024 * 1024

    def test_s3_unbounded(self):
        from cash.backends.s3_backend import S3Backend
        assert S3Backend.max_size_bytes is None


class TestTieredPerTierSkip:
    """A single set() may write to some tiers and not others, based on size."""

    @pytest.fixture
    def redis_backend(self):
        fakeredis = pytest.importorskip("fakeredis")
        _redis = pytest.importorskip("redis")
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            from cash.backends.redis_backend import RedisBackend
            yield RedisBackend(prefix="cash:cap:")

    def test_small_object_lands_in_every_tier(self, tmp_path, redis_backend):
        tiered = TieredBackend(
            [InMemoryBackend(), redis_backend, FileBackend(str(tmp_path))],
            promotion_policy=lambda exec_t, size: True,
        )
        meta = {"execution_time": 2.0}
        tiered.set("small", "x" * 100, meta)
        assert meta["storage"] == ["RAM", "REDIS", "DISK"]

    def test_oversized_object_skips_redis_but_reaches_disk(self, tmp_path, redis_backend):
        """20 MB > Redis cap (10 MB) — Redis is bypassed, RAM and DISK still get it."""
        tiered = TieredBackend(
            [InMemoryBackend(), redis_backend, FileBackend(str(tmp_path))],
            promotion_policy=lambda exec_t, size: True,
        )
        # Pre-size: 20 MB is well past Redis's 10 MB cap.
        big_value = "x" * (20 * 1024 * 1024)
        meta = {"execution_time": 2.0, "size": len(big_value)}
        tiered.set("big", big_value, meta)
        assert "RAM" in meta["storage"]
        assert "DISK" in meta["storage"]
        assert "REDIS" not in meta["storage"]

    def test_compute_floor_still_skips_everything_past_ram(self, tmp_path, redis_backend):
        """A fast computation never promotes past RAM — even tiny objects."""
        tiered = TieredBackend(
            [InMemoryBackend(), redis_backend, FileBackend(str(tmp_path))],
            # default promotion policy: <1s compute → no promotion
        )
        meta = {"execution_time": 0.5}  # below the 1s floor
        tiered.set("fast", "x" * 100, meta)
        assert meta["storage"] == ["RAM"]
        # Redis and Disk neither got the data nor were attempted.
        assert "REDIS" not in meta["storage"]
        assert "DISK" not in meta["storage"]

    def test_force_persist_still_respects_per_tier_max(self, tmp_path, redis_backend):
        """force_persist bypasses the compute floor but NOT the size cap."""
        tiered = TieredBackend(
            [InMemoryBackend(), redis_backend, FileBackend(str(tmp_path))],
        )
        big = "x" * (20 * 1024 * 1024)
        meta = {"execution_time": 0.01, "force_persist": True, "size": len(big)}
        tiered.set("forced_big", big, meta)
        # force_persist landed it in RAM + DISK, but Redis still says "no thanks".
        assert "RAM" in meta["storage"]
        assert "DISK" in meta["storage"]
        assert "REDIS" not in meta["storage"]


class TestBareBackendIgnoresMaxSize:
    """``max_size_bytes`` only gates *tiered* promotion. A user wiring up a
    bare backend can write anything they like — it's their explicit call."""

    def test_bare_redis_accepts_oversized_object(self):
        fakeredis = pytest.importorskip("fakeredis")
        _redis = pytest.importorskip("redis")
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            from cash.backends.redis_backend import RedisBackend
            b = RedisBackend(prefix="cash:bare:")
            big = "x" * (15 * 1024 * 1024)
            # Should not raise — the cap is a tier-promotion hint, not a gate.
            b.set("big", big)
            _, restored = b.get("big")
            assert restored == big

    def test_bare_sqlite_accepts_oversized_object(self, tmp_path):
        from cash.backends.sqlite_backend import SQLiteBackend
        b = SQLiteBackend(str(tmp_path / "c.db"))
        big = "x" * (150 * 1024 * 1024)  # > 100 MB cap
        b.set("big", big)
        _, restored = b.get("big")
        assert restored == big
