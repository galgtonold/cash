"""Tests for CacheBackend.tier_labels() — the declarative tier list that
the badge renderer uses to lay out one dot per configured tier.

Each concrete backend reports its own ``source_label`` as a single-element
tier list. Composite backends (TieredBackend, CascadingBackend) flatten
their child labels in order.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from cash.backends._base import CacheBackend
from cash.backends.cascading_backend import CascadingBackend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


class TestBareBackendTierLabels:
    def test_memory_reports_ram(self):
        assert InMemoryBackend().tier_labels() == ["RAM"]

    def test_file_reports_disk(self, tmp_path):
        assert FileBackend(str(tmp_path)).tier_labels() == ["DISK"]

    def test_redis_reports_redis(self):
        from cash.backends.redis_backend import RedisBackend
        with patch("redis.Redis"):
            b = RedisBackend()
        assert b.tier_labels() == ["REDIS"]

    def test_sqlite_reports_sqlite(self, tmp_path):
        from cash.backends.sqlite_backend import SQLiteBackend
        b = SQLiteBackend(str(tmp_path / "c.db"))
        assert b.tier_labels() == ["SQLITE"]

    def test_s3_reports_s3(self):
        pytest.importorskip("boto3")
        from cash.backends.s3_backend import S3Backend
        with patch("boto3.client"):
            b = S3Backend(bucket="x")
        assert b.tier_labels() == ["S3"]

    def test_default_matches_source_label(self):
        """A subclass with only source_label set must auto-pick it up
        without overriding tier_labels()."""
        class _Custom(CacheBackend):
            source_label = "MYTIER"
            def get(self, key): return None, None
            def set(self, key, value, metadata=None, serializer=None): pass
            def delete(self, key): pass
            def clear(self): pass
            def list_entries(self): return []

        assert _Custom().tier_labels() == ["MYTIER"]


class TestTieredBackendTierLabels:
    def test_two_tier_ram_disk(self, tmp_path):
        b = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path))])
        assert b.tier_labels() == ["RAM", "DISK"]

    def test_three_tier_ram_redis_disk(self, tmp_path):
        from cash.backends.redis_backend import RedisBackend
        with patch("redis.Redis"):
            redis = RedisBackend()
        b = TieredBackend([InMemoryBackend(), redis, FileBackend(str(tmp_path))])
        assert b.tier_labels() == ["RAM", "REDIS", "DISK"]

    def test_order_matches_tier_order(self, tmp_path):
        """The tier list must reflect the backend list's order, not alpha or set."""
        from cash.backends.redis_backend import RedisBackend
        with patch("redis.Redis"):
            redis = RedisBackend()
        # Reversed order — should still come out as-configured.
        b = TieredBackend([FileBackend(str(tmp_path)), redis, InMemoryBackend()])
        assert b.tier_labels() == ["DISK", "REDIS", "RAM"]

    def test_nested_tiered_flattens(self, tmp_path):
        """A TieredBackend inside a TieredBackend should still produce a flat list."""
        inner = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "a"))])
        outer = TieredBackend([inner, FileBackend(str(tmp_path / "b"))])
        assert outer.tier_labels() == ["RAM", "DISK", "DISK"]


class TestCascadingBackendTierLabels:
    def test_cascading_flattens_children(self, tmp_path):
        b = CascadingBackend([InMemoryBackend(), FileBackend(str(tmp_path))])
        assert b.tier_labels() == ["RAM", "DISK"]
