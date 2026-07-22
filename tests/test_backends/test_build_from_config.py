"""Construct a backend stack from a CashConfig.

The factory turns a declarative config into a concrete CacheBackend
(possibly wrapped in TieredBackend). Three input shapes:

  1. Empty tier list + backend == "tiered"
     → TieredBackend([InMemoryBackend, FileBackend]) using the top-level
       cache_dir/compress/max_cache_size/max_memory_entries fields.

  2. Empty tier list + backend != "tiered"
     → single backend of that type, built from the top-level
       per-backend connection fields (redis_host, s3_bucket, etc.).

  3. Non-empty tier list
     → TieredBackend([...]) built from the explicit tier specs. Each
       TierConfig overrides nothing at the top level — its own fields
       are authoritative.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from cash.config import CashConfig, TierConfig


# Lazy import so the module loads even before the factory is implemented.
def _build(config):
    from cash.backends.factory import build_backend_from_config
    return build_backend_from_config(config)


# ---------------------------------------------------------------------------
# Simple-mode: backend = "..."
# ---------------------------------------------------------------------------

class TestSimpleModeDefault:
    def test_default_config_builds_tiered_ram_disk(self, tmp_path):
        cfg = CashConfig(cache_dir=str(tmp_path / "c"))
        backend = _build(cfg)
        from cash.backends.tiered_backend import TieredBackend
        from cash.backends.memory_backend import InMemoryBackend
        from cash.backends.file_backend import FileBackend
        assert isinstance(backend, TieredBackend)
        assert isinstance(backend.backends[0], InMemoryBackend)
        assert isinstance(backend.backends[1], FileBackend)

    def test_default_threads_max_size_to_file_tier(self, tmp_path):
        cfg = CashConfig(cache_dir=str(tmp_path / "c"), max_cache_size=12345)
        backend = _build(cfg)
        assert backend.backends[1]._max_size_bytes == 12345

    def test_default_threads_compress(self, tmp_path):
        cfg = CashConfig(cache_dir=str(tmp_path / "c"), compress=True)
        backend = _build(cfg)
        assert backend.backends[1].compress is True

    def test_default_threads_max_memory_entries(self, tmp_path):
        cfg = CashConfig(cache_dir=str(tmp_path / "c"), max_memory_entries=42)
        backend = _build(cfg)
        assert backend.backends[0].max_entries == 42


class TestSimpleModeSingleBackend:
    def test_backend_memory_builds_just_inmemory(self):
        from cash.backends.memory_backend import InMemoryBackend
        cfg = CashConfig(backend="memory", max_memory_entries=99)
        backend = _build(cfg)
        assert isinstance(backend, InMemoryBackend)
        assert backend.max_entries == 99

    def test_backend_file_builds_just_file(self, tmp_path):
        from cash.backends.file_backend import FileBackend
        cfg = CashConfig(backend="file", cache_dir=str(tmp_path / "c"), compress=True)
        backend = _build(cfg)
        assert isinstance(backend, FileBackend)
        assert backend.compress is True

    def test_backend_sqlite_builds_just_sqlite(self, tmp_path):
        from cash.backends.sqlite_backend import SQLiteBackend
        db = tmp_path / "c.db"
        cfg = CashConfig(backend="sqlite", cache_dir=str(db))
        backend = _build(cfg)
        assert isinstance(backend, SQLiteBackend)

    def test_backend_redis_builds_just_redis(self):
        fakeredis = pytest.importorskip("fakeredis")
        _redis = pytest.importorskip("redis")
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            from cash.backends.redis_backend import RedisBackend
            cfg = CashConfig(
                backend="redis",
                redis_host="redis.example.com",
                redis_port=6380,
                redis_prefix="myapp:",
            )
            backend = _build(cfg)
            assert isinstance(backend, RedisBackend)
            assert backend.prefix == "myapp:"

    def test_backend_s3_builds_just_s3(self):
        pytest.importorskip("boto3")
        from unittest.mock import patch as _patch
        with _patch("boto3.client"):
            from cash.backends.s3_backend import S3Backend
            cfg = CashConfig(backend="s3", s3_bucket="my-bucket", s3_prefix="cache/")
            backend = _build(cfg)
            assert isinstance(backend, S3Backend)
            assert backend.bucket == "my-bucket"


# ---------------------------------------------------------------------------
# Advanced-mode: explicit tier list
# ---------------------------------------------------------------------------

class TestAdvancedModeTierList:
    def test_tier_list_builds_tiered_in_order(self, tmp_path):
        fakeredis = pytest.importorskip("fakeredis")
        _redis = pytest.importorskip("redis")
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            from cash.backends.memory_backend import InMemoryBackend
            from cash.backends.redis_backend import RedisBackend
            from cash.backends.file_backend import FileBackend
            from cash.backends.tiered_backend import TieredBackend
            cfg = CashConfig(
                tiers=[
                    TierConfig(type="memory", max_entries=100),
                    TierConfig(type="redis", host="r.example.com"),
                    TierConfig(type="file", cache_dir=str(tmp_path / "fb")),
                ],
            )
            backend = _build(cfg)
            assert isinstance(backend, TieredBackend)
            assert isinstance(backend.backends[0], InMemoryBackend)
            assert backend.backends[0].max_entries == 100
            assert isinstance(backend.backends[1], RedisBackend)
            assert isinstance(backend.backends[2], FileBackend)

    def test_tier_list_overrides_backend_field(self, tmp_path):
        """When tiers is non-empty, the simple-mode `backend` field is
        ignored entirely — tiers wins."""
        from cash.backends.tiered_backend import TieredBackend
        from cash.backends.memory_backend import InMemoryBackend
        cfg = CashConfig(
            backend="redis",   # would be ignored
            tiers=[TierConfig(type="memory")],
        )
        backend = _build(cfg)
        # Even a one-tier list still gets wrapped in TieredBackend so
        # promotion semantics are consistent.
        assert isinstance(backend, TieredBackend)
        assert isinstance(backend.backends[0], InMemoryBackend)


# ---------------------------------------------------------------------------
# Cash() integration — does the constructor use the new factory?
# ---------------------------------------------------------------------------

class TestCashConstructorUsesFactory:
    def test_cash_no_args_uses_default_tiered_stack(self, tmp_path, monkeypatch):
        from cash.backends.tiered_backend import TieredBackend
        from cash import Cash
        monkeypatch.setenv("CASH_CACHE_DIR", str(tmp_path / "c"))
        # Disable env-based pyproject discovery for the test isolation.
        c = Cash(register_magic=False)
        assert isinstance(c.backend, TieredBackend)
        c.shutdown()

    def test_cash_backend_kwarg_overrides_config(self, tmp_path, monkeypatch):
        from cash.backends.memory_backend import InMemoryBackend
        from cash import Cash
        monkeypatch.setenv("CASH_BACKEND", "file")
        # Pass explicit backend instance — kwargs win over env.
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        assert isinstance(c.backend, InMemoryBackend)
        c.shutdown()

    def test_cash_env_picks_backend_type(self, tmp_path, monkeypatch):
        from cash.backends.memory_backend import InMemoryBackend
        from cash import Cash, reset_session
        reset_session()
        monkeypatch.setenv("CASH_BACKEND", "memory")
        c = Cash(register_magic=False)
        assert isinstance(c.backend, InMemoryBackend)
        c.shutdown()


# ---------------------------------------------------------------------------
# The smart-persistence compute floor.
#
# `TieredBackend`'s own default is 1.0s, but the DEFAULT stack the factory
# builds overrides it to 0.1s. Every tiered test until now constructed
# TieredBackend directly, so they all exercised the 1.0s fallback and nothing
# pinned the value users actually get. The 0.1s figure is documented on
# docs/how-it-works/storage.md and is the reason a 0.3s statement persists at
# all -- a silent drift back to 1.0s would stop mid-cost work surviving a
# kernel restart, with no test to say so.
# ---------------------------------------------------------------------------

def _default_config(tmp_path):
    from cash.config import CashConfig
    return CashConfig(cache_dir=str(tmp_path / "cache"))


def test_default_stack_uses_the_tenth_of_a_second_compute_floor(tmp_path):
    from cash.backends.factory import _SMART_PERSIST_COMPUTE_FLOOR_S, build_backend_from_config

    backend = build_backend_from_config(_default_config(tmp_path))
    assert _SMART_PERSIST_COMPUTE_FLOOR_S == 0.1, (
        "the documented smart-persistence floor changed; docs/how-it-works/storage.md "
        "and the promotion-explorer widget quote 0.1s"
    )
    assert getattr(backend, "_min_persist_compute_s", None) == 0.1, (
        "the default backend no longer carries the 0.1s floor -- statements between "
        "0.1s and 1.0s would stop persisting across a kernel restart"
    )


def test_default_stack_keeps_the_twenty_percent_savings_test(tmp_path):
    from cash.backends.factory import build_backend_from_config

    backend = build_backend_from_config(_default_config(tmp_path))
    assert getattr(backend, "_min_persist_savings_pct", None) == 0.20


def test_smart_persistence_off_falls_back_to_the_one_second_floor(tmp_path):
    """`smart_persistence=False` does NOT persist everything unconditionally.

    It drops to `TieredBackend`'s own defaults, which still apply a cost model
    -- just at the more conservative 1.0s floor. `CashConfig.smart_persistence`'s
    docstring claims it "persists everything unconditionally", which is wrong in
    the same way the storage doc was; this pins the real behaviour.
    """
    from cash.backends.factory import build_backend_from_config

    config = _default_config(tmp_path)
    config.smart_persistence = False
    backend = build_backend_from_config(config)
    assert getattr(backend, "_min_persist_compute_s", None) == 1.0


def test_floor_decides_promotion_either_side_of_the_boundary(tmp_path):
    """Behavioural check, not just an attribute read.

    A policy that is present but never consulted would pass the assertions
    above, so drive the real decision on both sides of the floor.
    """
    from cash.backends.factory import build_backend_from_config

    backend = build_backend_from_config(_default_config(tmp_path))
    policy = getattr(backend, "promotion_policy", None)
    assert policy is not None, "default stack should carry a promotion policy"

    size = 5 * 1024 * 1024  # 5 MB: cheap to restore, so compute time decides
    assert policy(0.05, size) is False, "0.05s is under the floor and must stay RAM-only"
    assert policy(0.50, size) is True, "0.50s clears the floor and should persist"
