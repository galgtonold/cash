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
     → TieredBackend([...]) built from the explicit tier specs. A field
       a TierConfig leaves unset comes from the top-level field.
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
        from cash.backends.file_backend import FileBackend
        from cash.backends.memory_backend import InMemoryBackend
        from cash.backends.tiered_backend import TieredBackend

        assert isinstance(backend, TieredBackend)
        assert isinstance(backend.backends[0], InMemoryBackend)
        assert isinstance(backend.backends[1], FileBackend)

    def test_default_threads_max_size_to_file_tier(self, tmp_path):
        cfg = CashConfig(cache_dir=str(tmp_path / "c"), max_cache_size=12345)
        backend = _build(cfg)
        assert backend.backends[1].evictor.max_size_bytes == 12345

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
            from cash.backends.file_backend import FileBackend
            from cash.backends.memory_backend import InMemoryBackend
            from cash.backends.redis_backend import RedisBackend
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
        from cash.backends.memory_backend import InMemoryBackend
        from cash.backends.tiered_backend import TieredBackend

        cfg = CashConfig(
            backend="redis",  # would be ignored
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
        from cash import Cash
        from cash.backends.tiered_backend import TieredBackend

        monkeypatch.setenv("CASH_CACHE_DIR", str(tmp_path / "c"))
        # Disable env-based pyproject discovery for the test isolation.
        c = Cash(register_magic=False)
        assert isinstance(c.backend, TieredBackend)
        c.shutdown()

    def test_cash_backend_kwarg_overrides_config(self, tmp_path, monkeypatch):
        from cash import Cash
        from cash.backends.memory_backend import InMemoryBackend

        monkeypatch.setenv("CASH_BACKEND", "file")
        # Pass explicit backend instance — kwargs win over env.
        c = Cash(backend=InMemoryBackend(), register_magic=False)
        assert isinstance(c.backend, InMemoryBackend)
        c.shutdown()

    def test_cash_env_picks_backend_type(self, tmp_path, monkeypatch):
        from cash import Cash, reset_session
        from cash.backends.memory_backend import InMemoryBackend

        reset_session()
        monkeypatch.setenv("CASH_BACKEND", "memory")
        c = Cash(register_magic=False)
        assert isinstance(c.backend, InMemoryBackend)
        c.shutdown()


# ---------------------------------------------------------------------------
# The persistence compute floor: 0.1 s, documented on
# docs/how-it-works/storage.md, and the reason a 0.3 s statement survives a
# kernel restart at all.
# ---------------------------------------------------------------------------


def _default_config(tmp_path):
    from cash.config import CashConfig

    return CashConfig(cache_dir=str(tmp_path / "cache"))


def test_default_stack_uses_the_tenth_of_a_second_compute_floor(tmp_path):
    from cash.backends.factory import build_backend_from_config

    backend = build_backend_from_config(_default_config(tmp_path))
    assert backend.policy.compute_floor_s == 0.1, (
        "the documented persistence floor changed; docs/how-it-works/storage.md quotes 0.1s"
    )


def test_default_stack_keeps_the_twenty_percent_savings_test(tmp_path):
    from cash.backends.factory import build_backend_from_config

    backend = build_backend_from_config(_default_config(tmp_path))
    assert backend.policy.min_savings_pct == 0.20


def test_a_hand_built_stack_uses_the_same_floor(tmp_path):
    """There is one floor: a `TieredBackend` built in code does not get a
    stricter one than the stack cash builds from config."""
    from cash.backends import FileBackend, InMemoryBackend, TieredBackend

    backend = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"))])
    assert backend.policy.compute_floor_s == 0.1


def test_floor_decides_promotion_either_side_of_the_boundary(tmp_path):
    """Behavioural check, not just an attribute read.

    A policy that is present but never consulted would pass the assertions
    above, so drive the real decision on both sides of the floor.
    """
    from cash.backends.factory import build_backend_from_config

    backend = build_backend_from_config(_default_config(tmp_path))
    size = 5 * 1024 * 1024  # 5 MB: cheap to restore, so compute time decides
    assert backend.policy.pays_to_restore(0.05, size) is False, "0.05s is under the floor and must stay RAM-only"
    assert backend.policy.pays_to_restore(0.50, size) is True, "0.50s clears the floor and should persist"


# ---------------------------------------------------------------------------
# One way to describe a backend
# ---------------------------------------------------------------------------


def test_a_file_tier_from_a_list_is_sized_to_the_machine_like_the_default(tmp_path):
    """A tier list's file tier with no size set gets the adaptive cap the
    default stack's file tier gets, not a cap frozen at start-up."""
    backend = _build(
        CashConfig(cache_dir=str(tmp_path / "c"), tiers=[TierConfig(type="memory"), TierConfig(type="file")])
    )
    assert backend.backends[1].evictor.adaptive is True
    pinned = _build(
        CashConfig(cache_dir=str(tmp_path / "d"), tiers=[TierConfig(type="file", max_size_bytes=10_000_000)])
    )
    assert pinned.backends[0].evictor.adaptive is False


def test_tiered_is_not_a_tier_type():
    with pytest.raises(ValueError, match="tiered"):
        TierConfig(type="tiered")


def test_a_setting_no_tier_uses_leaves_the_described_stack_alone(tmp_path):
    from cash.backends.factory import tier_specs

    base = CashConfig(cache_dir=str(tmp_path / "c"))
    assert tier_specs(base) == tier_specs(CashConfig(cache_dir=str(tmp_path / "c"), redis_host="elsewhere"))
    assert tier_specs(base) != tier_specs(CashConfig(cache_dir=str(tmp_path / "other")))
    on_redis = CashConfig(backend="redis")
    assert tier_specs(on_redis) != tier_specs(CashConfig(backend="redis", redis_host="elsewhere"))


# ---------------------------------------------------------------------------
# Every tier setting is used, or said to do nothing
# ---------------------------------------------------------------------------


def test_a_sqlite_tier_gets_its_wal_mode(tmp_path):
    """``wal_mode`` was documented for SQLite tiers and never passed on."""
    cfg = CashConfig(cache_dir=str(tmp_path / "c"), tiers=[TierConfig(type="sqlite", wal_mode=False)])
    sqlite = _build(cfg).backends[0]
    try:
        assert sqlite._conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    finally:
        sqlite.shutdown()


def test_a_value_over_a_sqlite_tiers_cap_skips_it(tmp_path):
    """``max_size_bytes`` on a SQLite tier is its promotion cap, as the field
    promises: a 5 KB value is not offered to a 1 KB tier, written, and then
    evicted at once. The class-wide 100 MiB hint stood in for it."""
    cfg = CashConfig(
        cache_dir=str(tmp_path / "c"),
        tiers=[TierConfig(type="memory"), TierConfig(type="sqlite", max_size_bytes=1000)],
    )
    backend = _build(cfg)
    sqlite = backend.backends[1]
    try:
        assert sqlite.promotion_size_cap() == 1000
        backend.set("small", b"x" * 100, {"execution_time": 0.5})
        from cash.exceptions import CashCacheIneffectiveWarning

        with pytest.warns(CashCacheIneffectiveWarning, match="CACHE-VALUE-TOO-BIG"):
            backend.set("big", b"x" * 5000, {"execution_time": 0.5})
        sqlite._writes.wait_all()
        assert sqlite.get("small")[1] == b"x" * 100, "the control: a value under the cap is written"
        assert sqlite.get("big") == (None, None)
    finally:
        backend.shutdown()


@pytest.mark.parametrize(
    ("tier", "unused"),
    [
        (TierConfig(type="memory", default_ttl=5), "default_ttl"),
        (TierConfig(type="redis", max_size_bytes=10), "max_size_bytes"),
        (TierConfig(type="s3", bucket="b", default_ttl=5), "default_ttl"),
        (TierConfig(type="file", wal_mode=False), "wal_mode"),
    ],
    ids=lambda v: v if isinstance(v, str) else v.type,
)
def test_a_setting_a_tier_does_not_use_is_reported(tier, unused, monkeypatch):
    """Such a setting was dropped without a word; a memory tier's
    ``default_ttl`` looked like it gave RAM entries a lifetime."""
    import dataclasses

    from cash import config as cash_config
    from cash.exceptions import CashCacheIneffectiveWarning

    monkeypatch.setattr(cash_config, "_CONFIG_NOTICES", set())
    with pytest.warns(CashCacheIneffectiveWarning, match=rf"CONFIG-INVALID.*{tier.type} tier sets {unused}"):
        dataclasses.replace(tier)


def test_a_tier_using_only_its_own_settings_is_not_reported(recwarn):
    TierConfig(type="sqlite", default_ttl=5, wal_mode=False, max_size_bytes=10, db_path="x.db")
    TierConfig(type="redis", host="h", port=1, db=0, password="p", prefix="x:")
    assert not [w for w in recwarn if "CONFIG-INVALID" in str(w.message)]


@pytest.mark.parametrize("kind", ["memory", "file", "sqlite", "redis", "s3"])
def test_the_reported_fields_are_the_ones_the_factory_builds_from(kind):
    """The list of what a tier type uses is the factory's own."""
    from cash.backends.factory import _settings
    from cash.config import _TIER_FIELDS

    assert set(_settings(TierConfig(type=kind), CashConfig())) == _TIER_FIELDS[kind]
