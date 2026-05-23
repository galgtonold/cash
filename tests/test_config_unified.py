"""Contract tests for the unified config layer.

Resolution precedence (highest priority first):
    1. Explicit constructor kwargs            (Cash(redis_host="..."))
    2. Environment variables                  (CASH_REDIS_HOST=...)
    3. Project config (./pyproject.toml [tool.cash])
    4. User config (~/.config/cash/config.toml on Linux/macOS;
                    %APPDATA%/cash/config.toml on Windows)
    5. CashConfig dataclass defaults

Every field on CashConfig is settable via every layer. Tier-list entries
are settable individually via CASH_TIER_<N>_<FIELD> env vars.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Import names that don't exist yet are imported lazily inside tests so
# pytest collection doesn't blow up before the implementation lands.


# ---------------------------------------------------------------------------
# CashConfig structure
# ---------------------------------------------------------------------------

class TestCashConfigStructure:
    def test_defaults_are_sensible(self):
        from cash.config import CashConfig
        c = CashConfig()
        assert c.cache_dir == ".cash"
        assert c.debug is False
        assert c.compress is False
        assert c.max_cache_size == 1024 ** 3
        # Cost-aware policy unchanged.
        assert c.smart_persistence is True
        assert c.smart_persistence_threshold == 1.0
        assert c.min_execution_time_to_cache_seconds == 0.01
        # Backend selection — defaults to the auto-built tiered stack.
        assert c.backend == "tiered"
        # Connection details default to placeholders, not None, so they
        # round-trip through TOML unambiguously.
        assert c.redis_host == "localhost"
        assert c.redis_port == 6379
        assert c.s3_bucket == ""
        # No explicit tier list by default — backend-from-config will
        # synthesise [memory, file] from the top-level fields.
        assert c.tiers == []

    def test_to_dict_excludes_private_fields(self):
        from cash.config import CashConfig
        d = CashConfig().to_dict()
        assert "_source" not in d
        assert "cache_dir" in d
        assert "backend" in d
        assert "redis_host" in d


# ---------------------------------------------------------------------------
# TierConfig
# ---------------------------------------------------------------------------

class TestTierConfig:
    def test_construct_memory_tier(self):
        from cash.config import TierConfig
        t = TierConfig(type="memory", max_entries=1000)
        assert t.type == "memory"
        assert t.max_entries == 1000

    def test_construct_redis_tier(self):
        from cash.config import TierConfig
        t = TierConfig(type="redis", host="redis.internal", port=6380, prefix="myapp:")
        assert t.type == "redis"
        assert t.host == "redis.internal"
        assert t.port == 6380
        assert t.prefix == "myapp:"

    def test_construct_s3_tier(self):
        from cash.config import TierConfig
        t = TierConfig(type="s3", bucket="my-cache", region="us-east-1")
        assert t.type == "s3"
        assert t.bucket == "my-cache"

    def test_unknown_type_rejected(self):
        from cash.config import TierConfig
        with pytest.raises(ValueError, match="Unknown tier type"):
            TierConfig(type="quantum-flux-capacitor")


# ---------------------------------------------------------------------------
# Env var loading
# ---------------------------------------------------------------------------

class TestEnvVarLoading:
    """Every CashConfig field must have a CASH_<UPPER_NAME> binding."""

    def test_top_level_fields(self, monkeypatch):
        from cash.config import _load_env_config
        monkeypatch.setenv("CASH_CACHE_DIR", "/var/cache")
        monkeypatch.setenv("CASH_DEBUG", "true")
        monkeypatch.setenv("CASH_COMPRESS", "1")
        monkeypatch.setenv("CASH_MAX_CACHE_SIZE", "5000000000")
        monkeypatch.setenv("CASH_SMART_PERSISTENCE_THRESHOLD", "0.5")
        monkeypatch.setenv("CASH_BACKEND", "redis")
        monkeypatch.setenv("CASH_REDIS_HOST", "myredis.example.com")
        monkeypatch.setenv("CASH_REDIS_PORT", "6380")
        env = _load_env_config()
        assert env["cache_dir"] == "/var/cache"
        assert env["debug"] is True
        assert env["compress"] is True
        assert env["max_cache_size"] == 5000000000
        assert env["smart_persistence_threshold"] == 0.5
        assert env["backend"] == "redis"
        assert env["redis_host"] == "myredis.example.com"
        assert env["redis_port"] == 6380

    def test_invalid_int_skipped(self, monkeypatch):
        from cash.config import _load_env_config
        monkeypatch.setenv("CASH_MAX_CACHE_SIZE", "not_a_number")
        env = _load_env_config()
        assert "max_cache_size" not in env

    def test_tier_env_vars(self, monkeypatch):
        """CASH_TIER_<N>_<FIELD> populates the tiers list, replacing
        whatever was in the config files for that tier index."""
        from cash.config import _load_env_config
        monkeypatch.setenv("CASH_TIER_0_TYPE", "memory")
        monkeypatch.setenv("CASH_TIER_0_MAX_ENTRIES", "5000")
        monkeypatch.setenv("CASH_TIER_1_TYPE", "redis")
        monkeypatch.setenv("CASH_TIER_1_HOST", "redis.internal")
        monkeypatch.setenv("CASH_TIER_1_PORT", "6380")
        env = _load_env_config()
        # Tier overrides come through as a structured list of partial
        # dicts keyed by tier index.
        assert env["tiers"][0]["type"] == "memory"
        assert env["tiers"][0]["max_entries"] == 5000
        assert env["tiers"][1]["type"] == "redis"
        assert env["tiers"][1]["host"] == "redis.internal"
        assert env["tiers"][1]["port"] == 6380


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------

class TestTomlLoading:
    def test_load_pyproject_toml_section(self, tmp_path):
        """[tool.cash] in a pyproject.toml is what we look for."""
        from cash.config import _load_toml_config
        py = tmp_path / "pyproject.toml"
        py.write_text(
            '[tool.cash]\n'
            'cache_dir = "from_pyproject"\n'
            'debug = true\n'
            '[[tool.cash.tiers]]\n'
            'type = "memory"\n'
            'max_entries = 999\n'
            '[[tool.cash.tiers]]\n'
            'type = "redis"\n'
            'host = "myhost"\n',
            encoding="utf-8",
        )
        result = _load_toml_config(py)
        assert result["cache_dir"] == "from_pyproject"
        assert result["debug"] is True
        assert result["tiers"][0] == {"type": "memory", "max_entries": 999}
        assert result["tiers"][1] == {"type": "redis", "host": "myhost"}

    def test_load_standalone_toml_with_cash_section(self, tmp_path):
        """A bare ~/.config/cash/config.toml with [cash] at top level."""
        from cash.config import _load_toml_config
        p = tmp_path / "config.toml"
        p.write_text('[cash]\ncache_dir = "from_user_config"\n', encoding="utf-8")
        result = _load_toml_config(p)
        assert result["cache_dir"] == "from_user_config"

    def test_load_missing_file_returns_empty(self, tmp_path):
        from cash.config import _load_toml_config
        assert _load_toml_config(tmp_path / "absent.toml") == {}

    def test_load_invalid_toml_returns_empty(self, tmp_path):
        from cash.config import _load_toml_config
        p = tmp_path / "broken.toml"
        p.write_text("not [ valid toml", encoding="utf-8")
        assert _load_toml_config(p) == {}


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

class TestPrecedence:
    """kwargs > env > project (pyproject.toml) > user (~/.config) > defaults."""

    def test_user_config_overrides_defaults(self, tmp_path, monkeypatch):
        from cash.config import get_config
        user = tmp_path / "user_config.toml"
        user.write_text('[cash]\ncache_dir = "from_user"\n', encoding="utf-8")
        cfg = get_config(user_config_path=user, project_config_path=None)
        assert cfg.cache_dir == "from_user"

    def test_project_overrides_user(self, tmp_path, monkeypatch):
        from cash.config import get_config
        user = tmp_path / "user_config.toml"
        user.write_text('[cash]\ncache_dir = "from_user"\ndebug = false\n', encoding="utf-8")
        proj = tmp_path / "pyproject.toml"
        proj.write_text(
            '[tool.cash]\ncache_dir = "from_project"\n',
            encoding="utf-8",
        )
        cfg = get_config(user_config_path=user, project_config_path=proj)
        assert cfg.cache_dir == "from_project"
        # debug stays as user-config value (project didn't override it)
        assert cfg.debug is False

    def test_env_overrides_project(self, tmp_path, monkeypatch):
        from cash.config import get_config
        proj = tmp_path / "pyproject.toml"
        proj.write_text(
            '[tool.cash]\ncache_dir = "from_project"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CASH_CACHE_DIR", "from_env")
        cfg = get_config(user_config_path=None, project_config_path=proj)
        assert cfg.cache_dir == "from_env"

    def test_kwargs_override_env(self, tmp_path, monkeypatch):
        """Explicit kwargs to get_config (mirroring what Cash() does
        internally) override env vars."""
        from cash.config import get_config
        monkeypatch.setenv("CASH_CACHE_DIR", "from_env")
        cfg = get_config(overrides={"cache_dir": "from_kwarg"})
        assert cfg.cache_dir == "from_kwarg"

    def test_full_chain(self, tmp_path, monkeypatch):
        from cash.config import get_config
        user = tmp_path / "user_config.toml"
        user.write_text(
            '[cash]\ncache_dir = "from_user"\ndebug = false\ncompress = false\n',
            encoding="utf-8",
        )
        proj = tmp_path / "pyproject.toml"
        proj.write_text(
            '[tool.cash]\ndebug = true\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CASH_COMPRESS", "true")
        cfg = get_config(
            user_config_path=user,
            project_config_path=proj,
            overrides={"max_cache_size": 999},
        )
        # cache_dir: only user set → user
        assert cfg.cache_dir == "from_user"
        # debug: user=False, project=True → project
        assert cfg.debug is True
        # compress: user=False, env=True → env
        assert cfg.compress is True
        # max_cache_size: only overrides set → kwarg
        assert cfg.max_cache_size == 999


# ---------------------------------------------------------------------------
# Tier override from env merges with tier from TOML
# ---------------------------------------------------------------------------

class TestTierEnvMergesWithToml:
    def test_env_tier_field_overrides_toml_tier_field(self, tmp_path, monkeypatch):
        """The TOML declared a Redis tier; env var overrides one field of it."""
        from cash.config import get_config
        proj = tmp_path / "pyproject.toml"
        proj.write_text(
            '[[tool.cash.tiers]]\n'
            'type = "memory"\n'
            '[[tool.cash.tiers]]\n'
            'type = "redis"\n'
            'host = "default.example.com"\n'
            'port = 6379\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CASH_TIER_1_HOST", "prod-redis.example.com")
        cfg = get_config(user_config_path=None, project_config_path=proj)
        assert cfg.tiers[0].type == "memory"
        assert cfg.tiers[1].type == "redis"
        assert cfg.tiers[1].host == "prod-redis.example.com"
        assert cfg.tiers[1].port == 6379

    def test_env_tier_introduces_new_tier_when_toml_silent(self, tmp_path, monkeypatch):
        """No tiers in TOML — env declares them entirely."""
        from cash.config import get_config
        monkeypatch.setenv("CASH_TIER_0_TYPE", "redis")
        monkeypatch.setenv("CASH_TIER_0_HOST", "via-env.example.com")
        cfg = get_config(user_config_path=None, project_config_path=None)
        assert len(cfg.tiers) == 1
        assert cfg.tiers[0].type == "redis"
        assert cfg.tiers[0].host == "via-env.example.com"


# ---------------------------------------------------------------------------
# Source tracking — useful for `cash --info` debugging
# ---------------------------------------------------------------------------

class TestSourceTracking:
    def test_source_is_defaults_when_no_inputs(self, tmp_path):
        from cash.config import get_config
        cfg = get_config(user_config_path=None, project_config_path=None)
        assert cfg._source == "defaults"

    def test_source_mentions_project_when_used(self, tmp_path):
        from cash.config import get_config
        proj = tmp_path / "pyproject.toml"
        proj.write_text('[tool.cash]\ndebug = true\n', encoding="utf-8")
        cfg = get_config(user_config_path=None, project_config_path=proj)
        assert "project" in cfg._source

    def test_source_mentions_env(self, monkeypatch):
        from cash.config import get_config
        monkeypatch.setenv("CASH_DEBUG", "1")
        cfg = get_config(user_config_path=None, project_config_path=None)
        assert "env" in cfg._source
