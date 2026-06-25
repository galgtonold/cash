"""Runtime configuration mutation via ``cash.configure(**kwargs)``.

The function targets the default ``Cash`` singleton. Hot fields (debug,
policy knobs) update the active config in place. Backend-affecting
fields trigger a clean shutdown of the current backend and a rebuild
from the new config.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts and ends with no global Cash instance."""
    import cash
    cash.reset_session()
    yield
    cash.reset_session()


# ---------------------------------------------------------------------------
# Hot fields — just dataclass update, no backend rebuild
# ---------------------------------------------------------------------------

class TestHotFields:
    def test_configure_debug(self):
        import cash
        cash.configure(debug=True)
        assert cash._get_global_cash().config.debug is True
        assert cash._get_global_cash().debug is True

    def test_configure_smart_persistence_threshold(self):
        import cash
        cash.configure(smart_persistence_threshold=0.25)
        assert cash._get_global_cash().config.smart_persistence_threshold == 0.25

    def test_configure_min_cache_savings_pct(self):
        import cash
        cash.configure(min_cache_savings_pct=0.10)
        assert cash._get_global_cash().config.min_cache_savings_pct == 0.10

    def test_hot_field_does_not_rebuild_backend(self):
        import cash
        # Touch backend to ensure it's built and capture identity.
        backend_before = cash._get_global_cash().backend
        cash.configure(debug=True, smart_persistence_threshold=0.5)
        backend_after = cash._get_global_cash().backend
        assert backend_before is backend_after, \
            "hot field reconfigure should not rebuild the backend"


# ---------------------------------------------------------------------------
# Backend-affecting fields — rebuild + swap
# ---------------------------------------------------------------------------

class TestBackendRebuild:
    def test_configure_backend_switches_type(self, tmp_path):
        """Changing config.backend = 'memory' replaces the active backend."""
        import cash
        from cash.backends.memory_backend import InMemoryBackend
        # Force initial backend instantiation
        cash._get_global_cash().backend  # touch
        cash.configure(backend="memory")
        new_backend = cash._get_global_cash().backend
        assert isinstance(new_backend, InMemoryBackend)

    def test_configure_drains_old_backend(self, tmp_path):
        """The previous backend gets shutdown() called so its async
        writes don't get lost when we swap."""
        import cash
        c = cash._get_global_cash()
        old = c.backend
        shutdown_called = []
        original_shutdown = old.shutdown

        def spy():
            shutdown_called.append(True)
            return original_shutdown()
        old.shutdown = spy

        cash.configure(backend="memory")
        assert shutdown_called == [True]

    def test_configure_cache_dir_rebuilds(self, tmp_path):
        """cache_dir is a backend-affecting field even when backend type stays the same."""
        import cash
        c = cash._get_global_cash()
        before = c.backend
        cash.configure(cache_dir=str(tmp_path / "new"))
        after = c.backend
        assert before is not after


# ---------------------------------------------------------------------------
# Stale fields — silently store, no error, no rebuild
# ---------------------------------------------------------------------------

class TestStaleFields:
    def test_setting_redis_host_with_no_redis_tier_active_is_silent(self):
        """When backend='tiered' default, setting redis_host shouldn't
        rebuild anything — it's stored for later use."""
        import cash
        cash._get_global_cash().backend  # ensure built
        cash.configure(redis_host="future-redis.example.com")
        # Just sets the field; no rebuild because there's no active redis tier.
        assert cash._get_global_cash().config.redis_host == "future-redis.example.com"

    def test_setting_redis_then_switching_to_redis_uses_new_host(self):
        """Pre-setting a connection field before switching backend
        type means the switch picks it up."""
        import cash
        fakeredis = pytest.importorskip("fakeredis")
        _redis = pytest.importorskip("redis")
        from cash.backends.redis_backend import RedisBackend
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            cash.configure(redis_host="staging.example.com", redis_prefix="staging:")
            cash.configure(backend="redis")
            backend = cash._get_global_cash().backend
            assert isinstance(backend, RedisBackend)
            assert backend.prefix == "staging:"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

class TestArgumentValidation:
    def test_unknown_field_raises(self):
        import cash
        with pytest.raises(ValueError, match="not a configurable field"):
            cash.configure(this_field_does_not_exist=True)

    def test_empty_call_is_noop(self):
        import cash
        c = cash._get_global_cash()
        backend_before = c.backend
        cash.configure()
        assert c.backend is backend_before


# ---------------------------------------------------------------------------
# Does NOT mutate the user's TOML
# ---------------------------------------------------------------------------

class TestNoTomlMutation:
    def test_configure_does_not_write_to_disk(self, tmp_path, monkeypatch):
        import cash
        # Point user config at a fresh dir so we'd see any write.
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        cash.reset_session()
        cash.configure(debug=True)
        # Nothing should have been written to the XDG location.
        assert not (tmp_path / "xdg").exists()
