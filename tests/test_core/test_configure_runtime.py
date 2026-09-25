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

    def test_debug_and_verbose_are_the_config_itself(self):
        """One copy of each flag: what the instance reports is its config,
        whichever side changed it."""
        import cash

        c = cash._get_global_cash()
        cash.configure(verbose=True)
        assert c.verbose is True
        c.config.verbose = False
        assert c.verbose is False
        c.debug = True
        assert c.config.debug is True

    def test_configure_min_cache_savings_pct(self):
        import cash

        cash.configure(min_cache_savings_pct=0.10)
        assert cash._get_global_cash().config.min_cache_savings_pct == 0.10

    def test_hot_field_does_not_rebuild_backend(self):
        import cash

        # Touch backend to ensure it's built and capture identity.
        backend_before = cash._get_global_cash().backend
        cash.configure(debug=True, min_cache_savings_pct=0.5)
        backend_after = cash._get_global_cash().backend
        assert backend_before is backend_after, "hot field reconfigure should not rebuild the backend"


class TestPersistencePolicyFields:
    """``min_cache_savings_pct`` is read into the running backend's
    persistence policy when it is built. Changing it has to reach that
    policy, or the change silently does nothing."""

    @staticmethod
    def _persists(backend, key):
        metadata = {"execution_time": 0.5, "size": 10}
        backend.set(key, 1, metadata)
        return metadata["storage"] != ["RAM"]

    def test_min_cache_savings_pct_reaches_the_running_backend(self, tmp_path):
        import cash

        cash.configure(cache_dir=str(tmp_path / ".cash"))
        backend = cash._get_global_cash().backend
        assert self._persists(backend, "before"), "a 0.5 s result should persist by default"
        cash.configure(min_cache_savings_pct=0.999)
        assert cash._get_global_cash().backend is backend
        assert not self._persists(backend, "after")


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


# ---------------------------------------------------------------------------
# configure() takes a value the way Cash(...) does
# ---------------------------------------------------------------------------


class TestSameAsTheConstructor:
    """A setting means the same thing whether it is given to ``Cash(...)`` or
    to ``configure()``. The runtime path had its own checks: tier tables were
    stored unconverted (and broke every later call), ``~`` was not expanded,
    a backend the caller built was replaced, and ``debug=False`` did not stop
    the output ``debug=True`` started."""

    def test_tier_tables_are_built_into_tiers(self, tmp_path):
        from cash import Cash
        from cash.backends.memory_backend import InMemoryBackend
        from cash.config import TierConfig

        c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)
        c.backend  # built
        c.reconfigure(tiers=[{"type": "memory"}])
        assert c.config.tiers == [TierConfig(type="memory")]
        assert [type(b) for b in c.backend.backends] == [InMemoryBackend]
        c.reconfigure(max_cache_size=100)  # and the next call still works
        assert c.config.max_cache_size == 100

    def test_a_bad_value_changes_nothing(self, tmp_path):
        from cash import Cash

        c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)
        backend = c.backend
        before = c.config.max_cache_size
        with pytest.raises(ValueError, match="floppy"):
            c.reconfigure(max_cache_size=100, tiers=[{"type": "floppy"}])
        assert c.config.max_cache_size == before
        assert c.config.tiers == []
        assert c.backend is backend

    def test_a_backend_that_cannot_be_built_changes_nothing(self, tmp_path):
        """``backend="s3"`` without ``s3_bucket`` passes the value checks and
        fails only when the S3 tier is built. It was built after the new
        values were applied and the old backend shut down, so the raise left
        ``backend="s3"`` in the config and a shut-down backend in place:
        every later store failed for the rest of the process."""
        import warnings

        from cash import Cash

        c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)
        f = c.cache(assume_safe=True)(lambda x: list(range(x)))
        f(10)
        backend = c.backend
        with pytest.raises(ValueError, match="s3_bucket"):
            c.reconfigure(backend="s3")
        assert c.config.backend == "tiered"
        assert c.backend is backend
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert f(20) == list(range(20))
            backend.backends[-1]._writes.wait_all()
        assert [getattr(w.message, "code", None) for w in caught] == []
        assert f.cache_info()["misses"] == 2
        assert f(20) == list(range(20)) and f.cache_info()["hits"] == 1

    def test_a_home_relative_cache_dir_is_expanded(self, tmp_path, monkeypatch):
        from cash import Cash

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)
        c.backend
        c.reconfigure(cache_dir="~/b")
        expected = str(tmp_path / "home" / "b")
        assert c.config.cache_dir == expected
        assert c.backend.backends[-1].cache_dir == expected
        assert not (tmp_path / "~").exists()

    @pytest.mark.parametrize("from_a_file", [False, True])
    def test_a_home_relative_cache_dir_has_one_separator_style_on_windows(self, tmp_path, monkeypatch, from_a_file):
        # Windows' expanduser("~/b") is `C:\Users\me/b`: the home part in
        # backslashes, the rest as typed. Simulated with ntpath on any OS.
        import ntpath
        import os

        from cash import config as cash_config

        monkeypatch.setenv("USERPROFILE", r"C:\Users\me")
        for name in ("expanduser", "normpath", "isabs"):
            monkeypatch.setattr(os.path, name, getattr(ntpath, name))
        origin = tmp_path if from_a_file else cash_config._CALLER_RELATIVE
        assert cash_config._anchor_cache_dir("~/b", origin) == r"C:\Users\me\b"

    def test_a_backend_the_caller_built_is_never_replaced(self, tmp_path):
        from cash import Cash
        from cash.backends.sqlite_backend import SQLiteBackend

        mine = SQLiteBackend(str(tmp_path / "mine.db"))
        c = Cash(backend=mine, register_magic=False)
        before = c.config.max_cache_size
        with pytest.raises(ValueError, match="given its backend"):
            c.reconfigure(max_cache_size=10**6)
        assert c.backend is mine
        assert c.config.max_cache_size == before
        c.reconfigure(min_cache_savings_pct=0.5)  # not a tier setting: fine
        assert c.backend is mine
        mine.shutdown()

    def test_debug_off_stops_the_debug_output(self, tmp_path, monkeypatch, request):
        import logging

        from cash import Cash, _log

        cash_logger = logging.getLogger("cash")
        # From cash's logger as a fresh process has it, whatever earlier tests
        # in this worker set.
        monkeypatch.setattr(_log, "_LEVEL_SET", None)
        level = cash_logger.level
        cash_logger.setLevel(logging.NOTSET)
        request.addfinalizer(lambda: cash_logger.setLevel(level))
        c = Cash(cache_dir=str(tmp_path / "c"), register_magic=False)
        c.reconfigure(debug=True)
        assert cash_logger.isEnabledFor(logging.DEBUG)
        c.reconfigure(debug=False)
        assert not cash_logger.isEnabledFor(logging.DEBUG)
        assert not cash_logger.isEnabledFor(logging.INFO)
        c.reconfigure(verbose=True)
        assert cash_logger.isEnabledFor(logging.INFO) and not cash_logger.isEnabledFor(logging.DEBUG)
        c.reconfigure(verbose=False)
        assert not cash_logger.isEnabledFor(logging.INFO)
