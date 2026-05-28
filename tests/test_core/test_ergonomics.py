"""Tests for ergonomic improvements: register_magic auto-detect,
clear_all(), and verbose hit/miss logging.
"""
import logging
from unittest.mock import patch, MagicMock

import pytest

from cash import Cash


# ---------------------------------------------------------------------------
# 1. register_magic auto-detect
# ---------------------------------------------------------------------------

class TestRegisterMagicAutoDetect:
    def test_no_magic_registered_outside_ipython_by_default(self):
        """Cash() without arguments should NOT call register_magic() when
        there is no active IPython session."""
        with patch.object(Cash, 'register_magic') as mock_register:
            Cash()
            mock_register.assert_not_called()

    def test_magic_registered_when_ipython_active_by_default(self):
        """Cash() without arguments SHOULD call register_magic() when
        get_ipython() returns a live session."""
        fake_ip = MagicMock()
        with patch('cash.core.get_ipython', return_value=fake_ip):
            with patch.object(Cash, 'register_magic') as mock_register:
                Cash()
                mock_register.assert_called_once()

    def test_explicit_true_always_registers(self):
        """Cash(register_magic=True) must always call register_magic(),
        even outside IPython."""
        with patch.object(Cash, 'register_magic') as mock_register:
            Cash(register_magic=True)
            mock_register.assert_called_once()

    def test_explicit_false_never_registers(self):
        """Cash(register_magic=False) must never call register_magic()."""
        fake_ip = MagicMock()
        with patch('cash.core.get_ipython', return_value=fake_ip):
            with patch.object(Cash, 'register_magic') as mock_register:
                Cash(register_magic=False)
                mock_register.assert_not_called()


# ---------------------------------------------------------------------------
# 2. clear_all()
# ---------------------------------------------------------------------------

class TestClearAll:
    def test_clear_all_resets_stats_for_all_functions(self, tmp_path):
        """clear_all() should reset hit/miss stats for every registered function."""
        cash = Cash(cache_dir=str(tmp_path), register_magic=False)

        @cash.cache
        def add(a, b):
            return a + b

        @cash.cache
        def mul(a, b):
            return a * b

        # Populate caches
        add(1, 2); add(1, 2)   # 1 miss + 1 hit
        mul(3, 4); mul(3, 4)   # 1 miss + 1 hit

        assert add.cache_info()['hits'] == 1
        assert mul.cache_info()['hits'] == 1

        cash.clear_all()

        assert add.cache_info()['hits'] == 0
        assert add.cache_info()['misses'] == 0
        assert mul.cache_info()['hits'] == 0
        assert mul.cache_info()['misses'] == 0

    def test_clear_all_forces_recompute(self, tmp_path):
        """After clear_all(), decorated functions must recompute on the next call."""
        cash = Cash(cache_dir=str(tmp_path), register_magic=False)
        call_count = {'n': 0}

        @cash.cache
        def compute(x):
            call_count['n'] += 1
            return x * 10

        compute(5)   # first call — computes
        compute(5)   # cache hit — no compute
        assert call_count['n'] == 1

        cash.clear_all()
        compute(5)   # must recompute after clear
        assert call_count['n'] == 2

    def test_clear_all_on_empty_instance_does_not_raise(self):
        """clear_all() on a Cash instance with no decorated functions is a no-op."""
        cash = Cash(register_magic=False)
        cash.clear_all()  # should not raise


# ---------------------------------------------------------------------------
# 3. verbose hit/miss logging
# ---------------------------------------------------------------------------

class TestVerboseLogging:
    def test_miss_logged_at_info_when_verbose(self, tmp_path, caplog):
        """A cache miss should emit an INFO log when verbose=True."""
        cash = Cash(cache_dir=str(tmp_path), register_magic=False, verbose=True)

        @cash.cache
        def work(x):
            return x + 1

        with caplog.at_level(logging.INFO, logger='cash'):
            work(42)

        assert any('miss' in r.message.lower() and 'work' in r.message for r in caplog.records)

    def test_hit_logged_at_info_when_verbose(self, tmp_path, caplog):
        """A cache hit should emit an INFO log when verbose=True."""
        cash = Cash(cache_dir=str(tmp_path), register_magic=False, verbose=True)

        @cash.cache
        def work(x):
            return x + 1

        work(42)  # miss
        caplog.clear()

        with caplog.at_level(logging.INFO, logger='cash'):
            work(42)  # hit

        assert any('hit' in r.message.lower() and 'work' in r.message for r in caplog.records)

    def test_no_logging_when_not_verbose(self, tmp_path, caplog):
        """No hit/miss INFO logs should be emitted when verbose is not set."""
        cash = Cash(cache_dir=str(tmp_path), register_magic=False)

        @cash.cache
        def work(x):
            return x + 1

        with caplog.at_level(logging.INFO, logger='cash'):
            work(42)
            work(42)

        hit_miss_records = [
            r for r in caplog.records
            if 'hit' in r.message.lower() or 'miss' in r.message.lower()
        ]
        assert hit_miss_records == []
