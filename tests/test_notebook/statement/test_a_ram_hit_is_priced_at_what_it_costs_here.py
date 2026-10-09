"""A statement's RAM hit is priced at what it costs in this kernel.

The fitted model priced a RAM hit of an 80 MB array at ~20 ms (a 4.5 GB/s
copy); on a busy machine the copy measured 200-800 ms against a 30-100 ms
recompute, so three cheap cells over big arrays took 2.4 s per unchanged
Run All instead of 0.3 s. The RAM tier now times its own big copies and the
store prices a hit with that speed, and a table the tier shares (no copy at
all) at a shallow copy's cost.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing, memory_backend


@pytest.fixture
def copy_speed():
    """Pins the measured copy speed; restores what was measured after."""
    saved = list(memory_backend._COPY_SPEED)

    def pin(bytes_per_second: float) -> None:
        memory_backend._COPY_SPEED[:] = [bytes_per_second]

    yield pin
    memory_backend._COPY_SPEED[:] = saved


def _store():
    from cash.backends import InMemoryBackend
    from cash.core import Cash
    from cash.notebook.statement import StatementProcessor

    shell = MagicMock()
    shell.user_ns = {}
    sp = StatementProcessor(shell, Cash(backend=InMemoryBackend(), register_magic=False))
    config = MagicMock()
    config.min_cache_savings_pct = 0.20
    config.min_cache_fixed_budget_seconds = 0.05
    config.min_execution_time_to_cache_seconds = 0.0
    sp.cash_instance.config = config
    return sp._store


def test_a_slow_copy_makes_a_cheap_statement_over_an_array_stay_uncached(copy_speed):
    copy_speed(100e6)  # 100 MB/s: an 8 MB hit costs 80 ms, the run 30 ms
    skip, reason, prediction = _store().should_skip_large_object_caching({"y": np.ones(1_000_000)}, execution_time=0.03)
    assert skip and "@cash:persist" in reason
    assert prediction["restore_seconds"] == pytest.approx(0.08, rel=0.1)


def test_a_fast_copy_keeps_it_cached(copy_speed):
    copy_speed(100e9)
    skip, _reason, prediction = _store().should_skip_large_object_caching({"y": np.ones(1_000_000)}, execution_time=0.03)
    assert not skip and prediction["restore_seconds"] < 0.001


@pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")
def test_a_table_the_tier_shares_costs_no_copy_however_slow_copying_is(copy_speed):
    copy_speed(1e6)
    df = pd.DataFrame({"a": np.ones(1_000_000), "b": np.arange(1_000_000)})
    skip, _reason, prediction = _store().should_skip_large_object_caching({"df": df}, execution_time=0.03)
    assert not skip and prediction["restore_seconds"] == memory_backend.SHARED_HIT_SECONDS


class TestHitSeconds:
    def test_an_array_costs_a_copy_at_the_measured_speed(self, copy_speed):
        copy_speed(1e9)
        assert memory_backend.hit_seconds(np.ones(10), 8_000_000) == pytest.approx(0.008)

    def test_an_array_of_objects_is_left_to_the_fitted_model(self, copy_speed):
        assert memory_backend.hit_seconds(np.array([[1], [2]], dtype=object), 100) is None

    def test_a_table_with_an_object_column_is_left_to_the_fitted_model(self):
        assert memory_backend.hit_seconds(pd.DataFrame({"s": ["a", "b"]}, dtype=object), 100) is None

    def test_anything_else_is_left_to_the_fitted_model(self):
        assert memory_backend.hit_seconds({"a": 1}, 100) is None

    def test_a_big_copy_is_timed(self, copy_speed):
        memory_backend._COPY_SPEED.clear()
        memory_backend._copy_array(np.ones(1 << 18))  # 2 MB
        assert len(memory_backend._COPY_SPEED) == 1 and memory_backend._COPY_SPEED[0] > 0

    def test_the_fastest_recent_copy_sets_the_price(self, copy_speed):
        copy_speed(1e9)
        for _ in range(3):
            memory_backend._note_copy_speed(1_000_000, 0.01)  # 100 MB/s, a copy behind other work
        assert memory_backend.copy_seconds(1_000_000_000) == pytest.approx(1.0)
        for _ in range(memory_backend._COPY_READINGS):
            memory_backend._note_copy_speed(1_000_000, 0.01)
        assert memory_backend.copy_seconds(1_000_000_000) == pytest.approx(10.0)

    def test_nothing_measured_yet_is_probed_once(self, copy_speed):
        memory_backend._COPY_SPEED.clear()
        assert memory_backend.copy_seconds(1 << 20) > 0
        assert len(memory_backend._COPY_SPEED) == 2  # the probe copies twice
