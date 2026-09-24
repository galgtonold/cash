"""The measured cost of a computation, kept across kernels.

After a Restart & Run All, `%cash_stats` said
"Net time saved: at least -10.3s, at best 1.3min". A saving counted only
when THIS kernel had recomputed the same statement, and a fresh kernel has
recomputed nothing -- so the floor was exactly minus the overhead, in the one
measurement everyone takes. "The lower bound tells me nothing. For a team
lead the range reads as 'cash may have cost you time', which the measurement
contradicts."

The store keeps the MINIMUM cost ever measured on this machine, so a restore
after a restart is credited against a real measurement. The minimum only ever
ratchets the credit down: it cannot resurrect the stale-high baseline that the
verified net exists to guard against.
"""

from __future__ import annotations

import json

import pytest

from cash.notebook import compute_baselines


@pytest.fixture(autouse=True)
def _fresh_stores():
    compute_baselines._reset_stores_for_tests()
    yield
    compute_baselines._reset_stores_for_tests()


def test_the_minimum_measurement_wins(tmp_path):
    store = compute_baselines.get_store(str(tmp_path))
    store.record("m = fit(x)", 9.0)
    store.record("m = fit(x)", 4.0)
    store.record("m = fit(x)", 7.0)
    assert store.get("m = fit(x)") == pytest.approx(4.0)


def test_a_measurement_outlives_the_kernel(tmp_path):
    compute_baselines.get_store(str(tmp_path)).record("m = fit(x)", 12.0)
    compute_baselines.get_store(str(tmp_path)).flush()

    compute_baselines._reset_stores_for_tests()  # a new kernel
    assert compute_baselines.get_store(str(tmp_path)).get("m = fit(x)") == pytest.approx(12.0)


def test_an_unknown_computation_has_no_baseline(tmp_path):
    assert compute_baselines.get_store(str(tmp_path)).get("never = seen()") is None


def test_without_a_cache_dir_it_is_session_scoped(tmp_path):
    """An in-memory backend has nowhere to persist, and no restart to survive."""
    store = compute_baselines.get_store(None)
    store.record("m = fit(x)", 3.0)
    assert store.get("m = fit(x)") == pytest.approx(3.0)
    store.flush()  # must not raise


def test_a_corrupt_store_reads_as_empty(tmp_path):
    (tmp_path / compute_baselines._STORE_FILENAME).write_text("{not json", encoding="utf-8")
    assert compute_baselines.get_store(str(tmp_path)).get("m = fit(x)") is None


def test_a_store_from_a_future_version_is_ignored(tmp_path):
    (tmp_path / compute_baselines._STORE_FILENAME).write_text(
        json.dumps({"version": 99, "baselines": {"x": 5.0}}), encoding="utf-8"
    )
    assert compute_baselines.get_store(str(tmp_path)).get("m = fit(x)") is None


def test_the_cheapest_baselines_are_dropped_at_the_cap(tmp_path, monkeypatch):
    """The store stays small. What it keeps is what is worth crediting."""
    monkeypatch.setattr(compute_baselines, "_MAX_ENTRIES", 10)
    store = compute_baselines.get_store(str(tmp_path))
    for i in range(40):
        store.record(f"s{i}", float(i))
    assert store.get("s39") == pytest.approx(39.0)
    assert store.get("s0") is None
    assert len(store._items) <= 10


def test_clear_drops_the_file_too(tmp_path):
    store = compute_baselines.get_store(str(tmp_path))
    store.record("m = fit(x)", 8.0)
    store.flush()
    assert (tmp_path / compute_baselines._STORE_FILENAME).exists()

    store.clear()
    assert store.get("m = fit(x)") is None
    assert not (tmp_path / compute_baselines._STORE_FILENAME).exists()


def test_cash_does_not_track_its_own_store_as_a_dependency(tmp_path):
    """The file lives in the cache dir; reading it must never become a user
    variable's file dependency (the rule ``is_cash_file`` exists for)."""
    from cash.backends.cache_dir import is_cash_file

    assert is_cash_file(compute_baselines._STORE_FILENAME)
