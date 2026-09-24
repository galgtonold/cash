"""After a restart, the net is a measured number, not a range around zero.

"%cash_stats after a restart: 'Net time saved:
at least -10.3s, at best 1.3min'. A range whose lower bound is just
-overhead (nothing 'verified' in a fresh kernel). My pair measured 80.2s
saved, so the upper bound was the right one; the lower bound tells me
nothing. For a team lead the range reads as 'cash may have cost you time',
which the measurement contradicts."

A saving counted only where THIS kernel had recomputed the same statement.
The cost of a computation is now kept across kernels
(``compute_baselines``), so a restore after a restart is credited against a
real measurement -- labelled measured rather than verified, because the
measurement is from an earlier run on this machine.
"""

from __future__ import annotations

import json

import pytest

from cash.backends.file_backend import FileBackend
from cash.core import Cash
from cash.notebook import compute_baselines
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics
from tests.conftest import MockShell


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    """A fresh kernel against a cache directory that outlives it."""
    monkeypatch.setattr(compute_baselines._STORES, "_stores", {})

    def _start():
        # Each kernel has its own shell; the cache directory is what they share.
        cash = Cash(backend=FileBackend(cache_dir=str(tmp_path)), register_magic=False)
        return CashMagics(MockShell(), cash)

    return _start


def _stats_json(magics, capsys) -> dict:
    capsys.readouterr()
    magics.cash_stats("json")
    return json.loads(capsys.readouterr().out.strip())


def _text(magics, capsys) -> str:
    capsys.readouterr()
    magics.cash_stats("")
    return capsys.readouterr().out


def test_a_restore_in_a_later_kernel_is_credited_from_the_measurement(kernel, capsys):
    monday = kernel()
    monday._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.0, "code": "m = fit(x)"}],
        cell_total_time=9.1,
    )
    monday._session.baselines.flush()

    tuesday = kernel()  # Restart & Run All
    tuesday._update_session_stats(
        [{"status": CacheStatus.RESTORED, "saved_time": 9.0, "execution_time": 0.0, "code": "m = fit(x)"}],
        cell_total_time=0.4,
    )
    data = _stats_json(tuesday, capsys)
    assert data["total_measured_saved"] == pytest.approx(9.0)
    assert data["net_time_saved"] == pytest.approx(8.6)

    out = _text(tuesday, capsys)
    assert "at least" not in out, out
    assert "cash cost you" not in out, out


def test_the_cheaper_measurement_is_the_one_credited(kernel, capsys):
    """Two runs, 9 s then 4 s: the credit is 4 s. A baseline recorded on a
    cold first run must not be paid out forever."""
    monday = kernel()
    monday._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.0, "code": "m = fit(x)"}], cell_total_time=9.1
    )
    monday._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 4.0, "code": "m = fit(x)"}], cell_total_time=4.1
    )
    monday._session.baselines.flush()

    tuesday = kernel()
    tuesday._update_session_stats(
        [{"status": CacheStatus.RESTORED, "saved_time": 9.0, "execution_time": 0.0, "code": "m = fit(x)"}],
        cell_total_time=0.2,
    )
    data = _stats_json(tuesday, capsys)
    assert data["total_measured_saved"] == pytest.approx(4.0)


def test_a_statement_this_machine_never_ran_is_still_a_range(kernel, capsys):
    """Control: a cache built elsewhere vouches for nothing, and the report
    says so rather than inventing a number."""
    fresh = kernel()
    fresh._update_session_stats(
        [{"status": CacheStatus.RESTORED, "saved_time": 30.0, "execution_time": 0.0, "code": "m = fit(x)"}],
        cell_total_time=0.5,
    )
    data = _stats_json(fresh, capsys)
    assert data["total_measured_saved"] == pytest.approx(0.0)
    assert "at least" in _text(fresh, capsys)


def test_reset_forgets_the_measurements_it_claims_to_forget(kernel, capsys):
    monday = kernel()
    monday._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.0, "code": "m = fit(x)"}], cell_total_time=9.1
    )
    monday._session.baselines.flush()
    monday.cash_stats("reset")

    tuesday = kernel()
    tuesday._update_session_stats(
        [{"status": CacheStatus.RESTORED, "saved_time": 9.0, "execution_time": 0.0, "code": "m = fit(x)"}],
        cell_total_time=0.4,
    )
    assert _stats_json(tuesday, capsys)["total_measured_saved"] == pytest.approx(0.0)


def test_a_cached_call_is_credited_across_kernels_too(kernel, capsys):
    """The shape people actually measure: the work sits behind a call, and
    the Restart & Run All restores it."""
    monday = kernel()
    monday._update_session_stats(
        [
            {
                "status": CacheStatus.COMPUTED,
                "execution_time": 20.0,
                "code": "runs = qc(adata)",
                "decorator_calls": [{"cache_key": "qc:1", "cache_hit": False, "execution_time": 20.0}],
            }
        ],
        cell_total_time=20.2,
    )
    monday._session.baselines.flush()

    tuesday = kernel()
    tuesday._update_session_stats(
        [
            {
                "status": CacheStatus.COMPUTED,
                "execution_time": 0.3,
                "code": "runs = qc(adata)",
                "decorator_calls": [{"cache_key": "qc:1", "cache_hit": True, "time_saved": 20.0}],
            }
        ],
        cell_total_time=0.5,
    )
    assert _stats_json(tuesday, capsys)["total_measured_saved"] == pytest.approx(20.0)
