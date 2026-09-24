"""Net cache-savings reporting in ``%cash_stats``.

``%cash_stats`` used to advertise the GROSS recompute avoided while cash's own
per-cell overhead quietly ate into it — a 7.4s "saving" that really netted 4.4s
still showed 7.4s. A speed tool must not overstate the speedup. These tests pin
the honest accounting:

* gross saved, cash overhead, and NET (gross − overhead) are reported
  separately;
* NET is allowed to read *negative* — and is shown plainly, not floored or
  hidden — when a session of cheap cells paid overhead for no real saving; and
* accumulating the overhead is a float add, never a per-cell fsync.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest

from cash.analytics import AnalyticsManager
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell


def _stats_json(magics, capsys) -> dict:
    capsys.readouterr()  # drop anything buffered
    magics.cash_stats("json")
    return json.loads(capsys.readouterr().out.strip())


class TestNetPositive:
    """One expensive restore: overhead is subtracted from gross.

    The *headline* net counts verified savings only, so a restore whose
    baseline nobody re-measured no longer prints a positive net — that
    guarantee is pinned in ``test_stale_baseline_savings.py``. What these
    tests pin is that the overhead is subtracted
    at all rather than gross being paraded as the saving.
    """

    def test_expensive_restore_nets_positive_and_below_gross(self, cash_magics, capsys):
        # Gross = 7s of avoided recompute, paid for with 0.4s of cash wall time
        # (restore + simulation + badge machinery). No user compute ran.
        cash_magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 7.0, "execution_time": 0.0}],
            cell_total_time=0.4,
        )
        data = _stats_json(cash_magics, capsys)
        assert data["total_time_saved"] == pytest.approx(7.0)
        assert data["total_overhead"] == pytest.approx(0.4)
        # The best-case net is positive and STRICTLY below gross — overhead was
        # subtracted, which is the guarantee.
        assert data["net_time_saved_upper_bound"] == pytest.approx(6.6)
        assert 0 < data["net_time_saved_upper_bound"] < data["total_time_saved"]

    def test_human_output_reads_positive_and_non_alarming(self, cash_magics, capsys):
        # Compute the statement first, so the 7.0s baseline is one THIS session
        # measured and the saving is a verified win rather than a claim.
        cash_magics._update_session_stats(
            [{"status": CacheStatus.COMPUTED, "execution_time": 7.0, "code": "m = fit()"}],
            cell_total_time=7.1,
        )
        cash_magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 7.0, "execution_time": 0.0, "code": "m = fit()"}],
            cell_total_time=0.4,
        )
        capsys.readouterr()
        cash_magics.cash_stats("")
        out = capsys.readouterr().out
        assert "Gross time saved:" in out
        assert "Cash overhead:" in out
        assert "Net time saved:" in out
        # A genuinely net-positive session must not scare the user off.
        assert "cash cost you" not in out


class TestNetNegativeOrZero:
    """Cheap cells that store nothing: gross ~0, overhead > 0 → net <= 0."""

    def test_cheap_cells_net_negative(self, cash_magics, capsys):
        # Each assignment is far below the 10ms cache floor, so nothing is ever
        # stored and nothing is ever restored → gross stays 0 while cash's
        # per-cell overhead accrues.
        for i in range(6):
            run_cash_cell(cash_magics, f"cheap_{i} = {i} + 1")

        data = _stats_json(cash_magics, capsys)
        assert data["total_time_saved"] == 0.0
        assert data["total_overhead"] > 0.0
        # NET reported <= 0 and NOT floored to zero: it equals -overhead.
        assert data["net_time_saved"] < 0.0
        assert data["net_time_saved"] == pytest.approx(-data["total_overhead"])

    def test_negative_net_shown_honestly_in_human_output(self, cash_magics, capsys):
        for i in range(6):
            run_cash_cell(cash_magics, f"cheapo_{i} = {i} + 1")
        capsys.readouterr()
        cash_magics.cash_stats("")
        out = capsys.readouterr().out
        assert "Net time saved:" in out
        # The negative is stated plainly, not hidden behind the gross number.
        assert "cash cost you" in out


class TestOverheadAccountingIsCheap:
    """The overhead accumulator is a float add, never a per-cell fsync."""

    def test_overhead_accumulation_adds_no_per_cell_io(self, cash_magics, tmp_path):
        # Counting committed analytics rows across N cells is the deterministic
        # guard against a per-cell fsync: it stays 0 until a real flush, so a per-cell
        # commit sneaking back in (from the overhead accounting or anywhere in
        # the finaliser) would fail this immediately.
        am = AnalyticsManager(db_path=str(tmp_path / "analytics.db"))
        cash_magics._statement_processor.analytics_manager = am

        for i in range(10):
            run_cash_cell(cash_magics, f"guard_{i} = {i} + 1")

        # Overhead was accumulated purely in memory ...
        assert cash_magics._session.stats["total_overhead"] > 0.0
        # ... and NOTHING was committed to disk per cell.
        with contextlib.closing(sqlite3.connect(am.db_path)) as conn:
            committed = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert committed == 0


class TestDiscriminatesGrossOverstatement:
    """FAILS on a baseline that reports only gross and exposes no net,
    proving the suite detects the overstatement it is meant to guard against."""

    def test_stats_json_exposes_net_below_gross(self, cash_magics, capsys):
        # Drive the stats dict directly so this runs identically on the baseline,
        # whose %cash_stats never derives a net. Mirrors the real report: a 7.4s
        # gross saving with 3.0s of cash overhead → 4.4s net.
        cash_magics._session.stats.update(
            {
                "statements_restored": 1,
                "total_restored_time": 7.4,
                "total_time_saved": 7.4,
                "total_verified_saved": 7.4,
                "total_overhead": 3.0,
            }
        )
        data = _stats_json(cash_magics, capsys)
        # On the baseline there is no such key → KeyError → test fails (intended).
        assert "net_time_saved" in data
        assert data["net_time_saved"] == pytest.approx(4.4)
        assert data["net_time_saved"] < data["total_time_saved"]
