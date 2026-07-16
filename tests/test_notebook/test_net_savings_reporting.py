"""Net cache-savings reporting in ``%cash_stats`` (CAS-143).

``%cash_stats`` used to advertise the GROSS recompute avoided while cash's own
per-cell overhead quietly ate into it — a 7.4s "saving" that really netted 4.4s
still showed 7.4s. A speed tool must not overstate the speedup. These tests pin
the honest accounting:

* gross saved, cash overhead, and NET (gross − overhead) are reported
  separately;
* NET is allowed to read *negative* — and is shown plainly, not floored or
  hidden — when a session of cheap cells paid overhead for no real saving; and
* accumulating the overhead is a float add, never a per-cell fsync (CAS-149).
"""
from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.analytics import AnalyticsManager
from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics


class _MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = _MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


def _stats_json(magics, capsys) -> dict:
    capsys.readouterr()  # drop anything buffered
    magics.cash_stats("json")
    return json.loads(capsys.readouterr().out.strip())


class TestNetPositive:
    """One expensive restore: net clearly positive and below gross."""

    def test_expensive_restore_nets_positive_and_below_gross(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        # Gross = 7s of avoided recompute, paid for with 0.4s of cash wall time
        # (restore + simulation + badge machinery). No user compute ran.
        magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 7.0, "execution_time": 0.0}],
            cell_total_time=0.4,
        )
        data = _stats_json(magics, capsys)
        assert data["total_time_saved"] == pytest.approx(7.0)
        assert data["total_overhead"] == pytest.approx(0.4)
        # NET is positive and STRICTLY below gross — overhead was subtracted.
        assert data["net_time_saved"] == pytest.approx(6.6)
        assert 0 < data["net_time_saved"] < data["total_time_saved"]

    def test_human_output_reads_positive_and_non_alarming(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        magics._update_session_stats(
            [{"status": CacheStatus.RESTORED, "saved_time": 7.0, "execution_time": 0.0}],
            cell_total_time=0.4,
        )
        capsys.readouterr()
        magics.cash_stats("")
        out = capsys.readouterr().out
        assert "Gross time saved:" in out
        assert "Cash overhead:" in out
        assert "Net time saved:" in out
        # A genuinely net-positive session must not scare the user off.
        assert "cash cost you" not in out


class TestNetNegativeOrZero:
    """Cheap cells that store nothing: gross ~0, overhead > 0 → net <= 0."""

    def test_cheap_cells_net_negative(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        # Each assignment is far below the 10ms cache floor, so nothing is ever
        # stored and nothing is ever restored → gross stays 0 while cash's
        # per-cell overhead accrues.
        for i in range(6):
            magics.cash("", f"cheap_{i} = {i} + 1")

        data = _stats_json(magics, capsys)
        assert data["total_time_saved"] == 0.0
        assert data["total_overhead"] > 0.0
        # NET reported <= 0 and NOT floored to zero: it equals -overhead.
        assert data["net_time_saved"] < 0.0
        assert data["net_time_saved"] == pytest.approx(-data["total_overhead"])

    def test_negative_net_shown_honestly_in_human_output(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        for i in range(6):
            magics.cash("", f"cheapo_{i} = {i} + 1")
        capsys.readouterr()
        magics.cash_stats("")
        out = capsys.readouterr().out
        assert "Net time saved:" in out
        # The negative is stated plainly, not hidden behind the gross number.
        assert "cash cost you" in out


class TestOverheadAccountingIsCheap:
    """The overhead accumulator is a float add, never a per-cell fsync."""

    def test_overhead_accumulation_adds_no_per_cell_io(self, magics_fixture, tmp_path):
        # Counting committed analytics rows across N cells is the deterministic
        # CAS-149-style guard: it stays 0 until a real flush, so a per-cell
        # commit sneaking back in (from the overhead accounting or anywhere in
        # the finaliser) would fail this immediately.
        magics, _shell, _backend = magics_fixture
        am = AnalyticsManager(db_path=str(tmp_path / "analytics.db"))
        magics._statement_processor.analytics_manager = am

        for i in range(10):
            magics.cash("", f"guard_{i} = {i} + 1")

        # Overhead was accumulated purely in memory ...
        assert magics._session.stats["total_overhead"] > 0.0
        # ... and NOTHING was committed to disk per cell.
        with sqlite3.connect(am.db_path) as conn:
            committed = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert committed == 0


class TestDiscriminatesGrossOverstatement:
    """FAILS on the pre-CAS-143 baseline (reports only gross, exposes no net),
    proving the suite detects the overstatement it is meant to guard against."""

    def test_stats_json_exposes_net_below_gross(self, magics_fixture, capsys):
        magics, _shell, _backend = magics_fixture
        # Drive the stats dict directly so this runs identically on the baseline,
        # whose %cash_stats never derives a net. Mirrors the real report: a 7.4s
        # gross saving with 3.0s of cash overhead → 4.4s net.
        magics._session.stats.update({
            "statements_restored": 1,
            "total_restored_time": 7.4,
            "total_time_saved": 7.4,
            "total_overhead": 3.0,
        })
        data = _stats_json(magics, capsys)
        # On the baseline there is no such key → KeyError → test fails (intended).
        assert "net_time_saved" in data
        assert data["net_time_saved"] == pytest.approx(4.4)
        assert data["net_time_saved"] < data["total_time_saved"]
