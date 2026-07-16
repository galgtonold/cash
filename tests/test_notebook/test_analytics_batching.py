"""Analytics batching guards (CAS-149).

The per-cell finaliser used to call ``analytics_manager.flush()`` on *every*
cell, force-draining a buffer explicitly designed to batch 50 events. Each
flush is a SQLite ``connect + commit`` (an fsync), which dominated per-cell
wall time (~12 ms/cell measured). These tests pin the fix:

* the per-cell path must NOT commit-per-cell — events stay buffered until the
  50-event threshold, a stats query, or a clean-shutdown drain; and
* buffered events are NOT lost on the happy path — a normal ``flush()`` (and
  the ``atexit`` shutdown drain) persists them.

The magics-level test (``test_running_cells_does_not_commit_per_cell``) is the
stable regression guard: it counts committed rows across N cells rather than
timing anything, so it fails deterministically on the old fsync-per-cell code
(N committed rows) and passes on the batched code (0 committed rows until a
flush).
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.analytics import AnalyticsManager, _flush_live_managers_atexit
from cash.backends import InMemoryBackend
from cash.core import Cash
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


def _committed_row_count(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]


class TestPerCellDoesNotFsync:
    """The core CAS-149 guard: no commit-per-cell."""

    def test_running_cells_does_not_commit_per_cell(self, magics_fixture, tmp_path):
        magics, _shell, _backend = magics_fixture

        # Point the processor at an isolated DB so we can count committed rows.
        am = AnalyticsManager(db_path=str(tmp_path / "analytics.db"))
        magics._statement_processor.analytics_manager = am

        n_cells = 10  # well under the 50-event flush threshold
        for i in range(n_cells):
            magics.cash("", f"batch_var_{i} = {i} + 1")

        # Each first-run compute records a MISS event, so the buffer must have
        # grown — proving events ARE being recorded, not silently dropped.
        buffered = len(am._event_buffer)
        assert buffered >= 1, "expected the cells to record analytics events"

        # The regression guard: NOTHING was committed to disk per cell. On the
        # old fsync-per-cell code this would be >= n_cells.
        assert _committed_row_count(am.db_path) == 0

        # Happy path: a normal flush persists everything that was buffered.
        am.flush()
        assert _committed_row_count(am.db_path) == buffered
        assert am._event_buffer == []


class TestBufferPolicy:
    """AnalyticsManager buffer/flush/shutdown contract in isolation."""

    def test_events_buffered_below_threshold_not_persisted(self, tmp_path):
        am = AnalyticsManager(db_path=str(tmp_path / "a.db"))
        for _ in range(5):  # < 50
            am.record_event("MISS", 0.01)
        assert len(am._event_buffer) == 5
        # Not yet committed to disk.
        assert _committed_row_count(am.db_path) == 0

    def test_threshold_triggers_automatic_flush(self, tmp_path):
        am = AnalyticsManager(db_path=str(tmp_path / "a.db"))
        for _ in range(am._flush_threshold):  # exactly 50
            am.record_event("HIT", 0.01, saved_time=0.1)
        # Reaching the threshold drains the buffer to disk.
        assert am._event_buffer == []
        assert _committed_row_count(am.db_path) == am._flush_threshold

    def test_explicit_flush_persists_buffered_events(self, tmp_path):
        am = AnalyticsManager(db_path=str(tmp_path / "a.db"))
        am.record_event("MISS", 0.02)
        am.record_event("HIT", 0.01, saved_time=1.0)
        am.flush()
        assert _committed_row_count(am.db_path) == 2

    def test_atexit_drain_persists_buffered_events(self, tmp_path):
        """Clean-shutdown drain: no data loss on the happy path."""
        am = AnalyticsManager(db_path=str(tmp_path / "a.db"))
        am.record_event("MISS", 0.02)
        am.record_event("MISS", 0.03)
        assert _committed_row_count(am.db_path) == 0  # still buffered

        # Simulate the atexit hook firing on a clean interpreter/kernel exit.
        _flush_live_managers_atexit()

        assert _committed_row_count(am.db_path) == 2
        assert am._event_buffer == []
