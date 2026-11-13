"""Garbage earlier work left does not keep a statement out of the cache.

A collection starts wherever an allocation makes it, so the finalizers of
objects long gone run inside whatever statement is under way. The trace of
the statement recorded a ``__del__`` of the user's as a function it ran, and
one that changes a global (``LOG.append(...)``) as one whose change a stored
result would not make (NOTEBOOK-UNSEEN-WRITES): the statement was not
stored, and ran again on the next identical cell -- in a full test run, where
an earlier test had left such garbage, and not alone, and every time while
some was left.

From the trace alone that garbage cannot be told from the statement's own
(`test_a_finalizer_the_statement_causes_runs_every_time.py`). Code that ran
only inside a collection (`cash._gc_context.collecting`) keeps the statement
out of the cache this once, without a warning, and its next run starts with
a collection of its own: what earlier work left is freed before it, and the
statement is stored.
"""

from __future__ import annotations

import gc
import sys
import warnings

import pytest

from cash._gc_context import collecting
from cash.notebook.statement.collect_first import CollectFirstStore
from tests._cell_driver import run_cash_cell

needs_trace = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sys.monitoring needs Python 3.12: below it cash does not see the code a statement runs",
)

LOG: list[str] = []


class Litter:
    """Garbage of the user's: a cycle, freed by a collection only."""

    def __init__(self) -> None:
        self.me = self

    def __del__(self) -> None:
        LOG.append("finalized")


SETUP = "import gc\ndef build(n):\n    gc.collect()  # where an allocation might have started one\n    return list(range(n))"
STATEMENT = "y = build(3)"


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _statuses(cash_magics):
    return [(str(s["status"]), s.get("uncacheable_reasons")) for s in cash_magics._last_cell_metrics["statements"]]


@pytest.fixture
def collected_by_hand():
    """No collection but the ones the test asks for: one started by an
    allocation before the statement would free the garbage outside it."""
    was = gc.isenabled()
    gc.disable()
    yield
    if was:
        gc.enable()


@needs_trace
@pytest.mark.usefixtures("store_everything", "collected_by_hand")
def test_garbage_left_before_a_statement_keeps_it_out_once(cash_magics, mock_shell):
    LOG.clear()
    cells = [SETUP, STATEMENT]
    run_cash_cell(cash_magics, SETUP, cells=cells)
    seen = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            Litter()  # left by earlier work, before each run
            run_cash_cell(cash_magics, STATEMENT, cells=cells)
            seen.append(_statuses(cash_magics))
    gc.collect()  # the last run was served: its garbage is still there
    assert LOG == ["finalized"] * 3, "the garbage was not freed"
    first, second, third = seen
    assert first[0][0] == "COMPUTED", first
    assert second[0][0] == "COMPUTED", "the run with earlier work's garbage in it was stored"
    assert third[0][0] == "RESTORED", third
    assert mock_shell.user_ns["y"] == [0, 1, 2]
    assert not [w for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-UNSEEN-WRITES"]


@needs_trace
@pytest.mark.usefixtures("store_everything", "collected_by_hand")
def test_the_first_run_after_a_restart_collects_first_too(cash_magics, tmp_path):
    """Restart and run all: each statement runs once a kernel. Known in the
    kernel alone, the flag was gone before the run it was for, and the
    statement was refused in every kernel."""
    processor = cash_magics._statement_processor
    # As over a cache folder (the in-memory backend has none).
    processor._collect_first_store = CollectFirstStore(str(tmp_path))
    cells = [SETUP, STATEMENT]
    run_cash_cell(cash_magics, SETUP, cells=cells)
    Litter()
    run_cash_cell(cash_magics, STATEMENT, cells=cells)
    assert cash_magics.tracking_state.collect_first, "the run was not refused for the garbage"
    # A restart: nothing of the session, the flag as the disk keeps it.
    cash_magics.tracking_state.reset_session_state()
    processor._collect_first_store = CollectFirstStore(str(tmp_path))
    run_cash_cell(cash_magics, SETUP, cells=cells)
    seen = []
    for _ in range(2):
        Litter()
        run_cash_cell(cash_magics, STATEMENT, cells=cells)
        seen.append(_statuses(cash_magics))
    gc.collect()
    assert seen[0][0][0] == "COMPUTED", seen
    assert seen[1][0][0] == "RESTORED", "the first run after the restart was refused again"


def test_only_a_collection_reads_as_collecting():
    seen: list[bool] = []

    class Probe:
        def __init__(self) -> None:
            self.me = self

        def __del__(self) -> None:
            seen.append(collecting())

    assert not collecting()
    Probe()
    gc.collect()
    assert seen == [True]
    assert not collecting()
