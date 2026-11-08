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
import warnings

import pytest

from cash._gc_context import collecting
from tests._cell_driver import run_cash_cell

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
