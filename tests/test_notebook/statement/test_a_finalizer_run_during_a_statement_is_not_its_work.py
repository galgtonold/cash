"""What a garbage collection runs while a statement runs is not the statement's.

A collection starts wherever an allocation makes it, so the finalizers of
objects long gone run inside whatever statement is under way. The trace of
the statement recorded a ``__del__`` of the user's as a function it ran, and
one that changes a global (``LOG.append(...)``) as one whose change a stored
result would not make (NOTEBOOK-UNSEEN-WRITES): the statement was not
stored, and ran again on the next identical cell -- in a full test run, where
an earlier test had left such garbage, and not alone. The trace now leaves
out the code a collection runs (`cash._gc_context.collecting`), as the file
tracker does (`tests/test_notebook/calls/test_a_finalizer_run_during_a_call_is_not_the_calls_work.py`).
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
    return [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]


@pytest.mark.usefixtures("store_everything")
def test_a_statement_a_collection_ran_a_finalizer_in_is_stored(cash_magics, mock_shell):
    LOG.clear()
    cells = [SETUP, STATEMENT]
    run_cash_cell(cash_magics, SETUP, cells=cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Litter()
        run_cash_cell(cash_magics, STATEMENT, cells=cells)
        assert LOG == ["finalized"], "the collection never freed the garbage"
        Litter()
        run_cash_cell(cash_magics, STATEMENT, cells=cells)
    assert _statuses(cash_magics) == ["RESTORED"], "the statement ran again"
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
