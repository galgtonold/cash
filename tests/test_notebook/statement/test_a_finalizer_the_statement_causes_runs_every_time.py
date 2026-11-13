"""A finalizer the statement's own garbage runs keeps it running every time.

``churn(n)`` makes objects whose ``__del__`` (or a ``weakref.finalize``, or
a weakref's callback) changes a global, and they are freed in it: by their
last reference going, or by a collection, which ``gc.collect()`` or enough
allocation starts. Plain Python changes the global again on every run, so a
stored result would leave it behind. A collection's finalizers may as well
be garbage earlier work left (`test_a_finalizer_run_during_a_statement_is_not_its_work.py`):
the statement's next run collects that first, and what still runs inside it
is its own -- refused and said (NOTEBOOK-UNSEEN-WRITES), as any function
it runs that changes what it does not name.

Each callback here changes ``LOG`` (``COUNT``), which the statement's text
does not name. A ``weakref.finalize`` with a builtin (``LOG.append``) runs no
code a trace sees: it is told from the call to its callback.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from tests._cell_driver import run_cash_cell

NODE = "class Node:\n    def __init__(self):\n        self.me = self\n"
DEL = "    def __del__(self):\n        LOG.append(1)\n"

CASES = {
    "a cycle freed by the collections its allocations start": (
        "LOG = []\n" + NODE + DEL + "def churn(n):\n    for _ in range(20000):\n        Node()\n    return n"
    ),
    "a cycle freed by gc.collect()": (
        "import gc\nLOG = []\n" + NODE + DEL + "def churn(n):\n    Node()\n    gc.collect()\n    return n"
    ),
    "a cycle with a weakref.finalize of a builtin": (
        "import gc, weakref\nLOG = []\n"
        + NODE
        + "def churn(n):\n    o = Node()\n    weakref.finalize(o, LOG.append, 1)\n    del o\n    gc.collect()\n    return n"
    ),
    "a cycle with a weakref callback": (
        "import gc, weakref\nLOG = []\nKEEP = []\n"
        + NODE
        + "def churn(n):\n    o = Node()\n    KEEP.append(weakref.ref(o, lambda r: LOG.append(1)))\n    del o\n"
        "    gc.collect()\n    return n"
    ),
    "an object freed by its last reference": (
        "LOG = []\nclass Node:\n" + DEL + "def churn(n):\n    Node()\n    return n"
    ),
    "a cycle whose finalizer sets a global": (
        "import gc\nLOG = []\n"
        + NODE
        + "    def __del__(self):\n        global LOG\n        LOG = [*LOG, 1]\n"
        + "def churn(n):\n    Node()\n    gc.collect()\n    return n"
    ),
}

#: The callback is written in ``churn``'s text: the statement's analysis
#: reads that ``LOG`` changes, and each run sets it up again from the cell
#: above and runs (no warning: nothing is refused).
SEEN_BY_ITS_TEXT = {CASES["a cycle with a weakref callback"]}


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _status_of(cash_magics, statement):
    return [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"] if s.get("code") == statement]


#: Below Python 3.12 nothing records the code a statement runs, so a
#: finalizer its text does not name is not seen, and the statement is served.
needs_trace = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sys.monitoring needs Python 3.12: below it cash does not see the finalizer a statement runs",
)


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize(
    "setup",
    [pytest.param(case, marks=() if case in SEEN_BY_ITS_TEXT else needs_trace) for case in CASES.values()],
    ids=list(CASES),
)
def test_the_finalizer_runs_on_every_run(cash_magics, mock_shell, setup):
    """Each run of the statement runs it, and its finalizer: served, the
    change to ``LOG`` would be left out. (A run may set ``LOG`` up again
    first, from the cell above: what the finalizer changed is an input the
    cell above made.)"""
    statement = "y = churn(3)"
    cells = [setup, statement]
    run_cash_cell(cash_magics, setup, cells=cells)
    ns = mock_shell.user_ns
    runs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            run_cash_cell(cash_magics, statement, cells=cells)
            runs.append((_status_of(cash_magics, statement), len(ns["LOG"]) > 0))
    assert runs == [(["COMPUTED"], True)] * 3, f"served without the finalizer's change: {runs}"
    if setup in SEEN_BY_ITS_TEXT:
        return
    assert [w for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-UNSEEN-WRITES"], (
        "the statement ran every time without saying why"
    )
