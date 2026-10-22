"""What a refused statement binds is keyed by its value, unless the refusal
says the value is pinned by its key.

A statement is refused when its key cannot hold all it read: a thread's
data, a function a string names, a callable handed to a call over more
objects than cash looks through, an iterator. It runs every time, and the
same code over the same input lineages can bind a new value. Its outputs
then carry a digest of their value, so a stored statement below that reads
them is served only while they are the same. Keyed by code and inputs
alone, the reader was served with what the old value gave (``snap=0``
printed when ``snap`` was 3). A refusal whose value its key pins (an
import, an in-place change whose receiver's lineage moves on) says so
(``StatementRun.value_pinned``).
"""

from __future__ import annotations

import pytest

import cash.analysis.handed_callables as hc
import cash.notebook.shadow as shadow
from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

READER = "w = [v, 'read']"

#: name: (setup, the refused statement's expression, a change to what it
#: reads, made outside cash's view as a thread's would be)
CASES = {
    "thread": (
        "import threading\ndata = []\nev = threading.Event()\n"
        "def worker():\n    ev.wait()\n    data.extend([1, 2, 3])\n"
        "t = threading.Thread(target=worker, daemon=True)\nt.start()",
        "len(data)",
        "ev.set()\nt.join()",
    ),
    "string lookup": ("def g():\n    return 1", "globals()['g']()", "def g():\n    return 2"),
    "callable handed over many objects": (
        "K = 1\nclass Box:\n    def __init__(self, i):\n        self.i = i\n"
        "def g():\n    return K\nholder = [Box(i) for i in range(200)] + [g]",
        "holder[-1]()",
        "K = 2",
    ),
    "iterator": ("it = iter(range(100))", "next(it)", "_ = next(it)"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_reader_below_a_refused_statement_is_not_served_its_old_value(cash_magics, monkeypatch, case):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    # The string lookup is followed by the tracer where there is one
    # (Python 3.12+), and refused where there is none.
    monkeypatch.setattr(shadow, "SHADOW", None)
    monkeypatch.setattr(hc, "_INSTANCE_LIMIT", 50)
    setup, expr, change = CASES[case]
    cells = [setup, f"v = {expr}", READER]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = ns["v"]
    exec(change, ns)

    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["v"] != first, "the change did not reach the statement"
    assert ns["w"] == [ns["v"], "read"], f"the reader was served what {first!r} gave"


def test_a_reader_of_an_unchanged_value_is_still_served(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    setup, expr, _ = CASES["string lookup"]
    monkeypatch.setattr(shadow, "SHADOW", None)
    cells = [setup + "\nRUNS = []", f"v = {expr}", "w = [v, len(RUNS.append(1) or RUNS)]"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert ns["RUNS"] == [1], "the reader ran again over the same value"
    assert ns["w"] == [1, 1]
