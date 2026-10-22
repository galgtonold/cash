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
import cash.value_hash as value_hash
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
    lineage = cash_magics.tracking_state.variable_lineage
    before = lineage["v"]
    run_cash_cell(cash_magics, cells[1], cells=cells)
    # The value's digest is the same run to run: the reader is not only
    # served through the same-value cutoff.
    assert lineage["v"] == before, "the same value was keyed apart"
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert ns["RUNS"] == [1], "the reader ran again over the same value"
    assert ns["w"] == [1, 1]


#: name: (setup, the statement's expression, a change to what it reads made
#: outside cash's view, whether the tracer is off). Each is refused for a
#: reason that alone would pin what it binds -- an in-place change, a file
#: write, an alias -- or that does not pin it although it once did: a call
#: to a process. What else it reads keys its outputs by value all the same.
PINNED_ELSEWHERE = {
    "an append beside a string lookup": (
        "def g():\n    return 1\nL = []",
        "(L.append(0), globals()['g']())[1]",
        "def g():\n    return 2",
        True,
    ),
    "an append beside a string lookup, traced": (
        "def g():\n    return 1\nL = []",
        "(L.append(0), globals()['g']())[1]",
        "def g():\n    return 2",
        False,
    ),
    "a setdefault beside a string lookup": (
        "def g():\n    return 1\nD = {}",
        "D.setdefault('k', 0) + globals()['g']()",
        "def g():\n    return 2",
        True,
    ),
    "an append beside an iterator": (
        "it = iter(range(100))\nL = []",
        "(L.append(0), next(it))[1]",
        "_ = next(it)",
        False,
    ),
    "a process's output": (
        "import os, subprocess, sys, tempfile\np = os.path.join(tempfile.mkdtemp(), 'f.txt')\n"
        "open(p, 'w').write('1')\nCAT = [sys.executable, '-c', 'import sys; print(open(sys.argv[1]).read())']",
        "subprocess.check_output(CAT + [p]).decode().strip()",
        "open(p, 'w').write('2')",
        False,
    ),
    "a file write beside a string lookup": (
        "import os, tempfile\np = os.path.join(tempfile.mkdtemp(), 'f.txt')\ndef g():\n    return 1",
        "(open(p, 'w').write('x'), globals()['g']())[1]",
        "def g():\n    return 2",
        True,
    ),
    "a name read through the namespace": ("k = 1\nNS = globals()", "NS['k']", "k = 2", False),
}


@pytest.mark.parametrize("case", list(PINNED_ELSEWHERE))
def test_a_reason_that_pins_does_not_hide_one_that_does_not(cash_magics, monkeypatch, case):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    setup, expr, change, tracer_off = PINNED_ELSEWHERE[case]
    if tracer_off:
        monkeypatch.setattr(shadow, "SHADOW", None)
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


def test_a_value_too_costly_to_digest_gets_a_lineage_of_its_own(cash_magics, monkeypatch):
    """The value digest has a change check's budget: past it, the value is
    keyed apart each run and its readers recompute, rather than every run
    pay seconds for a value of many objects."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    monkeypatch.setattr(shadow, "SHADOW", None)
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 0.0)
    setup = (
        "class Box:\n    def __init__(self, i):\n        self.i = i\ndef g():\n    return [Box(i) for i in range(300)]"
    )
    cells = [setup, "v = globals()['g']()"]
    lineage = cash_magics.tracking_state.variable_lineage
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = lineage["v"]
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert lineage["v"] != before


#: name: (setup, the statement's expression, a change to what the code it
#: runs reads, which its key does not hold)
TRACE_REFUSED = {
    "the code it ran changed": (
        "def inner(x):\n    return x * 2",
        "inner(3)",
        "inner.__code__ = (lambda x: x * 10).__code__",
    ),
    "a function a string names changed": (
        "cfg = 1\nname = 'make'\ndef make(x):\n    return x + cfg * 1",
        "globals()[name](3)",
        "def make(x):\n    return x + cfg * 5",
    ),
}


@pytest.mark.parametrize("case", list(TRACE_REFUSED))
def test_a_hit_its_trace_refuses_gives_a_reader_below_the_new_value(cash_magics, monkeypatch, case):
    """An entry found under its key and refused by its trace runs again with
    the same key: the reader below, keyed on the same lineage, was served
    what the old value gave (``u = t + 1`` served 7 where 31 is right)."""
    if shadow.SHADOW is None:
        pytest.skip("traces are recorded on Python 3.12+ only")
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    setup, expr, change = TRACE_REFUSED[case]
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
