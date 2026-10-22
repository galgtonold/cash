"""A statement that reads what no lineage of its inputs follows keys what it
binds by its value, so a reader below it is not served stale.

A function a string names and the tracer follows (``globals()['g']()``, on
Python 3.12+), an iterator held where no name the statement reads holds it
(``next(m.g)``, a generator a called function keeps in its globals): each
can give a new value under the same key. The statement ran again and bound it, but the
reader below, keyed by the statement's code and inputs, was served what the
old value gave. A name asked of IPython's shell (``get_ipython().ev('k')``)
is tested in a kernel (``test_a_name_asked_of_the_shell_is_keyed_by_its_value``).
"""

from __future__ import annotations

import pytest

import cash.notebook.shadow as shadow
from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

READER = "w = [v, 'read']"

#: name: (setup, the statement's expression, a change to what it reads made
#: outside cash's view)
CASES = {
    "a string lookup the tracer follows": ("def g():\n    return 1", "globals()['g']()", "def g():\n    return 2"),
    "a generator a called function holds": (
        "gen = (i for i in range(100))\ndef f():\n    return next(gen)",
        "f()",
        "f()\nf()",
    ),
    "a generator in an attribute": (
        "class M:\n    pass\nm = M()\nm.g = (i for i in range(100))",
        "next(m.g)",
        "next(m.g)\nnext(m.g)",
    ),
    "a generator in an attribute a called function reads": (
        "class M:\n    pass\nm = M()\nm.g = (i for i in range(100))\ndef f():\n    return next(m.g)",
        "f()",
        "f()\nf()",
    ),
}


def _runs(cash_magics, cells, start=0):
    """Run *cells* from *start* on, in a notebook of all of them."""
    for cell in cells[start:]:
        run_cash_cell(cash_magics, cell, cells=cells)


@pytest.mark.parametrize("case", list(CASES))
def test_a_reader_below_is_not_served_what_the_old_value_gave(cash_magics, monkeypatch, case):
    if case.startswith("a string lookup") and shadow.SHADOW is None:
        pytest.skip("the tracer follows a string lookup on Python 3.12+ only")
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    setup, expr, change = CASES[case]
    cells = [setup, f"v = {expr}", READER]
    ns = cash_magics.shell.user_ns
    _runs(cash_magics, cells)
    first = ns["v"]
    exec(change, ns)

    _runs(cash_magics, cells, start=1)
    assert ns["v"] != first, "the change did not reach the statement"
    assert ns["w"] == [ns["v"], "read"], f"the reader was served what {first!r} gave"


def test_a_traced_lookup_and_its_reader_are_still_served_while_nothing_changed(cash_magics, monkeypatch):
    if shadow.SHADOW is None:
        pytest.skip("the tracer follows a string lookup on Python 3.12+ only")
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    setup, expr, _ = CASES["a string lookup the tracer follows"]
    cells = [setup + "\nRUNS = []", f"v = [{expr}, RUNS.append(1)][0]", "w = [v, len(RUNS.append(2) or RUNS)]"]
    ns = cash_magics.shell.user_ns
    _runs(cash_magics, cells)
    _runs(cash_magics, cells, start=1)
    assert ns["RUNS"] == [1, 2], "the statement or its reader ran again over the same values"


def test_an_attribute_that_is_not_an_iterator_keeps_the_statement_cached(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = ["class M:\n    pass\nm = M()\nm.g = [1, 2, 3]\nRUNS = []", "v = [len(m.g), RUNS.append(1)][0]"]
    _runs(cash_magics, cells)
    _runs(cash_magics, cells, start=1)
    assert cash_magics.shell.user_ns["RUNS"] == [1]
