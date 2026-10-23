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
    "a generator in an item a variable picks": (
        "gens = [(i for i in range(100))]\ni = 0",
        "next(gens[i])",
        "next(gens[0])",
    ),
    "a generator in an attribute's item a variable picks": (
        "class M:\n    pass\nm = M()\nm.gens = [(i for i in range(100))]\ni = 0",
        "next(m.gens[i])",
        "next(m.gens[0])",
    ),
    "a generator a method draws from its instance": (
        "class M:\n    def __init__(self):\n        self.g = (i for i in range(100))\n    def draw(self):\n        return next(self.g)\nm = M()",
        "m.draw()",
        "m.draw()\nm.draw()",
    ),
    "a generator a classmethod draws from its class": (
        "class C:\n    g = (i for i in range(100))\n    @classmethod\n    def draw(cls):\n        return next(cls.g)",
        "C.draw()",
        "C.draw()",
    ),
    "a generator a callable instance draws from": (
        "class D:\n    def __init__(self):\n        self.g = (i for i in range(100))\n    def __call__(self):\n        return next(self.g)\nd = D()",
        "d()",
        "d()",
    ),
    "a generator a partial of next holds": (
        "import functools\nf = functools.partial(next, (i for i in range(100)))",
        "f()",
        "f()",
    ),
    "a generator's bound __next__": ("nx = (i for i in range(100)).__next__", "nx()", "nx()"),
    "a generator a lambda in a dict reads": (
        "gen = (i for i in range(100))\nops = {'n': lambda: next(gen)}",
        "ops['n']()",
        "ops['n']()",
    ),
    "a generator in a default argument": ("def f(it=(i for i in range(100))):\n    return next(it)", "f()", "f()"),
    "a generator four calls deep": (
        "gen = (i for i in range(100))\ndef h():\n    return next(gen)\ndef g2():\n    return h()\n"
        "def g1():\n    return g2()\ndef f():\n    return g1()",
        "f()",
        "f()",
    ),
    "a generator a library object holds": (
        "import collections\nud = collections.UserDict(g=(i for i in range(100)))\ndef f():\n    return next(ud['g'])",
        "f()",
        "f()",
    ),
    "a generator in a pandas object column": (
        "import pandas as pd\ns = pd.Series([(i for i in range(100))], dtype=object)",
        "next(s.iloc[0])",
        "next(s.iloc[0])",
    ),
    "a list in a pandas object column a function reads": (
        "import pandas as pd\ndf = pd.DataFrame({'c': [[(i for i in range(100))]]})\ndef f():\n    return next(df['c'].iloc[0][0])",
        "f()",
        "f()",
    ),
    "a generator a function handed to a library call reads": (
        "gen = (i for i in range(100))\nf = lambda _: next(gen)",
        "list(map(f, [0]))[0]",
        "next(gen)",
    ),
    "a generator in a deque": (
        "import collections\ndq = collections.deque([(i for i in range(100))])\ndef f():\n    return next(dq[0])",
        "f()",
        "f()",
    ),
    "a generator in an OrderedDict a variable picks": (
        "import collections\nod = collections.OrderedDict(g=(i for i in range(100)))\nk = 'g'",
        "next(od[k])",
        "next(od['g'])",
    ),
    "a generator among more than 64 a variable picks": (
        "gens = [(i for i in range(100)) for _ in range(65)]\ni = 0",
        "next(gens[i])",
        "next(gens[0])",
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

    run_cash_cell(cash_magics, cells[1], cells=cells)
    ran = [str(s.get("status")) for s in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    # The statement runs: either it gives the new value, or cash rebuilt the
    # iterator above it as a run from the top would; never is it served.
    assert "RESTORED" not in ran[-1], f"the statement was served {first!r}"
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


def test_a_librarys_own_counter_does_not_refuse_the_statement(cash_magics, monkeypatch):
    """A library object keeps its own counters (matplotlib's callback ids are
    an ``itertools.count``): a statement whose function reaches one is still
    served."""
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [
        "import queue\nq = queue.Queue()\nRUNS = []\ndef f():\n    return q.qsize()",
        "v = [f(), RUNS.append(1)][0]",
    ]
    _runs(cash_magics, cells)
    _runs(cash_magics, cells, start=1)
    assert cash_magics.shell.user_ns["RUNS"] == [1]
