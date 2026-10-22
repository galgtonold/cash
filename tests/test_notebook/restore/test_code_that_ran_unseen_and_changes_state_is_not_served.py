"""A function of the user's that ran in a statement on a path cash does not
read from its text, and changes something the statement does not name, keeps
the statement from being stored (on Python 3.12+ from the trace of the run;
without one, from what its objects could run).

``Vec.__add__`` appends to ``calls``. Run by ``s + 1`` on an object-dtype
column of ``Vec``, by ``d[k][0] + 1``, by a closure, a classmethod, a
comprehension, ``sum(lst, Vec())`` or ``partial(operator.add, v)``: each was
served after a restart with ``calls == []``, where a run appends.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from cash.decorator import code_trace
from cash.notebook import shadow
from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

#: Without a trace (Python 3.10 and 3.11, ``CASH_TRACE=0``) the objects are
#: read instead (`writers_without_a_trace`): the same tests hold.
NEEDS_THE_TRACE = pytest.mark.skipif(not code_trace.on(), reason="reads the trace of the run (3.12+)")

SLEEP = f"(time.sleep({F}), 0)[1]"
HEAD = """import time, types, functools, operator
calls = []
class Vec:
    def __add__(self, x):
        calls.append(1)
        return 0
"""
NAMES = (
    "GEN",
    "H",
    "STATE",
    "St",
    "heapq",
    "Two",
    "calls",
    "Vec",
    "v",
    "d",
    "k",
    "i",
    "lst",
    "np",
    "pd",
    "arr",
    "s",
    "df",
    "K",
    "g",
    "mk",
    "p",
    "w",
    "time",
)

CASES = {
    "numpy_elementwise": (
        "import numpy as np\narr = np.empty(1, dtype=object)\narr[0] = Vec()",
        f"w = (arr + 1, {SLEEP})",
    ),
    "pandas_elementwise": (
        "import pandas as pd\ns = pd.Series([Vec()], dtype=object)",
        f"w = (s + 1, {SLEEP})",
    ),
    "two_steps_down": ("d = {'a': [Vec()]}\nk = 'a'", f"w = (d[k][0] + 1, {SLEEP})"),
    "closure": (
        "v = Vec()\ndef mk():\n    x = v\n    def g(a):\n        return x + 1\n    return g\ng = mk()",
        f"w = g({SLEEP})",
    ),
    "classmethod": (
        "v = Vec()\nclass K:\n    @classmethod\n    def m(cls, a):\n        return v + 1",
        f"w = K.m({SLEEP})",
    ),
    "comprehension": ("lst = [Vec()]", f"w = ([x + 1 for x in lst], {SLEEP})"),
    "sum_from_an_object": ("lst = [Vec()]", f"w = (sum(lst, Vec()), {SLEEP})"),
    "partial_of_an_operator": ("v = Vec()\np = functools.partial(operator.add, v)", f"w = p({SLEEP})"),
}


def _restart(magics):
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _codes(magics, cell):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(magics, cell)
    return [c for w in caught if (c := getattr(w.message, "code", None))]


@pytest.mark.parametrize("case", list(CASES))
def test_the_statement_runs_each_time_and_says_why(cash_magics, case):
    setup, statement = CASES[case]
    said = []
    for _ in range(2):
        run_cash_cell(cash_magics, HEAD + setup)
        said += _codes(cash_magics, statement)
        assert cash_magics.shell.user_ns["calls"] == [1]
        _restart(cash_magics)
    assert "NOTEBOOK-UNSEEN-WRITES" in said


def _status(magics):
    return [str(s["status"]) for s in magics.cash_status("dict")["last_cell"]["statements"]]


@NEEDS_THE_TRACE
def test_an_entry_stored_before_is_not_served(cash_magics, monkeypatch):
    """Stored while the check was not there, the entry's trace names
    ``Vec.__add__``: the hit is refused."""
    setup, statement = CASES["pandas_elementwise"]
    monkeypatch.setattr(processor, "unseen_writers", lambda *a: [])
    monkeypatch.setattr(shadow.Shadow, "unseen_writes", lambda *a: False)
    run_cash_cell(cash_magics, HEAD + setup)
    run_cash_cell(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]  # stored, without the check
    monkeypatch.undo()
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["COMPUTED"]
    assert cash_magics.shell.user_ns["calls"] == [1, 1]


def test_a_change_the_statement_names_is_stored(cash_magics):
    """``calls`` is an output of the statement (``calls += ...`` names it),
    so a served entry gives it back: stored and served."""
    setup, _ = CASES["pandas_elementwise"]
    statement = f"calls += [(s + 1).iloc[0], {SLEEP}]"
    run_cash_cell(cash_magics, HEAD + setup)
    assert not _codes(cash_magics, statement)
    run_cash_cell(cash_magics, "calls = []")
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]
    assert cash_magics.shell.user_ns["calls"] == [1, 0, 0]


def test_a_special_method_changing_nothing_outside_is_stored(cash_magics):
    pure = "import time\nimport pandas as pd\nclass Vec:\n    def __add__(self, x):\n        return 0\ns = pd.Series([Vec()], dtype=object)"
    statement = f"w = (s + 1, {SLEEP})"
    run_cash_cell(cash_magics, pure)
    assert not _codes(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]


def test_assume_safe_stores_it(cash_magics):
    setup, statement = CASES["pandas_elementwise"]
    run_cash_cell(cash_magics, HEAD + setup)
    assert not _codes(cash_magics, statement + "  # @cash:assume-safe")
    run_cash_cell(cash_magics, statement + "  # @cash:assume-safe")
    assert _status(cash_magics) == ["RESTORED"]


def test_a_generator_expression_is_read_with_the_function_it_is_made_in(cash_magics):
    """``sum(i * i for i in r)`` inside a function that ran: its text is
    the function's, which changes nothing outside."""
    run_cash_cell(cash_magics, "import time\ndef total(n):\n    return sum(i * i for i in range(n))")
    statement = f"w = (total(5), {SLEEP})"
    assert not _codes(cash_magics, statement)
    run_cash_cell(cash_magics, statement)
    assert _status(cash_magics) == ["RESTORED"]


@NEEDS_THE_TRACE
def test_a_generator_expression_run_on_its_own_is_read_from_its_line(tmp_path, monkeypatch):
    """Made in a function that has returned (``mk()`` hands it out), it
    runs without that function: read from its own line."""
    import importlib
    import types

    from cash.notebook.statement.unseen_writes import own_writes

    (tmp_path / "gx_made.py").write_text(
        "calls = []\ndef mk():\n    a = sum(i for i in range(3))\n    return a, (calls.append(1) for _ in [0])\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("gx_made")
    monkeypatch.delitem(sys.modules, "gx_made")
    made = [c for c in module.mk.__code__.co_consts if isinstance(c, types.CodeType)]
    assert sorted(own_writes(c) for c in made) == [frozenset(), frozenset({"calls"})]


MADE_BY_A_LIBRARY = {
    "dataclass": "from dataclasses import dataclass\n@dataclass\nclass P:\n    a: int = 1\n    b: int = 2\n",
    "frozen_dataclass": "from dataclasses import dataclass\n@dataclass(frozen=True, order=True)\nclass P:\n"
    "    a: int = 1\n    b: int = 2\n",
    "namedtuple": "import collections\nP = collections.namedtuple('P', 'a b', defaults=(1, 2))\n",
    "typed_namedtuple": "import typing\nclass P(typing.NamedTuple):\n    a: int = 1\n    b: int = 2\n",
}


@pytest.mark.parametrize("kind", list(MADE_BY_A_LIBRARY))
def test_a_method_a_library_wrote_for_a_class_in_a_module_is_stored(cash_magics, tmp_path, monkeypatch, kind):
    """``@dataclass`` writes ``__init__`` from text of its own (``<string>``):
    for a class in a module of the user's that counted as a function whose
    unreadable text changes anything, and every statement making such an
    object was refused (the parameter sweep of a backtest, all 32 runs)."""
    name = f"made_{kind}"
    (tmp_path / f"{name}.py").write_text(MADE_BY_A_LIBRARY[kind], encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    run_cash_cell(cash_magics, f"import time\nfrom {name} import P")
    statement = f"w = (P(a=3).a, P(3) == P(3), repr(P(3)), {SLEEP})"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for _ in range(2):
            run_cash_cell(cash_magics, statement)
    sys.modules.pop(name, None)
    assert _status(cash_magics) == ["RESTORED"]
    assert not [w for w in seen if "UNSEEN" in str(w.message)]


@pytest.mark.parametrize("how", ["lambda", "def", "staticmethod"])
def test_a_special_method_set_on_the_class_later_and_edited_runs_again(cash_magics, how):
    """``Vec.__add__ = lambda self, o: 10`` has no name of its own: an edit
    to 20 went unseen by the trace and the hit served 10. The trace finds it
    again as ``Vec.__add__``; code it cannot find again does not vouch."""
    made = {
        "lambda": "Vec.__add__ = lambda self, o: {mark}\n",
        "def": "def _add(self, o):\n    return {mark}\nVec.__add__ = _add\ndel _add\n",
        "staticmethod": "Vec.mk = staticmethod(lambda: {mark})\nVec.__add__ = lambda self, o: Vec.mk()\n",
    }[how]
    seen = []
    for mark in (10, 20):
        for cell in (
            "import time\nclass Vec:\n    pass\n",
            "v = Vec()\n",
            made.format(mark=mark),
            f"w = (v + 1, {SLEEP})",
        ):
            run_cash_cell(cash_magics, cell)
        seen.append(cash_magics.shell.user_ns["w"][0])
    assert seen == [10, 20]


WRITES = {
    "next": ("GEN = iter(range(5))", "next(GEN)", "next(GEN)", 1),
    "heappush": ("import heapq\nH = []", "heapq.heappush(H, 1)", "H", [1]),
    "handed_mutator": ("", "list(map(calls.append, [1]))", "calls", [1]),
    "method_on_a_global": (
        "class St:\n    def __init__(self):\n        self.n = 0\n    def bump(self):\n        self.n += 1\nSTATE = St()",
        "STATE.bump()",
        "STATE.n",
        1,
    ),
}


@pytest.mark.parametrize("kind", list(WRITES))
def test_a_special_method_that_changes_a_global_another_way_runs_each_time(cash_magics, kind):
    """``next(GEN)``, ``heapq.heappush(H, 1)``, ``map(calls.append, ...)`` and
    ``STATE.bump()`` in a special method reached two steps down
    (``d[k][0] + 1``) change a global as ``calls.append(1)`` does: a served
    entry would drop the change."""
    setup, body, effect, expected = WRITES[kind]
    head = (
        f"import time\ncalls = []\n{setup}\nclass Two:\n    def __add__(self, x):\n        {body}\n        return 0\n"
    )
    for _ in range(2):
        run_cash_cell(cash_magics, head + "d = {'a': [Two()]}\nk = 'a'")
        run_cash_cell(cash_magics, f"w = (d[k][0] + 1, {SLEEP})")
        assert eval(effect, cash_magics.shell.user_ns) == expected
        _restart(cash_magics)
