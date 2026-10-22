"""A function of the user's that ran in a statement on a path cash does not
read from its text, and changes something the statement does not name, keeps
the statement from being stored (Python 3.12+, from the trace of the run).

``Vec.__add__`` appends to ``calls``. Run by ``s + 1`` on an object-dtype
column of ``Vec``, by ``d[k][0] + 1``, by a closure, a classmethod, a
comprehension, ``sum(lst, Vec())`` or ``partial(operator.add, v)``: each was
served after a restart with ``calls == []``, where a run appends.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from cash.notebook import shadow
from cash.notebook.statement import processor
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="reads the trace of the run (3.12+)")

SLEEP = f"(time.sleep({F}), 0)[1]"
HEAD = """import time, types, functools, operator
calls = []
class Vec:
    def __add__(self, x):
        calls.append(1)
        return 0
"""
NAMES = ("calls", "Vec", "v", "d", "k", "i", "lst", "np", "pd", "arr", "s", "df", "K", "g", "mk", "p", "w", "time")

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
