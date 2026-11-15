"""A decorated function called in notebook cells, with an argument of a
class the notebook defines, gives what a restart and Run All in plain
Python gives, while cell after cell redefines the class under the same
name, rebinds its method, and changes a global the method reads.

The key folds the argument's class (`trace_items.remembered_class_digest`,
`code_args`), remembered per class while its world stands still
(`cash.decorator.class_memo`). That memo must not serve a class's fold
across any of these edits, and the watch it keeps must not hold the
notebook's variables: a probe of the list a method of ``Vec`` appends to was
one more holder of ``calls``, and the notebook, which counts a variable's
holders to tell whether a restored copy can stand in for it, then never
stored ``calls += [...]``.
"""

from __future__ import annotations

import pytest

from cash.decorator import class_memo
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

SLEEP = f"(time.sleep({F}), 0)[1]"

SETUP = f"""import time
calls = []
K = 5
class Vec:
    def __init__(self, n):
        self.n = n
    def __add__(self, x):
        return self.n + x
@CACHE
def f(v):
    time.sleep({F})
    return v + 1
r1 = f(Vec(1))
"""

EDITS = [
    # The class defined again under the same name, with another body.
    "class Vec:\n    def __init__(self, n):\n        self.n = n\n    def __add__(self, x):\n"
    "        return self.n * 10 + x\nr2 = f(Vec(1))",
    # Its method rebound by a later cell.
    "def _add(self, x):\n    return -x\nVec.__add__ = _add\nr3 = f(Vec(1))",
    # A method reading a global, then the global changed.
    "def _add(self, x):\n    return self.n + x + K\nVec.__add__ = _add\nr4 = f(Vec(1))",
    "K = 6\nr5 = f(Vec(1))",
    "r6 = f(Vec(1))",
]

#: A method appending to a notebook variable, as ``Vec.log`` does.
LOGGING = "def _log(self):\n    calls.append(1)\n    return 0\nVec.log = _log\nr2 = f(Vec(2))\nr3 = f(Vec(2))"

NAMES = ("r1", "r2", "r3", "r4", "r5", "r6")


def _plain() -> dict:
    """Restart and Run All, without cash."""
    ns: dict = {"CACHE": lambda fn: fn}
    for cell in (SETUP, *EDITS):
        exec(cell, ns)
    return ns


def test_edits_to_the_argument_class_are_followed_as_plain_python_does(cash_magics, tmp_path):
    cash_magics.shell.user_ns["CACHE"] = _cache(tmp_path)
    before = dict(class_memo.STATS)
    for cell in (SETUP, *EDITS):
        run_cash_cell(cash_magics, cell)
    ns, plain = cash_magics.shell.user_ns, _plain()
    assert [ns[n] for n in NAMES] == [plain[n] for n in NAMES]
    if class_memo.ENABLED:
        assert class_memo.STATS["scans"] > before["scans"], "the class memo was never asked"


def test_a_statement_changing_what_the_class_appends_to_is_stored(cash_magics, tmp_path):
    """After decorated calls met ``Vec`` (whose ``log`` appends to
    ``calls``), ``calls += [...]`` is still stored and served."""
    cash_magics.shell.user_ns["CACHE"] = _cache(tmp_path)
    before = dict(class_memo.STATS)
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, LOGGING)
    if class_memo.ENABLED:
        assert class_memo.STATS["scans"] > before["scans"], "the class memo was never asked"
    statement = f"calls += [Vec(3).log(), {SLEEP}]"
    # A decorated call after each rebinding: the class is watched again
    # while ``calls`` holds the list the statement then changes.
    reset = "calls = []\nr4 = f(Vec(4))"
    run_cash_cell(cash_magics, reset)
    run_cash_cell(cash_magics, statement)
    run_cash_cell(cash_magics, reset)
    run_cash_cell(cash_magics, statement)
    statuses = [str(s["status"]) for s in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    assert statuses == ["RESTORED"]
    plain: dict = {"CACHE": lambda fn: fn}
    for cell in (SETUP, LOGGING, reset, statement, reset):
        exec(cell, plain)
    exec(statement.replace(SLEEP, "0"), plain)
    assert cash_magics.shell.user_ns["calls"] == plain["calls"]


def _cache(tmp_path):
    import cash

    return cash.Cash(cache_dir=str(tmp_path / "decorated")).cache


pytestmark = pytest.mark.filterwarnings("ignore")
