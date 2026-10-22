"""A memo's counter called where the trace does not see it watched is still
caught: a method or a closure that called something plain first and the
counter next (``self.g()``, ``o.stat()``, a factory's ``g()``), and a counter
handed to code the trace does not see by a keyword, a later argument or a
method's argument (``groupby([f], key=DESC)``, ``pool.map(DESC, [f])``,
``pool.submit(pick(f))``). The statement reruns, as Python does. One that
calls no counter is still served."""

from __future__ import annotations

import dis
import linecache
import types

import pytest

from cash.decorator import code_trace
from tests.test_notebook.restore.test_review_memofill4_e2e import _plain
from tests.test_notebook.restore.test_review_memofill5_e2e import SLEEP, _cycle2

pytestmark = pytest.mark.skipif(not code_trace.on(), reason="what ran is recorded on Python 3.12+ only")

IMPORTS = "import functools, itertools, operator, time, types\nimport concurrent.futures as cf"
MEMO = "@functools.cache\ndef f(n):\n    return n * 2"
# The counter by a name no text holds: only the run-time trace sees it.
PICK = "PARTS = ('cache', 'info')\ndef pick(g):\n    return getattr(g, '_'.join(PARTS))\ndef plain():\n    return 0\n"
BOX = PICK + (
    "class Box:\n    def __init__(self, g):\n        self.g = g\n    def run(self):\n        return self.g()\n"
    "def setg(b, g):\n    b.g = g\n    return b\n"
)
DESC = PICK + "DESC = vars(type(f))['_'.join(PARTS)]\n"
# A look-alike: the same calls, handing back no counter.
ALIKE = "\ndef pick(g):\n    return lambda: types.SimpleNamespace(currsize=2)\nDESC = lambda g: types.SimpleNamespace(currsize=2)\n"

CASES = {
    # The counter is never a call's first argument here: the watch reads that
    # one as it is handed.
    "a_method_calling_an_attribute": (BOX, "(Box(plain).run(), setg(Box(plain), pick(f)).run())[1]"),
    "objects_holding_a_callable": (
        PICK + "def holder(_, g):\n    return types.SimpleNamespace(x=0, stat=g)\n",
        "[o.stat() for o in [holder(0, plain), holder(0, pick(f))]][1]",
    ),
    "a_factory_closure": (
        PICK + "def make(_, g):\n    def run():\n        return g()\n    return run\n",
        "(make(0, plain)(), make(0, pick(f))())[1]",
    ),
    "a_keyword_to_a_builtin": (DESC, "next(itertools.groupby([f], key=DESC))[0]"),
    "a_later_argument_to_a_builtin": (DESC, "functools.reduce(operator.call, [f], DESC)"),
    "a_thread_pool_map": (DESC, "list(cf.ThreadPoolExecutor(1).map(DESC, [f]))[0]"),
    "a_thread_pool_submit": (PICK, "cf.ThreadPoolExecutor(1).submit(pick(f)).result()"),
}


def _cells(setup: str, expr: str, edit: str) -> list[str]:
    # Inside a helper, so the statement is otherwise served.
    helper = setup + "\ndef go():\n    return " + expr + "\n"
    return [IMPORTS, MEMO, helper, edit, f"a = (f(1), go().currsize, time.sleep({SLEEP}))[:2]"]


@pytest.mark.parametrize("case", list(CASES))
def test_a_counter_held_or_handed_on_reruns(cash_magics, case):
    setup, expr = CASES[case]
    cells1 = _cells(setup, expr, "x = f(5)")
    cells2 = _cells(setup, expr, "x = f(5) + f(6)")
    plain, _ = _plain(cells2)
    _cycle2(cash_magics, cells1, cells2)
    assert cash_magics.shell.user_ns.get("a") == plain["a"] == (2, 3)


@pytest.mark.parametrize("case", list(CASES))
def test_the_same_calls_handing_back_no_counter_are_served(cash_magics, case):
    setup, expr = CASES[case]
    cells = _cells(setup + ALIKE, expr, "x = f(5)")
    _, seen, _ = _cycle2(cash_magics, cells, cells)
    assert seen[-1] == ["RESTORED"]


def test_a_method_the_receivers_class_holds_is_let_go(monkeypatch):
    # ``rows.append(x)``: the list's own method, not an attribute an object
    # was handed, which could be a counter. Such a site is let go.
    source = "rows.append(1)\n"
    monkeypatch.setitem(linecache.cache, "<append>", (len(source), None, [source], "<append>"))
    code = compile(source, "<append>", "exec")
    offset = next(i.offset for i in dis.get_instructions(code) if i.opname == "CALL")
    assert code_trace.callee_varies(code, offset, {"rows": []}) == "append"
    assert code_trace.from_class(list.append, [], "append")
    assert not code_trace.from_class(len, types.SimpleNamespace(stat=len), "stat")
