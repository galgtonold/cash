"""Without the run-time trace (Python 3.10 and 3.11),
a statement reaching a function of the user's that looks an attribute up by
a name it does not write out (``getattr(g, '_'.join(PARTS))``), or hands
``getattr`` on, or reads a memo's unbound counter (``vars(type(f))[...]``),
runs again while a memo is about, and says why: the name may be a counter's.
One whose lookups are written out is still served."""

from __future__ import annotations

import pytest

from cash.notebook import helper_memos
from cash.notebook.shadow import Shadow
from tests._cell_driver import run_cash_cell
from tests.test_notebook.restore.test_review_memofill4_e2e import _plain
from tests.test_notebook.restore.test_review_memofill5_e2e import SLEEP, _cycle2

IMPORTS = "import functools, operator, time, types"
MEMO = "@functools.cache\ndef f(n):\n    return n * 2"
PICK = "PARTS = ('cache', 'info')\ndef pick(g):\n    return getattr(g, '_'.join(PARTS))\ndef plain():\n    return 0\n"

CASES = {
    "a_loop_over_a_hidden_counter": (PICK, "[g() for g in [plain, pick(f)]][1]"),
    "a_name_built_from_letters": (
        "def info(g):\n    return getattr(g, ''.join(map(chr, [99, 97, 99, 104, 101, 95, 105, 110, 102, 111])))()",
        "info(f)",
    ),
    "getattr_handed_on": (PICK, "functools.reduce(getattr, ['_'.join(PARTS)], f)()"),
    "an_unbound_counter_in_a_global": (PICK + "DESC = vars(type(f))['_'.join(PARTS)]\n", "list(map(DESC, [f]))[0]"),
}


@pytest.fixture(autouse=True)
def _no_trace(monkeypatch):
    # What Python 3.10 and 3.11 leave: no trace of what a statement ran.
    monkeypatch.setattr(Shadow, "begin", lambda self, kind: None)


def _cells(setup: str, expr: str, edit: str) -> list[str]:
    helper = setup + "\ndef go():\n    return " + expr + "\n"
    return [IMPORTS, MEMO, helper, edit, f"a = (f(1), go().currsize, time.sleep({SLEEP}))[:2]"]


@pytest.mark.parametrize("case", list(CASES))
def test_a_name_no_text_holds_reruns_and_says_why(cash_magics, case):
    setup, expr = CASES[case]
    cells1 = _cells(setup, expr, "x = f(5)")
    cells2 = _cells(setup, expr, "x = f(5) + f(6)")
    plain, _ = _plain(cells2)
    _cycle2(cash_magics, cells1, cells2)
    assert cash_magics.shell.user_ns.get("a") == plain["a"] == (2, 3)


def test_the_reason_names_the_missing_trace(cash_magics, monkeypatch):
    told = []
    used = helper_memos.used

    def telling(*args, **kwargs):
        got = used(*args, **kwargs)
        told.append(got)
        return got

    monkeypatch.setattr(helper_memos, "used", telling)
    setup, expr = CASES["a_loop_over_a_hidden_counter"]
    cells = _cells(setup, expr, "x = f(5)")
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert isinstance(told[-1], tuple)
    assert "without the run-time trace" in told[-1][1]


def test_a_lookup_written_out_is_served(cash_magics):
    setup = "class Cfg:\n    k = 2\ndef plain():\n    return types.SimpleNamespace(currsize=getattr(Cfg, 'k'))\n"
    cells = _cells(setup, "plain()", "x = f(5)")
    _, seen, _ = _cycle2(cash_magics, cells, cells)
    assert seen[-1] == ["RESTORED"]
