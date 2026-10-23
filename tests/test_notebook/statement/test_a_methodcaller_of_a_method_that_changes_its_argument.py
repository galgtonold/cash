"""``operator.methodcaller('add', lst, 1)(a)`` runs ``A.add(a, lst, 1)``,
which appends to ``lst``. The statement hands ``lst`` only to
``methodcaller``: what the methodcaller hands the method in a parameter the
method changes is now changed by the statement, and a method that only
reads stays served. In a kernel the statement was served stale
(``tests/test_notebook_integration/calls``); here a methodcaller with
keywords only also stopped cash (its reduce has no positional name).
"""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

SETUP = (
    "import operator, time\n"
    "class A:\n"
    "    def add(self, lst, x):\n"
    "        time.sleep(0.05)\n"
    "        lst.append(x)\n"
    "        return len(lst)\n"
    "    def size(self, lst, x):\n"
    "        time.sleep(0.05)\n"
    "        return len(lst) + x\n"
    "a = A()"
)
CASES = {
    "inline": ("lst = [0]", "r = operator.methodcaller('add', lst, 1)(a)"),
    "named": ("lst = [0]\nmc = operator.methodcaller('add', lst, 1)", "r = mc(a)"),
    "inline keyword": ("lst = [0]", "r = operator.methodcaller('add', lst, x=1)(a)"),
    "named keywords": ("lst = [0]\nmc = operator.methodcaller('add', lst=lst, x=1)", "r = mc(a)"),
    "on the class": ("lst = [0]", "r = operator.methodcaller('add', a, lst, 1)(A)"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_run_all_again_appends_again(cash_magics, case):
    # Stored only as dear as the kernel stores them: with every statement
    # kept, ``lst = [0]`` is restored with the list the served call left,
    # and the stale result hides behind it.
    build, call = CASES[case]
    cells = [SETUP, build, call, "n = (len(lst), r)"]
    ns = cash_magics.shell.user_ns
    # The class's cell runs once, as Run All skips it when unchanged: the
    # same object is handed the second time.
    for run in range(2):
        for cell in cells[1 if run else 0 :]:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["n"] == (2, 2), "the methodcaller's append was served, not run"


def test_a_method_that_only_reads_is_served(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = [SETUP, "lst = [0]", "r = operator.methodcaller('size', lst, 1)(a)"]
    for run in range(2):
        for cell in cells[1 if run else 0 :]:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics._last_cell_metrics["statements"][-1]["status"] == CacheStatus.RESTORED
