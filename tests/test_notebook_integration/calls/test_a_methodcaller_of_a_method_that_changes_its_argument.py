"""``r = operator.methodcaller('add', lst, 1)(a)`` runs ``A.add(a, lst,
1)``, which appends to ``lst``. In a kernel, where Run All skips the
unchanged class cell and hands the same ``a`` again, the statement was
served and ``lst`` stayed one item short, inline and with the methodcaller
named. (The in-process twin of this test is in
``tests/test_notebook/statement``.)
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = (
    "import operator, time\n"
    "class A:\n"
    "    def add(self, lst, x):\n"
    "        time.sleep(0.3)\n"
    "        lst.append(x)\n"
    "        return len(lst)\n"
    "a = A()"
)


@pytest.mark.parametrize(
    ("build", "call"),
    [
        ("lst = [0]", "r = operator.methodcaller('add', lst, 1)(a)"),
        ("lst = [0]\nmc = operator.methodcaller('add', lst, 1)", "r = mc(a)"),
    ],
    ids=["inline", "named"],
)
def test_run_all_again_appends_again(nb_runner, build, call):
    nb_runner.create_notebook([SETUP, build, call, "n = (len(lst), r)"])
    nb_runner.start_kernel()
    for _ in range(3):
        nb_runner.run_all()
        assert nb_runner.peek("n") == "(2, 2)", "the methodcaller's append was served, not run"
