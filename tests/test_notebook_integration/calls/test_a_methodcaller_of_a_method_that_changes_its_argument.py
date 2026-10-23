"""``r = operator.methodcaller('add', lst, 1)(a)`` runs ``A.add(a, lst,
1)``, which appends to ``lst``. In a kernel, where Run All skips the
unchanged class cell and hands the same ``a`` again, the statement was
served and ``lst`` stayed one item short, inline and with the methodcaller
named. So it was when the methodcaller was read out of a list or a dict,
handed to ``map``, or called a method that takes ``*items`` or ``**kw``, or
hands ``lst`` to a function that appends: what a methodcaller hands its
method is now read as a call of the method itself reads it. (The
in-process twin of this test is in ``tests/test_notebook/statement``.)
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = (
    "import operator, time\n"
    "def helper(lst, x):\n"
    "    lst.append(x)\n"
    "class A:\n"
    "    def add(self, lst, x):\n"
    "        time.sleep(0.3)\n"
    "        lst.append(x)\n"
    "        return len(lst)\n"
    "    def add_star(self, *items):\n"
    "        time.sleep(0.3)\n"
    "        items[0].append(1)\n"
    "        return len(items[0])\n"
    "    def add_kw(self, **kw):\n"
    "        time.sleep(0.3)\n"
    "        kw['lst'].append(1)\n"
    "        return len(kw['lst'])\n"
    "    def add_via(self, lst, x):\n"
    "        time.sleep(0.3)\n"
    "        helper(lst, x)\n"
    "        return len(lst)\n"
    "a = A()"
)


@pytest.mark.parametrize(
    ("build", "call"),
    [
        ("lst = [0]", "r = operator.methodcaller('add', lst, 1)(a)"),
        ("lst = [0]\nmc = operator.methodcaller('add', lst, 1)", "r = mc(a)"),
        ("lst = [0]\nmcs = [operator.methodcaller('add', lst, 1)]", "r = mcs[0](a)"),
        ("lst = [0]\nmcs = {'k': operator.methodcaller('add', lst, 1)}", "r = mcs['k'](a)"),
        ("lst = [0]", "r = list(map(operator.methodcaller('add', lst, 1), [a]))[0]"),
        ("lst = [0]\nmc = operator.methodcaller('add', lst, 1)", "r = list(map(mc, [a]))[0]"),
        ("lst = [0]", "r = operator.methodcaller('add_star', lst)(a)"),
        ("lst = [0]", "r = operator.methodcaller('add_kw', lst=lst)(a)"),
        ("lst = [0]", "r = operator.methodcaller('add_via', lst, 1)(a)"),
    ],
    ids=[
        "inline",
        "named",
        "in_a_list",
        "in_a_dict",
        "handed_to_map",
        "named_handed_to_map",
        "star_args",
        "star_kwargs",
        "through_a_helper",
    ],
)
def test_run_all_again_appends_again(nb_runner, build, call):
    nb_runner.create_notebook([SETUP, build, call, "n = (len(lst), r)"])
    nb_runner.start_kernel()
    for _ in range(3):
        nb_runner.run_all()
        assert nb_runner.peek("n") == "(2, 2)", "the methodcaller's append was served, not run"
