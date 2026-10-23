"""A ``functools.partial`` whose function changes an argument it binds
changes that argument, built in the statement or bound to a name.

``r = functools.partial(a.add, lst)(1)``, with ``A.add`` appending to the
list it is handed, was served on the next Run All without running: ``lst``,
rebuilt by the cell above, missed the append plain Python makes. A partial
built in the statement was not read at all, and one of a bound method was
not read through the method's function.
"""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

HELPERS = (
    "import functools, time\n"
    "class A:\n"
    "    def add(self, lst, x):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "        lst.append(x)\n"
    "        return len(lst)\n"
    "    def size(self, lst, x):\n"
    f"        time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "        return len(lst) + x\n"
    "a = A()\n"
    "def add(lst, x):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    lst.append(x)\n"
    "    return len(lst)"
)
CALLS = {
    "a_bound_method_built_in_the_statement": ("lst = [0]", "r = functools.partial(a.add, lst)(1)"),
    "a_function_built_in_the_statement": ("lst = [0]", "r = functools.partial(add, lst)(1)"),
    "the_class_function_built_in_the_statement": ("lst = [0]", "r = functools.partial(A.add, a, lst)(1)"),
    "a_bound_method_bound_to_a_name": ("lst = [0]\np = functools.partial(a.add, lst)", "r = p(1)"),
}


def _run_all(cash_magics, cells):
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)


@pytest.mark.parametrize("case", list(CALLS))
def test_the_bound_argument_is_changed_on_every_run_all(cash_magics, case):
    build, call = CALLS[case]
    cells = [HELPERS, build, call, "n = len(lst)"]
    for _ in range(2):
        _run_all(cash_magics, cells)
        assert cash_magics.shell.user_ns["n"] == 2
        assert cash_magics.shell.user_ns["lst"] == [0, 1]


def test_a_partial_reading_what_it_binds_is_served(cash_magics):
    cells = [HELPERS, "lst = [0]", "r = functools.partial(a.size, lst)(1)"]
    for _ in range(2):
        _run_all(cash_magics, cells)
    assert [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]] == ["RESTORED"]
    assert cash_magics.shell.user_ns["r"] == 2
