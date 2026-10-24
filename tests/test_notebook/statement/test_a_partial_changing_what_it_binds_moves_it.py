"""A ``functools.partial`` whose function changes an argument it binds
changes that argument, built in the statement or bound to a name.

``r = functools.partial(a.add, lst)(1)``, with ``A.add`` appending to the
list it is handed, was served on the next Run All without running: ``lst``,
rebuilt by the cell above, missed the append plain Python makes. A partial
built in the statement was not read at all, and one of a bound method was
not read through the method's function. Nor was one whose function or
argument is any of several objects (``lists[i]``, ``lst if flag else
other``), or is itself a partial built in the statement.
"""

import ast

import pytest

from cash.analysis.handed_callables import handed_callables
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
    "an_item_under_a_variable_index": (
        "lists = [[0]]\ni = 0\nlst = lists[0]",
        "r = functools.partial(a.add, lists[i])(1)",
    ),
    "an_argument_of_a_conditional": (
        "lst = [0]\nother = [9]\nflag = True",
        "r = functools.partial(add, lst if flag else other)(1)",
    ),
    "a_function_of_a_conditional": (
        "lst = [0]\nflag = True",
        "r = functools.partial(a.add if flag else a.size, lst)(1)",
    ),
    "a_nested_partial": ("lst = [0]", "r = functools.partial(functools.partial(add, lst), 1)()"),
    "a_nested_partial_of_a_method": ("lst = [0]", "r = functools.partial(functools.partial(a.add, lst), 1)()"),
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


@pytest.mark.parametrize(
    "statement",
    ["r = functools.partial(add, pick(lists))(1)", "r = functools.partial(add, *lists)(1)"],
    ids=["what_a_call_of_yours_returns", "arguments_unpacked"],
)
def test_a_partial_binding_what_cannot_be_told_cannot_be_read(statement):
    """Bound to what only running tells, the partial makes the statement
    unreadable, so it runs every time."""
    import linecache

    code = (
        "import functools\ndef add(lst, x):\n    lst.append(x)\ndef pick(lists):\n    return lists[0]\nlists = [[0]]\n"
    )
    filename = "<partial-binds-unknown>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(code, filename, "exec"), namespace)
    found = handed_callables(ast.parse(statement), namespace)
    assert any("builds a partial" in why for why in found.unreadable)
