"""A statement whose callee changes its closure is never restored whole.

``add = make_log(); n1 = add('a'); n2 = add('a')`` run again with statements
cached restored ``n1`` and ``n2`` without running ``add``: the numbers were
right, and the list ``add`` appends to stayed empty. The list is no notebook
variable, so it has no lineage to bump and nothing to capture; the statement
re-executes, as one whose callee changes a global does, and the call inside it
is still served from the call cache.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

DEFS = (
    "def make_log():\n"
    "    seen = []\n"
    "    def add(x):\n"
    "        seen.append(x)\n"
    "        return len(seen)\n"
    "    return add\n"
    "def make_counter():\n"
    "    count = 0\n"
    "    def counter():\n"
    "        nonlocal count\n"
    "        count += 1\n"
    "        return count\n"
    "    return counter"
)


@pytest.mark.parametrize("module", ["__main__", "helpers"])
@pytest.mark.parametrize(
    ("body", "name", "expected"),
    [
        ("add = make_log()\nn1 = add('a')\nn2 = add('a')", "add", ["a", "a"]),
        ("add = make_log()\nfor x in 'ab':\n    n = add(x)", "add", ["a", "b"]),
        ("counter = make_counter()\nv1 = counter()\nv2 = counter()", "counter", 2),
    ],
)
def test_a_rerun_leaves_the_closure_as_a_run_does(cash_magics, cash_instance, mock_shell, body, name, expected, module):
    """*module* is where the factories live: a notebook's own (the kernel's
    ``__main__``), or a module it imported, whose closures the statement
    store does not already refuse to keep by value."""
    mock_shell.user_ns["__name__"] = module
    # Statements and calls are stored however cheap, as on a slow machine.
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [DEFS, body]
    run_cash_cell(cash_magics, DEFS, cells=cells)
    run_cash_cell(cash_magics, body, cells=cells)

    run_cash_cell(cash_magics, body, cells=cells)

    assert mock_shell.user_ns[name].__closure__[0].cell_contents == expected
