"""Sizing a statement's output runs none of the user's code.

Before storing ``pair = [Hook(), L]`` cash weighs it: the RAM tier and the
store's size check read an ``nbytes`` and call ``sys.getsizeof``. On a class
of the user's, that ran its ``nbytes`` property, its ``__getattr__`` (asked
for ``nbytes``) or its ``__sizeof__``; one that changes ``L`` changed it twice
behind the statement's back, and a later ``s = sum(L)`` was served from
before. A size is only a guess: cash reads it without calling the user.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

HOOKS = {
    "nbytes_property": "class Hook:\n    @property\n    def nbytes(self):\n        L.append(1)\n        return 8",
    "getattr_fallback": (
        "class Hook:\n"
        "    def __getattr__(self, name):\n"
        "        if not name.startswith('__'):\n"
        "            L.append(1)\n"
        "        raise AttributeError(name)"
    ),
    "sizeof": "class Hook:\n    def __sizeof__(self):\n        L.append(1)\n        return 64",
}


@pytest.mark.parametrize("hook", sorted(HOOKS))
@pytest.mark.parametrize("statement", ["pair = [Hook(), L]", "pair = [Hook(), len(L)]"])
def test_storing_a_value_leaves_the_users_list_alone(cash_magics, cash_instance, hook, statement):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = ["L = [0]\n" + HOOKS[hook], "s = sum(L)", statement, "# again\ns = sum(L)", "t = len(L)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["L"] == [0], "cash ran the user's hook while sizing the value"
    assert ns["s"] == sum(ns["L"])
    assert ns["t"] == len(ns["L"])
