"""After a restart, a cell run alone reads a notebook variable as the
functions called above it left it.

``registry = []``, ``def register(f): registry.append(...)``, ``register(a)``,
then a reader: the simulation asks which statements above change
``registry``. Whether a name a called function changes is a variable of the
notebook's was asked of the kernel's namespace alone, which holds nothing
right after a restart: the call was taken to change nothing, and the reader
got the list as its first assignment left it.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.mutation_effects import statement_effects
from tests._cell_driver import run_cash_cell

CASES = {
    "a call that appends to a list": (
        [
            "registry = []\ndef register(f):\n    registry.append(f.__name__)\n    return f",
            "def a(): pass\nregister(a)",
        ],
        "out = list(registry)",
        ["a"],
    ),
    "a counter rebound under global": (
        ["cnt = 0\ndef bump():\n    global cnt\n    cnt += 1", "bump()"],
        "out = cnt",
        1,
    ),
    "a dict a helper fills": (
        ["state = {}\ndef setk(k, v):\n    state[k] = v", "setk('a', 1)"],
        "out = dict(state)",
        {"a": 1},
    ),
    "a key function that logs": (
        ["seen = []\ndef k(v):\n    seen.append(v)\n    return v", "s = sorted([3, 1, 2], key=k)"],
        "out = list(seen)",
        [3, 1, 2],
    ),
    "two cells calling the same helper": (
        ["LOG = []\ndef rec(v):\n    LOG.append(v)\n    return v * 2", "a = rec(1)", "b = rec(2)"],
        "out = list(LOG)",
        [1, 2],
    ),
    "a helper called in a loop": (
        ["LOG = []\ndef note(v):\n    LOG.append(v)", "for i in range(3):\n    note(i)"],
        "out = list(LOG)",
        [0, 1, 2],
    ),
}


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the notebook's names."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    user_ns = magics.shell.user_ns
    for name in [n for n in user_ns if not n.startswith("_") and n not in ("In", "Out", "get_ipython", "exit", "quit")]:
        del user_ns[name]


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_sees_what_the_calls_above_changed(cash_magics, case):
    above, reader, expected = CASES[case]
    cells = [*above, reader]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["out"] == expected, "control: the top-to-bottom run"

    _restart(cash_magics)
    run_cash_cell(cash_magics, reader, cells=cells)

    assert cash_magics.shell.user_ns["out"] == expected


SOURCES = {
    "register": "def register(f):\n    registry.append(f.__name__)\n    return f",
    "bump": "def bump():\n    global cnt\n    cnt += 1",
}


def _effects(code: str, simulated=frozenset()):
    return statement_effects(code, ast.parse(code), namespace={}, resolve_source=SOURCES.get, simulated=simulated)


def test_a_name_the_simulation_bound_is_a_variable_a_callee_changes():
    """With nothing in the namespace, what the simulation bound above the
    statement says which names are the notebook's variables."""
    assert "registry" in _effects("register(a)", {"registry", "register", "a"}).callee_globals
    assert "cnt" in _effects("bump()", {"bump"}).outputs, "a global the function binds is the notebook's"


def test_a_name_neither_holds_is_not_taken_for_a_variable():
    """Control: a name the callee changes that neither the kernel nor the
    simulation binds is the callee's own, and declaring it would invent a
    variable."""
    assert "registry" not in _effects("register(a)").callee_globals
    assert "registry" not in _effects("register(a)", {"register", "a"}).callee_globals
    assert "cnt" not in _effects("bump()").outputs
