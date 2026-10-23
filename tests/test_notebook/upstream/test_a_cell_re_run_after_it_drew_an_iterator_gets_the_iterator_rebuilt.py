"""A cell re-run on its own after it drew an iterator dry draws what a
restart and Run All draws.

``it = iter(range(6))``, ``first3 = [next(it) for _ in range(3)]``, then
``rest = list(it)``: re-run alone, the last cell found ``it`` exhausted, and
nothing said it had moved since the cells above left it: a draw moved no
lineage. ``rest`` came back ``[]``, and before that its ``print`` was served
the ``[3, 4, 5]`` of the first run while ``rest`` held ``[]``. A draw from an
iterator now moves its lineage, as one from a random generator does, so the
re-run cell has it rebuilt as the cells above leave it.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SOURCES = {
    "an_iterator": "it = iter(range(6))",
    "a_generator": "def count(n):\n    yield from range(n)\nit = count(6)",
    "a_map": "it = map(int, '012345')",
}


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("source", list(SOURCES))
def test_the_tail_drawn_again_is_what_run_all_draws(cash_magics, cash_instance, source, persist):
    if persist:
        cash_instance.reconfigure(persist_all=True)
    cells = [SOURCES[source], "first3 = [next(it) for _ in range(3)]\nprint(first3)", "rest = list(it)\nprint(rest)"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (ns["first3"], ns["rest"]) == ([0, 1, 2], [3, 4, 5])
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert (ns["first3"], ns["rest"]) == ([0, 1, 2], [3, 4, 5])
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert ns["rest"] == [3, 4, 5], "a second re-run on its own"


def test_a_run_all_after_the_draw_draws_once_per_cell(cash_magics):
    cells = ["it = iter(range(6))", "first3 = [next(it) for _ in range(3)]", "rest = list(it)"]
    ns = cash_magics.shell.user_ns
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (ns["first3"], ns["rest"]) == ([0, 1, 2], [3, 4, 5])


LOGGED = {
    "a_map": "log = []\ndef f(x):\n    log.append(x)\n    return x\nit = map(f, range(6))\nit2 = it",
    "a_generator": "log = []\ndef g():\n    for i in range(6):\n        log.append(i)\n        yield i\nit = g()\nit2 = it",
}


@pytest.mark.parametrize("source", list(LOGGED))
def test_a_draw_through_another_name_runs_nothing_above_again(cash_magics, source):
    """``it2 = it``, a cell drawing ``it``, then ``rest = list(it2)``: the
    draw moved ``it`` and not ``it2``, which the simulation moved through
    the edge ``it2 = it`` linked. The reader found ``it2`` moved and had the
    cells above run again inside every Run All, the first one too: ``log``
    held 8 draws, not 6."""
    cells = [LOGGED[source], "first = [next(it) for _ in range(2)]", "rest = list(it2)\nn = len(log)"]
    ns = cash_magics.shell.user_ns
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert (ns["first"], ns["rest"], ns["n"]) == ([0, 1], [2, 3, 4, 5], 6)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert ns["rest"] == [2, 3, 4, 5], "re-run on its own"
