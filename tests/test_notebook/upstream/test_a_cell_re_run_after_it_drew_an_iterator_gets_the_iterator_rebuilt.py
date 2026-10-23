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
