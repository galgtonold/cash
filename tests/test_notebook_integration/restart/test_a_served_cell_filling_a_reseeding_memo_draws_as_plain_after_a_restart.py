"""A served cell whose run filled a memo helper that reseeds ``random`` leaves
the stream where the run left it, after a restart too, when a cell above
called the same helper.

Shaped like the TSP notebook: ``random_cities(n)`` reseeds on a miss, a cell
above calls it for another ``n``, and the cell below draws.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import functools, random, time"
CITIES = (
    "@functools.cache\n"
    "def random_cities(n, seed=1234):\n"
    "    random.seed(n * seed)\n"
    "    return frozenset(complex(round(random.uniform(0, 300)), round(random.uniform(0, 200))) for _ in range(n))"
)
ABOVE = "small = random_cities(5)"
SERVED = "t = (sorted(random_cities(9), key=abs), time.sleep(0.3))[0]"
DRAW = "x = random.random()"
SERVED_CELL = 4


def _cells(above: str = ABOVE) -> list[str]:
    return ["%cash_badge print\n" + SETUP, CITIES, above, SERVED, DRAW]


def _plain(cells: list[str]) -> dict:
    """What a restart and Run All in plain Python leave."""
    import random

    state = random.getstate()
    ns: dict = {"__name__": "__plain__"}
    try:
        for cell in cells:
            exec(cell.replace("%cash_badge print\n", ""), ns)
        return {"t": ns["t"], "x": ns["x"]}
    finally:
        random.setstate(state)


def _check(nb_runner, plain: dict, when: str) -> None:
    assert shows_cached(nb_runner.get_output(SERVED_CELL)), (when, nb_runner.get_output(SERVED_CELL))
    assert nb_runner.peek("t") == repr(plain["t"]), when
    assert float(nb_runner.peek("x")) == plain["x"], when


def test_the_draw_below_is_plain_after_a_restart_when_a_cell_above_called_the_helper(nb_runner):
    cells = _cells()
    plain = _plain(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert float(nb_runner.peek("x")) == plain["x"]
    nb_runner.settle_writes()
    for rep in range(2):
        nb_runner.restart()
        nb_runner._init_cash()
        nb_runner.run_all()
        _check(nb_runner, plain, f"restart {rep}")


def test_the_draw_below_is_plain_on_a_second_run_all_in_the_same_kernel(nb_runner):
    """Control: the same kernel, Run All twice."""
    cells = _cells()
    plain = _plain(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, plain, "second Run All")


def test_the_draw_below_is_plain_after_a_restart_with_no_cell_above_calling_the_helper(nb_runner):
    """Control: nothing above calls ``random_cities``."""
    cells = _cells("small = 1")
    plain = _plain(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    _check(nb_runner, plain, "restart")
