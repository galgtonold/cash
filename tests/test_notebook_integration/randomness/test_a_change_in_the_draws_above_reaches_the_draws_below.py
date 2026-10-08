"""A change in the draws above a seeded draw gives it the numbers a plain kernel does.

With ``np.random.seed(42)`` in a setup cell, changing how many numbers a cell
draws, inserting a draw cell or swapping two, then Run All or Restart & Run
All, served the draws below from the cache with the numbers of the old stream
position. An unchanged notebook is still served from the cache on every Run
All: a bare re-seed keys the same each time.
"""

from __future__ import annotations

import ast

import nbformat
import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

np = pytest.importorskip("numpy")

SETUP = (
    "import numpy as np, time\n"
    "def slow(a):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n"
    "    return a\n"
    "np.random.seed(42)"
)
SHOW_Y = "Y = slow(np.random.rand(2))\nprint('Y', Y.round(4).tolist())"

_STATUSES = (
    "[str(s.get('status')) for s in "
    "get_ipython().magics_manager.magics['line']['cash_status'].__self__"
    "._last_cell_metrics['statements']]"
)


def _plain(*sizes):
    np.random.seed(42)
    return [np.random.rand(n).round(4).tolist() for n in sizes]


def test_more_draws_above_on_run_all_and_restart(nb_runner):
    nb_runner.create_notebook([SETUP, "X = slow(np.random.rand(3))", SHOW_Y])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert f"Y {_plain(3, 2)[1]}" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, "X = slow(np.random.rand(5))")
    nb_runner.run_all()
    assert f"Y {_plain(5, 2)[1]}" in nb_runner.get_output(3)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert f"Y {_plain(5, 2)[1]}" in nb_runner.get_output(3)


def test_an_inserted_draw_cell_above(nb_runner):
    nb_runner.create_notebook([SETUP, SHOW_Y])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cell = nbformat.v4.new_code_cell("X = np.random.rand(3)")
    cell.id = "inserted"
    nb_runner.nb.cells.insert(1, cell)
    nb_runner._save_notebook()
    nb_runner.run_all()
    assert f"Y {_plain(3, 2)[1]}" in nb_runner.get_output(3)


def test_an_unchanged_notebook_is_restored_on_the_second_run_all(nb_runner):
    nb_runner.create_notebook([SETUP, "X = slow(np.random.rand(3))", SHOW_Y])
    nb_runner.start_kernel()
    nb_runner.run_all()
    for cell in (1, 2):
        nb_runner.run_cell(cell)
    assert ast.literal_eval(nb_runner.peek(_STATUSES)) == ["RESTORED"]
    nb_runner.run_cell(3)
    assert "RESTORED" in ast.literal_eval(nb_runner.peek(_STATUSES))
    assert f"Y {_plain(3, 2)[1]}" in nb_runner.get_output(3)
