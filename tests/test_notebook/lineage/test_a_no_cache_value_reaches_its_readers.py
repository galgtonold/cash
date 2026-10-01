"""A statement that reads a ``# @cash:no-cache`` value sees this run's value.

A no-cache statement runs every time, as if cash were not installed. Its output
used to keep the same lineage on every run, since its source and inputs had not
changed, so a cached reader below it kept its key and returned what it computed
from the first run's value. Each test runs the notebook top to bottom twice,
with every statement stored (``%cash_persist on``) so the reader is cached
however cheap it is.
"""

from __future__ import annotations

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell


def _run_all(magics, cells: list[str]) -> None:
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


def _two_runs(magics, cells: list[str]) -> list[tuple]:
    magics.cash_persist("on")
    seen = []
    for _ in range(2):
        _run_all(magics, cells)
        seen.append((magics.shell.user_ns["t"], magics.shell.user_ns["y"]))
    return seen


def test_a_reader_below_sees_the_new_value(cash_magics):
    cells = ["import random", "# @cash:no-cache\nt = random.random()", "y = t + 0"]
    (t1, y1), (t2, y2) = _two_runs(cash_magics, cells)
    assert t1 != t2, "setup: the no-cache draw must change"
    assert (y1, y2) == (t1, t2), "the reader must be computed from this run's value, not replayed"


def test_a_reader_in_the_same_cell_sees_the_new_value(cash_magics):
    cells = ["import random", "t = random.random()  # @cash:no-cache\ny = t + 0"]
    (t1, y1), (t2, y2) = _two_runs(cash_magics, cells)
    assert (y1, y2) == (t1, t2)


def test_an_unchanged_value_keeps_its_reader_cached(cash_magics):
    """Only a new value re-keys the reader: the same value still hits."""
    cells = ["# @cash:no-cache\nt = 5", "y = t + 0"]
    _two_runs(cash_magics, cells)
    statuses = [m.get("status") for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    assert statuses == [CacheStatus.RESTORED], statuses


def test_running_the_reader_alone_leaves_the_value_alone(cash_magics):
    """The value's lineage matches what the upstream check expects, so running
    the reader on its own does not run the no-cache statement again."""
    cells = ["import random", "# @cash:no-cache\nt = random.random()", "y = t + 0"]
    _run_all(cash_magics, cells)
    t = cash_magics.shell.user_ns["t"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.shell.user_ns["t"] == t
    assert cash_magics.shell.user_ns["y"] == t
