"""A statement that runs every time because what it binds changes from run to
run (the clock, a fresh id, pyplot's current figure read through a helper)
hands its readers this run's value.

Its source and inputs are the same on every run, so its output kept the same
lineage, and a cached reader below kept its key and returned what it computed
from the first run's value. Each test runs the notebook top to bottom twice,
with every statement stored (``%cash_persist on``) so the reader is cached
however cheap it is.
"""

from __future__ import annotations

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell


def _run_all(magics, cells: list[str]) -> None:
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


def _two_runs(magics, cells: list[str], name: str) -> list[tuple]:
    magics.cash_persist("on")
    seen = []
    for _ in range(2):
        _run_all(magics, cells)
        ns = magics.shell.user_ns
        seen.append((ns[name], ns["report"]))
    return seen


@pytest.mark.parametrize(
    ("setup", "statement"),
    [
        ("from datetime import datetime", "run_id = datetime.now().strftime('%H%M%S%f')"),
        ("import uuid", "run_id = uuid.uuid4().hex"),
        ("import secrets", "run_id = secrets.token_hex(8)"),
        ("import time\ndef tick():\n    return time.perf_counter_ns()", "run_id = tick()"),
    ],
    ids=["clock", "uuid", "secrets", "clock-helper"],
)
def test_a_reader_below_sees_this_runs_value(cash_magics, setup, statement):
    cells = [setup, statement, "report = {'run': run_id}"]
    (v1, r1), (v2, r2) = _two_runs(cash_magics, cells, "run_id")
    assert v1 != v2, "setup: the value must change between runs"
    assert (r1["run"], r2["run"]) == (v1, v2), "the reader must be computed from this run's value"


def test_a_reader_in_the_same_cell_sees_this_runs_value(cash_magics):
    cells = ["import uuid", "run_id = uuid.uuid4().hex\nreport = {'run': run_id}"]
    (v1, r1), (v2, r2) = _two_runs(cash_magics, cells, "run_id")
    assert (r1["run"], r2["run"]) == (v1, v2)


def test_a_count_read_from_pyplot_through_a_helper_reaches_its_reader(cash_magics):
    helpers = (
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "def n_lines():\n    return len(plt.gca().lines)"
    )
    reader = "k = n_lines()\nreport = f'{k} lines'"
    cash_magics.cash_persist("on")
    one = [helpers, "plt.figure()\nplt.plot([1, 2])", reader]
    _run_all(cash_magics, one)
    assert cash_magics.shell.user_ns["report"] == "1 lines"
    two = [helpers, "plt.figure()\nplt.plot([1, 2])\nplt.plot([3, 4])", reader]
    _run_all(cash_magics, two)
    assert cash_magics.shell.user_ns["k"] == 2
    assert cash_magics.shell.user_ns["report"] == "2 lines"


def test_the_same_value_keeps_its_reader_cached(cash_magics):
    """Only a new value re-keys the reader: today's date twice still hits."""
    cells = ["from datetime import date", "day = date.today().isoformat()", "report = {'run': day}"]
    _two_runs(cash_magics, cells, "day")
    statuses = [m.get("status") for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    assert statuses == [CacheStatus.RESTORED], statuses
