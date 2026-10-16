"""The threads alive as a cell starts are not held by cash while it runs.

The check for a thread a cell leaves running (NOTEBOOK-THREAD-ALIVE) kept
the ``Thread`` objects alive at the cell's start in a set until the cell
ended. The walk for the notebook functions a value holds reached that set
through the cell executor, and from each thread whatever its target and
arguments hold: under pytest, pytest-timeout's timer thread and through it
every collected test item. Each walk visited about 676,000 objects instead
of about 1,000, and a test calling the same lambdas 63 times timed out on
Windows. The set now holds thread idents.
"""

from __future__ import annotations

import threading

import pytest

from cash.notebook import held_functions
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

DEFS = f"""
import time
def convert(m, unit):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return m / (1000 if unit == 'km' else 1609.344)
"""
LAMBDAS = "to_km = lambda m: convert(m, 'km')\nto_mi = lambda m: convert(m, 'mi')"

#: The objects the thread's argument holds: far more than any walk below visits.
BIG = 200_000


class _Record:
    def __init__(self, i: int):
        self.i = i


@pytest.fixture
def thread_holding_a_big_graph():
    """A thread, started before the cell, whose argument holds BIG objects."""
    stop = threading.Event()
    graph = [_Record(i) for i in range(BIG)]
    thread = threading.Thread(target=lambda event, held: event.wait(), args=(stop, graph), daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join()


def test_a_thread_alive_while_a_cell_runs_is_not_walked(
    cash_magics, mock_shell, monkeypatch, thread_holding_a_big_graph
):
    cells = [DEFS, LAMBDAS, "dist = 42195", "x = to_km(dist)", "ys = list(map(to_mi, [dist]))"]
    for i, code in enumerate(cells[:-1]):
        run_cash_cell(cash_magics, code, cells=cells[: i + 1])

    visited = [0]
    real = held_functions._visit_group

    def counting(kind, group, *args):
        visited[0] += len(group)
        return real(kind, group, *args)

    monkeypatch.setattr(held_functions, "_visit_group", counting)
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert mock_shell.user_ns["ys"] == [42195 / 1609.344]
    assert visited[0] < BIG // 10, f"the walks visited {visited[0]} objects"
