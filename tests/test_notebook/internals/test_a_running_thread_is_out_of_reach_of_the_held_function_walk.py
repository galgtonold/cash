"""The threads alive as a cell starts are not held by cash while it runs.

The check for a thread a cell leaves running (NOTEBOOK-THREAD-ALIVE) kept
the ``Thread`` objects alive at the cell's start in a set until the cell
ended. The walk for the notebook functions a value holds reached that set
through the cell executor, and from each thread whatever its target and
arguments hold: under pytest, pytest-timeout's timer thread and through it
every collected test item. Each walk visited about 676,000 objects instead
of about 1,000, and a test calling the same lambdas 63 times timed out on
Windows. The set now holds thread idents.

The walk starts at the check's own state, so what the rest of the process
holds (other threads, the test run) does not change the count.
"""

from __future__ import annotations

import threading

import pytest

from cash.notebook import held_functions
from cash.notebook.background_threads import BackgroundThreads

#: The objects the thread's argument holds: far more than the walk below visits.
BIG = 200_000


class _Record:
    def __init__(self, i: int):
        self.i = i


@pytest.fixture
def thread_holding_a_big_graph():
    """A running thread whose argument holds BIG objects."""
    stop = threading.Event()
    graph = [_Record(i) for i in range(BIG)]
    thread = threading.Thread(target=lambda event, held: event.wait(), args=(stop, graph), daemon=True)
    thread.start()
    yield
    stop.set()
    thread.join()


def _notebook() -> dict:
    """A namespace with a notebook function in it, so the walk goes to the
    end rather than stopping once it finds none can exist."""
    namespace: dict = {"__name__": "__main__"}
    exec("def make(x):\n    return x\n", namespace)
    return namespace


def _visited(monkeypatch, value, namespace) -> int:
    seen = [0]
    real = held_functions._visit_group

    def counting(kind, group, *args):
        seen[0] += len(group)
        return real(kind, group, *args)

    monkeypatch.setattr(held_functions, "_visit_group", counting)
    held_functions.held_notebook_code(value, namespace)
    monkeypatch.setattr(held_functions, "_visit_group", real)
    return seen[0]


def test_what_a_cell_start_records_does_not_reach_a_running_thread(monkeypatch, thread_holding_a_big_graph):
    threads = BackgroundThreads()
    threads.cell_starts()

    visited = _visited(monkeypatch, threads, _notebook())

    assert visited < 1_000, f"the walk from the cell-start record visited {visited} objects"
