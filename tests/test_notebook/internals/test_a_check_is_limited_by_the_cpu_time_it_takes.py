"""The share check and a statement's analysis are limited by this thread's
CPU time (`CpuLimit`), not the wall clock.

Timed by the wall clock, a busy machine charged a check for the time other
processes ran: a share check of 67 ms idle gave up at its 0.1 s limit in
198 of 200 runs with six busy processes on four cores, and the statement
ran uncached. Waiting is not what a check costs; running is.

A limit kept waiting still ends after `WALL_STRETCH` times its seconds of
wall time. The tests of a waiting check therefore run on a stand-in wall
clock and CPU clock (`clocks`): slept on a real one, a loaded CI runner
(macOS, 3.10) once kept a 0.15 s sleep waiting past the 0.5 s end of a
0.05 s limit, and the analysis gave up as it should after that long.
"""

from __future__ import annotations

import threading
import time

import pytest

from cash import _clock
from cash.analysis import analysis_budget as budget_module
from cash.analysis.analysis_budget import (
    AnalysisBudget,
    AnalysisBudgetExceeded,
    analysis_budget,
    check_analysis_budget,
)
from cash.notebook import shared_objects
from cash.notebook.shared_objects import WalkBudgetExceeded, check_walk_budget, walk_budget

WALL_STRETCH, CpuLimit = _clock.WALL_STRETCH, _clock.CpuLimit


def _busy(seconds: float) -> None:
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


class _Clocks:
    """A wall clock and this thread's CPU clock that move only when told."""

    def __init__(self) -> None:
        self.wall = 1000.0
        self.cpu = 0.0

    def wait(self, seconds: float) -> None:
        self.wall += seconds

    def run(self, seconds: float) -> None:
        self.wall += seconds
        self.cpu += seconds


@pytest.fixture
def clocks(monkeypatch):
    c = _Clocks()
    monkeypatch.setattr(_clock, "_REAL", (lambda: c.wall,))
    monkeypatch.setattr(_clock, "_thread_time", lambda: c.cpu)
    # The checks read the wall clock first, by their own name for it.
    monkeypatch.setattr(budget_module, "_perf_counter", lambda: c.wall)
    monkeypatch.setattr(shared_objects, "_perf_counter", lambda: c.wall)
    return c


def test_a_sleeping_thread_spends_no_cpu_time():
    limit = CpuLimit(0.05)
    time.sleep(0.15)
    assert limit.spent() < 0.05, "this thread's CPU clock counted the time it slept"


def test_time_the_thread_spends_waiting_does_not_count(clocks):
    limit = CpuLimit(0.05)
    clocks.wait(0.15)
    assert not limit.passed()
    assert limit.deadline > limit.started + 0.15, "the deadline moved on by what is left"


def test_time_the_thread_spends_running_counts():
    limit = CpuLimit(0.05)
    _busy(0.1)
    assert limit.passed()


def test_a_limit_kept_waiting_still_ends():
    limit = CpuLimit(0.01)
    time.sleep(0.01 * WALL_STRETCH + 0.05)
    assert limit.passed()


def test_a_zero_limit_is_spent_at_once():
    limit = CpuLimit(0.0)
    _busy(0.001)
    assert limit.passed()


def test_asked_from_another_thread_the_wall_clock_answers():
    limit = CpuLimit(0.05)
    time.sleep(0.1)
    answers = []
    thread = threading.Thread(target=lambda: answers.append(limit.passed()))
    thread.start()
    thread.join()
    assert answers == [True]


def test_a_share_check_kept_waiting_runs_on(clocks):
    with walk_budget(0.05):
        clocks.wait(0.15)
        check_walk_budget()
        clocks.run(0.1)
        with pytest.raises(WalkBudgetExceeded):
            check_walk_budget()


def test_an_analysis_kept_waiting_runs_on(clocks):
    with analysis_budget(AnalysisBudget(0.05)):
        clocks.wait(0.15)
        check_analysis_budget()
        clocks.run(0.1)
        with pytest.raises(AnalysisBudgetExceeded):
            check_analysis_budget()


def test_an_analysis_kept_waiting_past_the_stretch_ends(clocks):
    with analysis_budget(AnalysisBudget(0.05)):
        clocks.wait(0.05 * WALL_STRETCH + 0.01)
        with pytest.raises(AnalysisBudgetExceeded):
            check_analysis_budget()
