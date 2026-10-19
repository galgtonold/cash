"""The share check and a statement's analysis are limited by this thread's
CPU time (`CpuLimit`), not the wall clock.

Timed by the wall clock, a busy machine charged a check for the time other
processes ran: a share check of 67 ms idle gave up at its 0.1 s limit in
198 of 200 runs with six busy processes on four cores, and the statement
ran uncached. Waiting is not what a check costs; running is.
"""

from __future__ import annotations

import threading
import time

import pytest

from cash._clock import WALL_STRETCH, CpuLimit
from cash.analysis.analysis_budget import (
    AnalysisBudget,
    AnalysisBudgetExceeded,
    analysis_budget,
    check_analysis_budget,
)
from cash.notebook.shared_objects import WalkBudgetExceeded, check_walk_budget, walk_budget


def _busy(seconds: float) -> None:
    end = time.thread_time() + seconds
    while time.thread_time() < end:
        pass


def test_time_the_thread_spends_waiting_does_not_count():
    limit = CpuLimit(0.05)
    time.sleep(0.15)
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


def test_a_share_check_kept_waiting_runs_on():
    with walk_budget(0.05):
        time.sleep(0.15)
        check_walk_budget()
        _busy(0.1)
        with pytest.raises(WalkBudgetExceeded):
            check_walk_budget()


def test_an_analysis_kept_waiting_runs_on():
    with analysis_budget(AnalysisBudget(0.05)):
        time.sleep(0.15)
        check_analysis_budget()
        _busy(0.1)
        with pytest.raises(AnalysisBudgetExceeded):
            check_analysis_budget()
