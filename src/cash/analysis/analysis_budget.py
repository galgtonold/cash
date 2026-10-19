"""How long cash may spend working out how to key one notebook statement.

Keying a statement means reading what it can run (the callables its calls
are handed, the methods of the objects it passes, `handed_callables`) and,
after it ran, which other variables share its outputs' objects (the share
check). Over a big graph of the user's own objects that reading can cost
far more than the statement: 20,000 agents a ``market.step()`` reaches, a
0.03 s statement analysed for a second. Caching such a statement can never
pay: every run, hit or miss, pays the analysis.

So the analysis has a budget, ``max(ANALYSIS_BUDGET_FLOOR_S,
ANALYSIS_BUDGET_FACTOR * expected run time)``, where the expected run time is
what the statement last took (`CashSession.expected_compute`), counted in
this thread's CPU time (`CpuLimit`). The walks check it as they go (`check_analysis_budget`) and raise
`AnalysisBudgetExceeded` once it is spent. Nothing is skipped in silence:
each place that asks has an answer for "could not tell in time" that is what
it answers for a callee it cannot read -- the statement runs uncached, a
check that would refuse caching refuses it, a writer check says it may
write -- and the statement's outputs are keyed by their value, as a
``# @cash:no-cache`` statement's are (``no_cache_value_digest``): a reader
below hits only when what it reads is the same.

The factor: a hit pays the analysis again, so a statement analysed for
longer than it runs is cheaper run plain. 4, not 1, because a statement
cached keeps the lineages below it settled, so the cells reading it hit too
(the saving is often a few times its own run time), and a run time measured
on another run can be off by that much on a loaded machine. The floor: an
ordinary statement's analysis takes milliseconds, so half a second is never
reached by one, and a first run (no measurement yet) gets that.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from time import perf_counter as _perf_counter

from cash._clock import CpuLimit

__all__ = [
    "ANALYSIS_BUDGET_FACTOR",
    "ANALYSIS_BUDGET_FLOOR_S",
    "ANALYSIS_BUDGET_REASON",
    "AnalysisBudget",
    "AnalysisBudgetExceeded",
    "analysis_budget",
    "analysis_budget_seconds",
    "check_analysis_budget",
    "no_analysis_budget",
]

#: The least a statement's analysis may take, and what one never measured
#: gets.
ANALYSIS_BUDGET_FLOOR_S = 0.5
#: The analysis may take this many times what the statement last took.
ANALYSIS_BUDGET_FACTOR = 4.0


#: Why a statement whose analysis ran out of its budget runs uncached.
ANALYSIS_BUDGET_REASON = (
    "Working out what it can run, through the objects it hands over, took longer than caching it "
    "could save, so it runs every time and what it binds is keyed by its value"
)


class AnalysisBudgetExceeded(Exception):
    """The analysis of the statement in progress spent its budget."""


class AnalysisBudget(CpuLimit):
    """One statement's analysis budget, in this thread's CPU time
    (`CpuLimit`): when it started, when it ends, and whether a walk ran
    past it."""

    __slots__ = ("tripped",)

    def __init__(self, seconds: float) -> None:
        super().__init__(seconds)
        self.tripped = False

    def left(self) -> float:
        """CPU seconds left, never below zero."""
        return max(0.0, self.seconds - self.spent())


_CURRENT: list[AnalysisBudget | None] = [None]


def analysis_budget_seconds(expected: float | None) -> float:
    """The budget of a statement expected to run *expected* seconds (None:
    never measured)."""
    if expected is None or expected <= 0:
        return ANALYSIS_BUDGET_FLOOR_S
    return max(ANALYSIS_BUDGET_FLOOR_S, ANALYSIS_BUDGET_FACTOR * expected)


@contextlib.contextmanager
def analysis_budget(budget: AnalysisBudget) -> Iterator[AnalysisBudget]:
    """Let the walks below run until *budget* is spent; past it,
    `check_analysis_budget` raises. Once tripped it stays tripped: every
    later question of the statement is answered "could not tell" at once."""
    outer = _CURRENT[0]
    _CURRENT[0] = budget
    try:
        yield budget
    finally:
        _CURRENT[0] = outer


@contextlib.contextmanager
def no_analysis_budget() -> Iterator[None]:
    """Run what is below without a budget: the cheap fallbacks a tripped
    analysis still has to answer completely."""
    outer = _CURRENT[0]
    _CURRENT[0] = None
    try:
        yield
    finally:
        _CURRENT[0] = outer


def check_analysis_budget() -> None:
    """Raise `AnalysisBudgetExceeded` when the analysis in progress spent its
    budget; nothing outside `analysis_budget`."""
    budget = _CURRENT[0]
    if budget is None:
        return
    if budget.tripped or (_perf_counter() > budget.deadline and budget.passed()):
        budget.tripped = True
        raise AnalysisBudgetExceeded
