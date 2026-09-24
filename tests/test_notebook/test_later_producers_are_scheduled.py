"""A re-run producer of a variable brings every later producer of it along.

The plan re-ran ``results = {}`` and the functions reading
``results``, but not the ``for`` loop below the init that fills it; ``results``
was left empty in the kernel and the report raised ``UpstreamStateError:
'logreg'``. The trigger (functions counted as built on an older ``results``)
is fixed in 8e3444e; this guards what it exposed. Once a statement that binds
or writes ``v`` re-runs, ``v`` holds the state right after it -- unless every
later statement that also writes ``v`` re-runs too. The backward completion
only asks for the producer BEFORE a reader, and ``results = {}`` is one.
"""

from __future__ import annotations

import types

from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner


def _planner(user_ns: dict) -> ReexecutionPlanner:
    vl = types.SimpleNamespace(shell=types.SimpleNamespace(user_ns=user_ns), tracking_state=TrackingState())
    return ReexecutionPlanner(vl, classifier=None)


def _entry(stmt, outputs=(), inputs=()):
    return TraceEntry(stmt, set(outputs), set(inputs), {}, {}, False)


TRACE = [
    _entry("families = ['logreg', 'forest']", ("families",)),  # 0
    _entry("results = {}", ("results",)),  # 1
    _entry(
        "for name in families:\n    results[name] = evaluate(name)",
        ("results", "name"),
        ("families", "results", "evaluate"),
    ),  # 2
    _entry("comparison = summarize(results)", ("comparison",), ("results", "summarize")),  # 3
    _entry("def draw_best():\n    return max(results)", ("draw_best",), ("results",)),  # 4
    _entry("report = draw_best()", ("report",), ("draw_best",)),  # 5
]


def test_rerunning_an_accumulator_init_reruns_the_loop_that_fills_it():
    planner = _planner(
        {"families": [], "results": {}, "evaluate": len, "summarize": len, "draw_best": len, "comparison": 0}
    )
    scheduled = planner.complete_later_producers([1, 4, 5], TRACE)
    assert 2 in scheduled, "results = {} was re-run without the loop that fills it"
    assert 3 not in scheduled, "a reader that is not a producer was dragged in"


def test_a_last_producer_needs_nothing_after_it():
    planner = _planner({"results": {}})
    assert planner.complete_later_producers([2, 4], TRACE) == [2, 4]


# What actually put `results = {}` in that plan (traced on the original repro):
# the accumulator-init pass decides a loop accumulator is being fully re-run
# when a scheduled statement's TEXT matches `results.<method>(` or
# `results[...] =`. `def draw_roc` iterating `results.items()` matched, so the
# init was scheduled to stop the loop's writes doubling -- and the loop was
# never scheduled at all.
from cash.notebook.upstream.loop_rules import LoopRules  # noqa: E402


def test_a_function_reading_an_accumulator_is_not_a_rerun_of_its_loop():
    trace = [
        _entry("results = {}", ("results",)),
        _entry(
            "for name in families:\n    results[name] = evaluate(name)", ("results", "name"), ("families", "results")
        ),
        _entry(
            "def draw_roc(ax):\n    for name, r in results.items():\n        ax.plot(r)", ("draw_roc",), ("results",)
        ),
    ]
    fully = LoopRules._loop_vars_fully_rescheduled(None, [2], trace, {"results"})
    assert fully == set(), "reading results.items() was taken for re-running the loop"
    assert LoopRules._loop_vars_fully_rescheduled(None, [1], trace, {"results"}) == {"results"}
