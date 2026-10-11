"""A loop run again to refill a drained queue rebuilds what else it changes.

``for i in range(3): heapq.heappop(h); q.get()`` drains ``q`` and pops ``h``.
When the queue is found drained and the loop is scheduled to refill it, ``h``
is rebuilt with it, from the statement that made it through every later one
changing it, or the pops would be applied twice. When such a name has no
statement that makes it in the trace, the loop is not scheduled at all.
"""

from __future__ import annotations

from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner

LOOP = "for i in range(3):\n    heapq.heappop(h)\n    out.append(q.get())"


def _entry(code: str, outputs: set[str], inputs: set[str]) -> TraceEntry:
    return TraceEntry(code, outputs, inputs, {}, {}, False)


def _schedule(trace: list[TraceEntry], scheduled: list[int]) -> list[int]:
    planner = object.__new__(ReexecutionPlanner)
    return planner._schedule_consumable_producer_touches(scheduled, trace, {"q"})


def test_the_heap_is_rebuilt_with_the_loop():
    trace = [
        _entry("h = list(data)", {"h"}, {"data"}),
        _entry("heapq.heapify(h)", {"h"}, {"h", "heapq"}),
        _entry("q = queue.Queue()", {"q"}, {"queue"}),
        _entry("out = []", {"out"}, set()),
        _entry(LOOP, {"i", "h", "q", "out"}, {"h", "q", "out", "heapq"}),
    ]
    assert _schedule(trace, [2]) == [0, 1, 2, 3, 4]


def test_a_heap_nothing_makes_leaves_the_loop_out():
    trace = [
        _entry("q = queue.Queue()", {"q"}, {"queue"}),
        _entry(LOOP, {"i", "h", "q", "out"}, {"h", "q", "out", "heapq"}),
    ]
    assert _schedule(trace, [0]) == [0]
