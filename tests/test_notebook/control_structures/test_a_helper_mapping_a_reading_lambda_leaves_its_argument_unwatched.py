"""A helper that maps a lambda calling nothing over what it is handed only
reads it, so a loop calling the helper does not watch what it iterates.

``def in_row(log): ... list(map(lambda x: x[0], log[1]))`` made the loop
``for log in logs: in_row(log)`` fingerprint ``logs`` before and after it
ran: 87,000 sessions pickled twice, 12 s a run where the loop took 0.3 s.
``map`` and ``filter`` hand each item to the lambda alone. A lambda that
calls something is followed as before, and so is a ``map`` the notebook
bound to its own function.
"""

from __future__ import annotations

import ast
import heapq

import pytest

from cash.analysis.mutation_effects import loop_watch_candidates

SOURCES = {
    "firsts": "def firsts(log):\n    return list(map(lambda x: x[0], log[1]))\n",
    "late": "def late(log):\n    return list(filter(lambda r: r[1] > 3, log[1]))\n",
    "push_all": "def push_all(h, xs):\n    return list(map(lambda v: heapq.heappush(h, v), xs))\n",
    "touch": "def touch(log):\n    return list(map(lambda r: r.append(1), log[1]))\n",
}


def _candidates(loop: str, **namespace) -> frozenset[str]:
    ns = {"heapq": heapq, "logs": [("u", [[1, 2], [3, 4]])], "h": [], **namespace}
    return loop_watch_candidates(ast.parse(loop).body[0], ns, lambda name: name == "heapq", SOURCES.get)


@pytest.mark.parametrize("helper", ["firsts", "late"])
def test_a_reading_lambda_leaves_the_loop_unwatched(helper):
    assert _candidates(f"for log in logs:\n    {helper}(log)") == frozenset()


@pytest.mark.parametrize(
    ("loop", "expected"),
    [
        ("for log in logs:\n    push_all(h, log[1])", {"h", "logs"}),
        ("for log in logs:\n    touch(log)", {"logs"}),
    ],
    ids=["a lambda calling a library", "a lambda calling a method"],
)
def test_a_lambda_that_calls_something_is_still_followed(loop, expected):
    assert _candidates(loop) == expected


def test_a_map_of_the_notebooks_own_is_still_followed():
    assert _candidates("for log in logs:\n    firsts(log)", map=lambda f, xs: [f(x) for x in xs]) == {"logs"}
