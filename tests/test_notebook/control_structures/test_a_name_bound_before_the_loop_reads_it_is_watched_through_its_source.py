"""A name a loop binds before reading it is watched through what it is bound from.

What it held before the loop is never handed to anything, and each pass
binds it to a value taken out of other names: a call changing that value
changes them, and they are watched, as a loop target stands for what its
loop walks. A name the loop reads before binding it, or binds in a way
the analysis does not follow, is watched as itself.
"""

from __future__ import annotations

import ast
import random

import pytest

from cash.analysis.mutation_effects import loop_watch_candidates

NAMESPACE = {
    "random": random,
    "sessions": [{"actions": [1, 2]}],
    "rows": [[1]],
    "actions": [9],
    "acc": [0],
}


def _candidates(code: str) -> frozenset[str]:
    node = ast.parse(code).body[0]
    return loop_watch_candidates(node, NAMESPACE, lambda name: name == "random")


@pytest.mark.parametrize(
    "loop, expected",
    [
        ("for s in sessions:\n    actions = s['actions']\n    random.shuffle(actions)", {"sessions"}),
        (
            "for s in sessions:\n    actions = sorted(s['actions'], key=lambda x: -x)\n    random.shuffle(actions)",
            {"sessions"},
        ),
        ("for i in range(3):\n    actions = rows[i]\n    random.shuffle(actions)", {"rows"}),
        ("for i in range(3):\n    actions = [i]\n    random.shuffle(actions)", set()),
        # Read before the loop binds it: what it held before is handed over.
        ("for s in sessions:\n    random.shuffle(actions)\n    actions = s['actions']", {"actions", "sessions"}),
        (
            "for s in sessions:\n    actions = actions + s['actions']\n    random.shuffle(actions)",
            {"actions", "sessions"},
        ),
        ("while actions:\n    actions = s['actions']\n    random.shuffle(actions)", {"actions"}),
        # Bound again by ``:=``: not followed.
        (
            "for s in sessions:\n    actions = s['actions']\n    if (actions := acc):\n        random.shuffle(actions)",
            {"actions", "sessions"},
        ),
    ],
    ids=[
        "an item of the walked list",
        "a sorted copy of one",
        "an item of another list",
        "a new list",
        "read before it is bound",
        "bound from itself",
        "read by the loop's condition",
        "bound again by an assignment expression",
    ],
)
def test_what_a_bound_name_is_watched_through(loop, expected):
    assert _candidates(loop) == expected
