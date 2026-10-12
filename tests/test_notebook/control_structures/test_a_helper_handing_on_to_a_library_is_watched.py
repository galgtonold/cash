"""A function of the user's that hands what it is given, or a global, to a
library function is watched as the library call itself would be.

``def push(h, x): heapq.heappush(h, x)`` changes ``h`` and
``def reshuffle(): random.Random(s).shuffle(DECK)`` changes ``DECK``, and
nothing in either text says so. A loop calling one watches the argument or
the global around the loop; a statement calling one fingerprints the global
as it does the arguments it hands over. A helper that only reads keeps both
out of the watch.
"""

from __future__ import annotations

import ast
import heapq
import random

import pytest

from cash.analysis.callee_effects import resolve_function_def
from cash.analysis.mutation_effects import loop_watch_candidates
from cash.analysis.namespace_effects import call_arguments

SOURCES = {
    "push": "def push(heap, x):\n    heapq.heappush(heap, x)\n",
    "push_kw": "def push_kw(x, *, heap):\n    heapq.heappush(heap, x)\n",
    "outer": "def outer(q, x):\n    push(q, x)\n",
    "mix": "def mix(rows, seed):\n    random.Random(seed).shuffle(rows)\n",
    "push_global": "def push_global(x):\n    heapq.heappush(H, x)\n",
    "reshuffle": "def reshuffle():\n    random.Random(SEED).shuffle(DECK)\n",
    "total": "def total(rows):\n    return sum(sum(r) for r in rows)\n",
    "count": "def count(rows):\n    return len(rows) + len(H)\n",
    "rec": "def rec(h, n):\n    if n:\n        rec(h, n - 1)\n    heapq.heappush(h, n)\n",
}

NAMESPACE = {"heapq": heapq, "random": random, "h": [], "H": [], "DECK": [1, 2, 3], "SEED": 1, "xs": [3, 1], "n": 2}


def _is_module(name: str) -> bool:
    return name in {"heapq", "random"}


def _loop(code: str) -> frozenset[str]:
    return loop_watch_candidates(ast.parse(code).body[0], NAMESPACE, _is_module, SOURCES.get)


@pytest.mark.parametrize(
    "loop, expected",
    [
        ("while xs:\n    push(h, xs.pop())", {"h", "xs"}),
        ("for x in xs:\n    if x is None:\n        break\n    push(h, x)", {"h"}),
        ("while n:\n    push_kw(n, heap=h)\n    n -= 1", {"h"}),
        ("while xs:\n    outer(h, xs.pop())", {"h", "xs"}),
        ("while n:\n    mix(xs, n)\n    n -= 1", {"xs"}),
        ("for i in range(2):\n    push_global(i)", {"H"}),
        ("while n:\n    reshuffle()\n    n -= 1", {"DECK"}),
        ("while n:\n    list(map(lambda v: push(h, v), [1]))\n    n -= 1", {"h"}),
        ("while n:\n    rec(h, 2)\n    n -= 1", {"h"}),
    ],
    ids=[
        "while",
        "for-break",
        "keyword",
        "through-a-helper",
        "shuffle",
        "global",
        "global-shuffle",
        "lambda",
        "recursive",
    ],
)
def test_a_loop_watches_what_a_helper_hands_on(loop, expected):
    assert expected <= _loop(loop)


@pytest.mark.parametrize(
    "loop",
    ["while xs:\n    s = total([xs])\n    xs.pop()", "for i in range(2):\n    c = count(xs)"],
    ids=["reads-its-argument", "reads-a-global"],
)
def test_a_reading_helper_adds_nothing_to_the_watch(loop):
    assert not _loop(loop) & {"h", "H", "DECK"}


def test_what_a_helper_hands_on_is_split_into_parameters_and_globals():
    from cash.analysis.namespace_effects import function_handoffs

    fdef = resolve_function_def("reshuffle", SOURCES.get)
    # SEED (an int) is handed too, but no call changes an int in place.
    assert function_handoffs(fdef, NAMESPACE, SOURCES.get) == (frozenset(), frozenset({"DECK"}))
    fdef = resolve_function_def("outer", SOURCES.get)
    assert function_handoffs(fdef, NAMESPACE, SOURCES.get)[0] == {"q", "x"}


@pytest.mark.parametrize(
    "statement, global_name",
    [("push_global(xs[-1])", "H"), ("reshuffle()", "DECK"), ("r = [push_global(x) for x in [1]]", "H")],
)
def test_a_statement_watches_the_global_a_helper_hands_on(statement, global_name):
    assert global_name in call_arguments(ast.parse(statement), NAMESPACE, SOURCES.get)


def test_a_statement_calling_a_reading_helper_watches_no_global():
    assert "H" not in call_arguments(ast.parse("c = count(xs)"), NAMESPACE, SOURCES.get)
