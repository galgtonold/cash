"""A loop that hands a name to a library function is watched for what it changes.

``heapq.heappush(h, x)``, ``np.copyto(a, b)``, ``rng.shuffle(xs)``,
``if heapq.heappop(h) > t:`` and ``x = dq.popleft()`` change what they are
handed, and nothing in the text tells them from a call that only reads it.
The names are fingerprinted around the loop the first time its source runs,
and what changed is recorded for the loop's lineage update and the simulation.
After the loop is edited, a cell reading what it changed is not served the
value the old loop left.
"""

from __future__ import annotations

import ast
import collections
import heapq
import random

import numpy as np
import pytest

from cash.notebook.upstream import NotebookSimulator
from tests._cell_driver import run_cash_cell

NAMESPACE = {
    "heapq": heapq,
    "np": np,
    "random": random,
    "collections": collections,
    "h": [],
    "dq": collections.deque(range(8)),
    "a": np.zeros(3),
    "xs": [1, 2, 3],
    "lists": [[1, 2], [3, 4]],
    "acc": [],
    "n": 3,
}


def _candidates(code: str) -> frozenset[str]:
    from cash.analysis.mutation_effects import loop_watch_candidates

    node = ast.parse(code).body[0]
    return loop_watch_candidates(node, NAMESPACE, lambda name: name in {"heapq", "np", "random", "collections"})


@pytest.mark.parametrize(
    "loop, expected",
    [
        ("for i in range(3):\n    heapq.heappush(h, i)", {"h"}),
        ("for i in range(3):\n    if heapq.heappop(h) > 4:\n        print(i)", {"h"}),
        ("for i in range(3):\n    x = dq.popleft()", {"dq"}),
        ("for i in range(3):\n    np.copyto(a, np.full(3, i))", {"a"}),
        ("for i in range(3):\n    random.Random(i).shuffle(xs)", {"xs"}),
        # A loop target hands over items of what the loop walks.
        ("for row in lists:\n    random.shuffle(row)", {"lists"}),
    ],
    ids=["heappush", "heappop in a condition", "popleft", "copyto", "shuffle", "shuffle of an item"],
)
def test_what_a_loop_hands_to_a_library_is_watched(loop, expected):
    assert _candidates(loop) == expected


@pytest.mark.parametrize(
    "loop",
    [
        "for i in range(3):\n    print(xs, len(xs), sum(xs))",
        "for i in range(3):\n    acc.append(xs)",
        "for i in range(3):\n    heapq.heappush(h, n)\n    m = np.mean(a)",
        "for i in range(3):\n    def f(v):\n        heapq.heappush(v, 1)",
    ],
    ids=["reading builtins", "a list storing what it is handed", "an int and a read-only library call", "a def"],
)
def test_what_only_reads_or_cannot_change_is_not_watched(loop):
    # ``h`` in the third loop is watched; ``n``, an int, and ``a``, read by
    # ``np.mean``, are not.
    assert _candidates(loop) <= {"h"}


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


SETUP = "import heapq\nh = []"
LOOP = "for i in range({n}):\n    heapq.heappush(h, i)"
READ = "r = list(h)"


@pytest.mark.usefixtures("store_everything")
def test_an_edited_loop_is_not_served_the_old_heap(cash_magics, mock_shell):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [SETUP, LOOP.format(n=3), READ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["r"] == [0, 1, 2]

    cells[1] = LOOP.format(n=4)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["r"] == [0, 1, 2, 3]
    assert mock_shell.user_ns["h"] == [0, 1, 2, 3]


@pytest.mark.usefixtures("store_everything")
def test_an_unchanged_loop_still_restores_what_reads_it(cash_magics, mock_shell):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [SETUP, LOOP.format(n=3), READ]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["r"] == [0, 1, 2]
    assert mock_shell.user_ns["h"] == [0, 1, 2]


def test_the_simulation_applies_what_the_runtime_saw(cash_magics, cash_instance, mock_shell):
    from cash.notebook.loop_watch import loop_watch_hash

    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(cash_magics, "import heapq\nimport numpy as np\nh = []\nmean_of = []")
    loop = "for i in range(3):\n    heapq.heappush(h, i)\n    x = np.zeros(2)\n    np.mean(x)"
    run_cash_cell(cash_magics, loop)

    verdict = cash_magics.tracking_state.mutation_verdicts.get(loop_watch_hash(loop.strip()))
    assert verdict == {"h"}
    simulation = NotebookSimulator(mock_shell, cash_instance, cash_magics.tracking_state).virtual_lineage
    simulated = simulation.controls.collect_mutations(ast.parse(loop).body[0], set(), set(), set())
    assert "h" in simulated


def test_a_method_of_a_name_the_loop_binds_is_judged_when_it_runs():
    """``ax`` in ``for ax, col in zip(axes, cols): ax.plot(emb[col])`` is not
    bound before the loop runs. Read as a name that would fail before its
    arguments are evaluated, ``emb`` went unwatched, was taken as changed,
    and a cell reading it re-drew a chart instead of naming it stale."""
    namespace = {**NAMESPACE, "axes": [object(), object()], "emb": {"a": [1], "b": [2]}}
    from cash.analysis.mutation_effects import loop_watch_candidates

    node = ast.parse("for ax, col in zip(axes, ['a', 'b']):\n    ax.plot(emb[col])").body[0]
    assert "emb" in loop_watch_candidates(node, namespace, lambda name: name in {"heapq", "np"})
