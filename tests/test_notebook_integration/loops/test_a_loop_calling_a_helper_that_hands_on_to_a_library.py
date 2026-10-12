"""A loop calling a helper of the user's that hands its argument, or a
global, to a library function that changes it.

``def push(h, x): heapq.heappush(h, x)``, ``def mix(rows, s):
random.Random(s).shuffle(rows)`` and ``def push(x): heapq.heappush(H, x)``
change what they reach, and nothing in their text says so. Called in a
``while`` loop, a ``for`` loop with ``break`` or a plain ``for`` loop, after
an edit above the cell reading what they changed must not be served its old
value; an unchanged Run All still restores it.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

BADGE = "%cash_badge print"
SLOW = "import time\ndef slow(v):\n    time.sleep(0.25)\n    return v\n"

CASES = {
    "heap-while": (
        "import heapq\ndef push(heap, x):\n    heapq.heappush(heap, x * K)\nK = 1",
        "pending = [5, 3]\nh = []",
        "while pending:\n    push(h, pending.pop())",
        "total = slow(sum(h))\nprint(total)",
        ("K = 1", "K = 2"),
        "8",
        "16",
    ),
    "heap-break": (
        "import heapq\ndef push(heap, x):\n    heapq.heappush(heap, x * K)\nK = 1",
        "pending = [5, 3]\nh = []",
        "for x in pending:\n    if x is None:\n        break\n    push(h, x)",
        "total = slow(sum(h))\nprint(total)",
        ("K = 1", "K = 2"),
        "8",
        "16",
    ),
    "helper-edit": (
        "import heapq\ndef push(heap, x):\n    heapq.heappush(heap, x * K)\nK = 1",
        "pending = [5, 3]\nh = []",
        "while pending:\n    push(h, pending.pop())",
        "total = slow(sum(h))\nprint(total)",
        ("x * K", "x * K + 100"),
        "8",
        "208",
    ),
    "shuffle-while": (
        "import random\ndef mix(rows, seed):\n    random.Random(seed).shuffle(rows)\nSEED = 1",
        "deck = list(range(8))\nrounds = 2",
        "while rounds:\n    mix(deck, SEED)\n    rounds -= 1",
        "top = slow(deck[:3])\nprint(top)",
        ("SEED = 1", "SEED = 2"),
        "[5, 4, 6]",
        "[6, 1, 2]",
    ),
    "global-for": (
        "import heapq\nxs = [1, 2]\ndef push(x):\n    heapq.heappush(H, x)",
        "H = []",
        "for x in xs:\n    push(x)",
        "total = slow(sum(H))\nprint(total)",
        ("xs = [1, 2]", "xs = [1, 2, 5]"),
        "3",
        "8",
    ),
    "global-while": (
        "import heapq\nxs = [1, 2]\ndef push(x):\n    heapq.heappush(H, x)",
        "H = []",
        "i = 0\nwhile i < len(xs):\n    push(xs[i])\n    i += 1",
        "total = slow(sum(H))\nprint(total)",
        ("xs = [1, 2]", "xs = [1, 2, 5]"),
        "3",
        "8",
    ),
    "global-single-call": (
        "import heapq\nxs = [1, 2]\ndef push(x):\n    heapq.heappush(H, x)",
        "H = []",
        "push(xs[-1])",
        "total = slow(sum(H))\nprint(total)",
        ("xs = [1, 2]", "xs = [1, 2, 5]"),
        "2",
        "5",
    ),
    "global-shuffle": (
        "import random\nSEED = 1\ndef reshuffle():\n    random.Random(SEED).shuffle(DECK)",
        "DECK = list(range(8))",
        "reshuffle()",
        "top = slow(DECK[:3])\nprint(top)",
        ("SEED = 1", "SEED = 2"),
        "[3, 6, 1]",
        "[5, 3, 4]",
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_an_edit_above_reaches_the_reader(nb_runner, case):
    first, make, loop, read, (old, new), before, after = CASES[case]
    nb_runner.create_notebook([BADGE, SLOW + first, make, loop, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == before
    nb_runner.set_cell_source(2, SLOW + first.replace(old, new))
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == after, nb_runner.get_raw_output(5)


@pytest.mark.parametrize("case", ["heap-while", "shuffle-while"])
def test_an_unchanged_run_all_restores_the_reader(nb_runner, case):
    first, make, loop, read, _edit, before, _after = CASES[case]
    nb_runner.create_notebook([BADGE, SLOW + first, make, loop, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == before
    assert shows_cached(nb_runner.get_raw_output(5)), nb_runner.get_raw_output(5)
