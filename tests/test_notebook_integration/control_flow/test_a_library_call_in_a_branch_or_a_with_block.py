"""A library call that changes what it is handed, written straight inside an
``if``/``else``, a ``try`` or a ``with`` block.

``heapq.heappush(h, K)``, ``np.copyto(a, K)``, ``random.Random(K).shuffle(deck)``
and ``job = dq.popleft()`` change their argument or receiver, and nothing in
the text says so. After an edit above, the cell reading what they changed must
see the new content, as plain Python does: on a Run All, when only the cell
below is run, and when the branch was not taken on the first run. An unchanged
Run All still restores the reader.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

BADGE = "%cash_badge print"
FIRST = (
    "import heapq, contextlib, random, collections, time\nimport numpy as np\nuse_heap = True\nK = 1\n"
    "def slow(v):\n    time.sleep(0.25)\n    return v"
)
MAKE = "h = []\na = np.zeros(2)\ndeck = list(range(6))\ndq = collections.deque([7, 8, 9])"
READ = "total = slow((sum(h), float(a.sum()), deck[:3], list(dq)))\nprint(total)"

BLOCKS = {
    "if": "if use_heap:\n    heapq.heappush(h, K)",
    "if-else": "if K > 100:\n    pass\nelse:\n    heapq.heappush(h, K)",
    "try": "try:\n    heapq.heappush(h, K)\nexcept ValueError:\n    pass",
    "with": "with contextlib.suppress(ValueError):\n    heapq.heappush(h, K)",
    "with-copyto": "with np.errstate(all='ignore'):\n    np.copyto(a, K)",
    "with-shuffle": "with contextlib.suppress(ValueError):\n    random.Random(K).shuffle(deck)",
    "with-a-loop": "with contextlib.suppress(ValueError):\n    for x in range(2):\n        heapq.heappush(h, K + x)",
    "branch-first-not-taken": "if K > 1:\n    job = dq.popleft()",
}


def _plain(block: str, k: int) -> str:
    ns: dict = {}
    exec(FIRST.replace("K = 1", f"K = {k}").replace("time.sleep(0.25)", "pass"), ns)
    exec(MAKE, ns)
    exec(block, ns)
    exec(READ.replace("print(total)", ""), ns)
    return str(ns["total"])


def _notebook(nb_runner, block):
    nb_runner.create_notebook([BADGE, FIRST, MAKE, block, READ])
    nb_runner.start_kernel()


def _value(nb_runner) -> str:
    return nb_runner.get_output(5).strip().splitlines()[0]


@pytest.mark.parametrize("case", list(BLOCKS))
def test_an_edit_above_reaches_the_reader_on_a_run_all(nb_runner, case):
    block = BLOCKS[case]
    _notebook(nb_runner, block)
    nb_runner.run_all()
    assert _value(nb_runner) == _plain(block, 1)
    nb_runner.set_cell_source(2, FIRST.replace("K = 1", "K = 2"))
    nb_runner.run_all()
    assert _value(nb_runner) == _plain(block, 2), nb_runner.get_raw_output(5)


@pytest.mark.parametrize("case", ["if", "try", "with", "branch-first-not-taken"])
def test_an_edit_above_reaches_the_reader_run_on_its_own(nb_runner, case):
    block = BLOCKS[case]
    _notebook(nb_runner, block)
    nb_runner.run_all()
    nb_runner.set_cell_source(2, FIRST.replace("K = 1", "K = 2"))
    nb_runner.run_cell(5)
    assert _value(nb_runner) == _plain(block, 2), nb_runner.get_raw_output(5)


@pytest.mark.parametrize("case", ["if", "with"])
def test_an_unchanged_run_all_restores_the_reader(nb_runner, case):
    _notebook(nb_runner, BLOCKS[case])
    nb_runner.run_all()
    nb_runner.run_all()
    assert _value(nb_runner) == _plain(BLOCKS[case], 1)
    assert shows_cached(nb_runner.get_raw_output(5)), nb_runner.get_raw_output(5)
