"""A loop that hands a value to a library function that changes it.

``heapq.heappush(h, i)``, ``if heapq.heappop(h) > 4:``, ``x = dq.popleft()``,
``np.copyto(a, ...)`` and ``random.Random(i).shuffle(xs)`` change what they
are handed. After the loop is edited (``range(3)`` to ``range(4)``) and the
notebook runs again, the cell reading the value must not be served the old
one. Each case is compared with plain Python.

The last test drains a heap and a ``queue.Queue`` in one loop: the queue
makes the loop run again on every run, and the heap must be rebuilt with it,
or the pops are applied twice.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = (
    "import heapq, collections, time, random\nimport numpy as np\n"
    "def slow(x):\n    time.sleep(0.15)\n    return [float(v) for v in x]"
)

CASES = {
    "heappush": (
        "h = []",
        "for i in range(3):\n    heapq.heappush(h, i)",
        "h",
        "[0.0, 1.0, 2.0]",
        "[0.0, 1.0, 2.0, 3.0]",
    ),
    "heappop": (
        "h = [5, 3, 8, 1, 9]; heapq.heapify(h)",
        "for i in range(3):\n    if heapq.heappop(h) > 4:\n        pass",
        "h",
        "[8.0, 9.0]",
        "[9.0]",
    ),
    "popleft": (
        "dq = collections.deque(range(8))",
        "for i in range(3):\n    x = dq.popleft()",
        "dq",
        "[3.0, 4.0, 5.0, 6.0, 7.0]",
        "[4.0, 5.0, 6.0, 7.0]",
    ),
    "copyto": (
        "a = np.zeros(3)",
        "for i in range(3):\n    np.copyto(a, np.full(3, i))",
        "a",
        "[2.0, 2.0, 2.0]",
        "[3.0, 3.0, 3.0]",
    ),
    "shuffle": (
        "xs = list(range(6))",
        "for i in range(3):\n    random.Random(i).shuffle(xs)",
        "xs",
        None,
        None,
    ),
}


def _plain(make: str, loop: str, name: str) -> str:
    ns: dict = {}
    exec("import heapq, collections, random\nimport numpy as np", ns)
    exec(make, ns)
    exec(loop, ns)
    return str([float(v) for v in ns[name]])


@pytest.mark.parametrize("case", list(CASES))
def test_an_edited_loop_reaches_the_reader(nb_runner, case):
    make, loop, name, before, after = CASES[case]
    edited = loop.replace("range(3)", "range(4)")
    before = before or _plain(make, loop, name)
    after = after or _plain(make, edited, name)
    read = f"r = slow({name})\nprint(r)"
    nb_runner.create_notebook([SETUP, make, loop, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == before
    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == after


def test_a_loop_draining_a_heap_and_a_queue(nb_runner):
    cells = [
        "import heapq, queue, time\ndata = [5, 3, 8, 1, 9, 2, 7, 6]\nthr = 4\n"
        "def slow(h, q):\n    time.sleep(0.15)\n    return sorted(h), q.qsize()",
        "h = list(data); heapq.heapify(h)\nq = queue.Queue()\nfor v in data: q.put(v)\nout = []",
        "for i in range(3):\n    if heapq.heappop(h) > thr:\n        out.append(i)\n    if q.get() > thr:\n        out.append(100 + i)",
        "r = slow(h, q)\nprint(r, out)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "([5, 6, 7, 8, 9], 5) [100, 102]"
    nb_runner.set_cell_source(3, cells[2].replace("range(3)", "range(4)"))
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "([6, 7, 8, 9], 4) [100, 102, 3]"
