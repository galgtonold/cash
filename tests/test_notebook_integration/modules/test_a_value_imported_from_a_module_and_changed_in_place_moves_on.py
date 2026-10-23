"""A value imported from a local module and changed in place by a cell
(``D.pop(t)``, ``heapq.heappop(H)``, ``R.random()``) is the same object the
next Run All's import hands back, so each Run All goes on from where the
last one left it, as plain Python does, also when cash stores even the
cheapest statement.

With every statement stored, the readers were keyed on the import alone,
which did not move: ``t = heapq.heappop(H)`` was served 0 on every Run
All, ``t = next(iter(D))`` served ``'a'`` and the ``D.pop(t)`` after it
raised KeyError.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

PREAMBLE = "import sys, os\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\n"

CASES = {
    "a dict popped": (
        "hd_iter.py",
        "D = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}\n",
        [PREAMBLE + "from hd_iter import D", "t = next(iter(D))", "D.pop(t)"],
        ["'a'", "'b'", "'c'"],
    ),
    "a heap popped": (
        "hh_heap.py",
        "H = list(range(100))\n",
        [PREAMBLE + "import heapq\nfrom hh_heap import H", "t = heapq.heappop(H)", "u = t + 1"],
        ["0", "1", "2"],
    ),
    "a seeded random instance drawn from": (
        "hr_rng.py",
        "import random\nR = random.Random(0)\n",
        [PREAMBLE + "from hr_rng import R", "t = R.random()", "u = round(t * 1000)"],
        None,
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_each_run_all_goes_on_from_the_last(nb_runner, tmp_path, case):
    filename, source, cells, expected = CASES[case]
    (tmp_path / filename).write_text(source, encoding="utf-8")
    if expected is None:  # what plain Python draws from the same seed
        import random

        r = random.Random(0)
        expected = [repr(r.random()) for _ in range(3)]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    got = []
    for _ in expected:
        nb_runner.run_all()
        got.append(nb_runner.peek("t"))
    assert got == expected
