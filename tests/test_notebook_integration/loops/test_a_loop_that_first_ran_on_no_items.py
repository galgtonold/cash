"""A loop that hands a list to a library call, first run on no items.

``while pending: heapq.heappush(h, pending.pop())`` changes nothing when
``pending`` is empty, and that run is no verdict on the next one: once items
appear above, the cell reading ``h`` must see them, on a Run All and when
only the cell below is run. A loop with ``break``, one with ``else`` and a
long plain loop (run as one unit) alike. An unchanged Run All still restores
the reader.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

BADGE = "%cash_badge print"
FIRST = "import heapq, time\ndef slow(v):\n    time.sleep(0.25)\n    return v\npending = []"
LOOPS = {
    "while": "while pending:\n    heapq.heappush(h, pending.pop())",
    "for-break": "for x in pending:\n    if x < 0:\n        break\n    heapq.heappush(h, x)",
    "for-else": "for x in pending:\n    heapq.heappush(h, x)\nelse:\n    done = True",
    "long-for": "for x in pending:\n    heapq.heappush(h, x)",
}
READ = "total = slow((sum(h), len(h)))\nprint(total)"


def _filled(case: str) -> str:
    items = "list(range(5000))" if case == "long-for" else "[5, 3, 9]"
    return FIRST.replace("pending = []", f"pending = {items}")


def _expected(case: str) -> str:
    return "(12497500, 5000)" if case == "long-for" else "(17, 3)"


def _notebook(nb_runner, case):
    nb_runner.create_notebook([BADGE, FIRST, "h = []", LOOPS[case], READ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == "(0, 0)"


@pytest.mark.parametrize("case", list(LOOPS))
def test_items_appearing_above_reach_the_reader_on_a_run_all(nb_runner, case):
    _notebook(nb_runner, case)
    nb_runner.set_cell_source(2, _filled(case))
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == _expected(case), nb_runner.get_raw_output(5)


@pytest.mark.parametrize("case", ["while", "for-break"])
def test_items_appearing_above_reach_the_reader_run_on_its_own(nb_runner, case):
    _notebook(nb_runner, case)
    nb_runner.set_cell_source(2, _filled(case))
    nb_runner.run_cell(5)
    assert nb_runner.get_output(5).strip().splitlines()[0] == _expected(case), nb_runner.get_raw_output(5)


@pytest.mark.parametrize("case", ["while", "for-break"])
def test_an_unchanged_rerun_restores_the_reader(nb_runner, case):
    _notebook(nb_runner, case)
    nb_runner.set_cell_source(2, _filled(case))
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == _expected(case)
    assert shows_cached(nb_runner.get_raw_output(5)), nb_runner.get_raw_output(5)
    nb_runner.run_cell(5)
    assert shows_cached(nb_runner.get_raw_output(5)), nb_runner.get_raw_output(5)
