"""A statement that hands a list or an array to a library call that changes
it, whose first run changed nothing.

``[heapq.heappush(h, x) for x in pending]`` (and the ``map``/``lambda`` and
generator forms) over an empty ``pending``, or ``np.copyto(a, src)`` with
``src`` equal to ``a``, leave the object as it was. That run is no verdict on
the next: once the input above changes, the cell reading the object must see
what the statement did, on a Run All and when only the cell below is run. An
unchanged Run All still restores the reader.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

BADGE = "%cash_badge print"
SLOW = "import time\ndef slow(v):\n    time.sleep(0.25)\n    return v\n"
HEAP = ("import heapq\npending = []", "h = []", "total = slow(sum(h) * 100 + len(h))\nprint(total)")
COPY = ("import numpy as np\nsrc = np.zeros(4)", "a = np.zeros(4)", "total = slow(float(a.sum()))\nprint(total)")
CASES = {
    "comprehension": (HEAP, "_ = [heapq.heappush(h, x) for x in pending]", ("[]", "[5, 3, 9]"), "0", "1703"),
    "map-lambda": (HEAP, "list(map(lambda x: heapq.heappush(h, x), pending))", ("[]", "[5, 3, 9]"), "0", "1703"),
    "generator": (HEAP, "any(heapq.heappush(h, x) for x in pending)", ("[]", "[5, 3, 9]"), "0", "1703"),
    "copyto": (COPY, "np.copyto(a, src)", ("np.zeros(4)", "np.ones(4)"), "0.0", "4.0"),
}


def _notebook(nb_runner, case):
    (first, make, read), statement, _edit, before, _after = CASES[case]
    nb_runner.create_notebook([BADGE, SLOW + first, make, statement, read])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _value(nb_runner) == before


def _edit(nb_runner, case):
    (first, _make, _read), _statement, (old, new), _before, _after = CASES[case]
    nb_runner.set_cell_source(2, SLOW + first.replace(old, new))


def _value(nb_runner) -> str:
    return nb_runner.get_output(5).strip().splitlines()[0]


@pytest.mark.parametrize("case", list(CASES))
def test_a_changed_input_reaches_the_reader_on_a_run_all(nb_runner, case):
    _notebook(nb_runner, case)
    _edit(nb_runner, case)
    nb_runner.run_all()
    assert _value(nb_runner) == CASES[case][4], nb_runner.get_raw_output(5)


@pytest.mark.parametrize("case", ["comprehension", "copyto"])
def test_a_changed_input_reaches_the_reader_run_on_its_own(nb_runner, case):
    _notebook(nb_runner, case)
    _edit(nb_runner, case)
    nb_runner.run_cell(5)
    assert _value(nb_runner) == CASES[case][4], nb_runner.get_raw_output(5)


@pytest.mark.parametrize("case", ["comprehension", "copyto"])
def test_an_unchanged_run_all_restores_the_reader(nb_runner, case):
    _notebook(nb_runner, case)
    nb_runner.run_all()
    assert _value(nb_runner) == CASES[case][3]
    assert shows_cached(nb_runner.get_raw_output(5)), nb_runner.get_raw_output(5)
