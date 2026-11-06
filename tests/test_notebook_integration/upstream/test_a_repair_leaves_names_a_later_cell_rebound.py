"""An upstream repair does not put back names a later cell rebound.

``a, b = split()`` binds two names; a cell below rebinds ``a``
(``a = list(a)``) and a later cell rebinds ``b`` from itself
(``b = b * 10``). Running that last cell again needs ``b`` as ``split()``
left it, so cash repairs it from ``a, b = split()``. The repair put back
``a`` as well, and the cell saw the array ``split()`` made where a plain
run sees the list ``a = list(a)`` made of it. Each test compares the cell's
output with the notebook run from the top in plain Python.
"""

import contextlib
import io

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

SPLIT = (
    "import time, numpy as np\n"
    "def split():\n"
    "    time.sleep(0.05)\n"
    "    return np.arange(4), np.arange(2)\n"
    "def split_lists():\n"
    "    time.sleep(0.05)\n"
    "    return list(range(4)), np.arange(2)\n"
    "def slow_list(x):\n"
    "    time.sleep(0.05)\n"
    "    return list(x)"
)
SHOW = "b = b * 10; print(type(a).__name__, a, b)"


def _plain(cells):
    """What the last of *cells* prints when the notebook runs from the top
    in plain Python."""
    ns: dict = {}
    out = io.StringIO()
    for cell in cells:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(cell, ns)
    return out.getvalue().strip()


def _rerun_last_matches_plain(nb_runner, cells):
    expected = _plain(cells)
    n = len(cells)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(n).strip() == expected
    for _ in range(2):
        nb_runner.run_cell(n)
        assert nb_runner.get_output(n).strip() == expected


def test_the_name_a_later_cell_rebound_keeps_that_value(nb_runner):
    """The repro: ``a = list(a)`` below the unpack, ``b`` repaired for the
    cell rebinding it from itself."""
    cells = [SPLIT, "a, b = split()", "a = list(a)", "b = b * 10; print(type(a).__name__, b)"]
    _rerun_last_matches_plain(nb_runner, cells)
    assert nb_runner.peek("type(a).__name__") == "'list'"
    assert nb_runner.peek("b.tolist()") == "[0, 10]"


def test_a_name_changed_in_place_below_keeps_the_change(nb_runner):
    cells = [SPLIT, "a, b = split_lists()", "a.append(9)", SHOW]
    _rerun_last_matches_plain(nb_runner, cells)
    assert nb_runner.peek("a") == "[0, 1, 2, 3, 9]"


def test_a_rebinding_in_the_unpacking_cell_is_kept(nb_runner):
    cells = [SPLIT, "a, b = split()\na = list(a)", SHOW]
    _rerun_last_matches_plain(nb_runner, cells)
    assert nb_runner.peek("type(a).__name__") == "'list'"


@pytest.mark.parametrize("rebind", ["a = list(a)", "a = slow_list(a)"], ids=["replayed", "restored"])
def test_both_names_needed_are_rebuilt_in_order(nb_runner, rebind):
    """The cell rebinds both names from themselves: ``a`` needs the
    rebinding below the unpack again, which needs ``a`` as the unpack left
    it, whether that rebinding runs again or is restored itself."""
    cells = [SPLIT, "a, b = split()", rebind, "a = a + ['x']; b = b * 10; print(type(a).__name__, a, b)"]
    _rerun_last_matches_plain(nb_runner, cells)
    assert nb_runner.peek("a") == "[np.int64(0), np.int64(1), np.int64(2), np.int64(3), 'x']"


def test_a_cell_rebinding_both_names_from_themselves(nb_runner):
    cells = [SPLIT, "a, b = split()", "a = list(a); b = b * 10; print(type(a).__name__, a, b)"]
    _rerun_last_matches_plain(nb_runner, cells)
