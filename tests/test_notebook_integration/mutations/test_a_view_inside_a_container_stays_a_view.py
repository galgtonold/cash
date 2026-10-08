"""Numpy views held inside a container, and views of an array that is itself
held in a dict, are never restored from the statement cache as detached
copies: on the next Run All, writing through them still changes the base.

``parts = split(a)`` returning ``np.split`` views and
``w = window(data['x'], 2)`` were restored as copies, so ``a`` and
``data['x']`` stayed zeros from the second Run All on (bug-hunt-5 AL-04).
Kernel state is read out of band with ``peek``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = "import cash\n%cash_on\nimport numpy as np, time"
SPLIT = "def split(arr):\n    time.sleep(0.3)\n    return np.split(arr, 2)"
WINDOW = "def window(arr, i):\n    time.sleep(0.3)\n    return arr[i:i+3]"


@pytest.mark.parametrize(
    ("cells", "base", "view", "total"),
    [
        ([SPLIT, "a = np.zeros(10)", "parts = split(a)", "parts[0][:] = 1"], "a", "parts[0]", "5.0"),
        ([WINDOW, "data = {'x': np.zeros(10)}", "w = window(data['x'], 2)", "w[:] = 5"], "data['x']", "w", "15.0"),
    ],
    ids=["list-of-views", "view-of-an-array-in-a-dict"],
)
def test_a_second_run_all_still_writes_into_the_base(nb_runner, cells, base, view, total):
    nb_runner.create_notebook([SETUP, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek(f"{base}.sum()") in (total, f"np.float64({total})")
    assert nb_runner.peek(f"np.shares_memory({base}, {view})") == "True"
