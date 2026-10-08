"""An expensive notebook function that returns a numpy view of an array it
was given, or of a global array, still returns a view of that array on the
next Run All and after a restart: writing through it changes the base.

The call cache used to serve such a call as an array of its own, so
``w = window(a, 2)`` then ``w[:] = 5`` left ``a`` at zeros from the second run
on (bug-hunt-5 AL-03). Kernel state is read out of band with ``peek``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SETUP = "import cash\n%cash_on\nimport numpy as np, time"


@pytest.mark.fresh_kernel
def test_a_view_of_an_argument_writes_into_it_after_a_restart(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "def window(arr, i):\n    time.sleep(0.3)\n    return arr[i:i+3]",
            "a = np.zeros(10)",
            "w = window(a, 2)",
            "w[:] = 5",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("a.sum()") in ("15.0", "np.float64(15.0)")
    assert nb_runner.peek("np.shares_memory(a, w)") == "True"


def test_a_view_of_a_global_writes_into_it_on_the_next_run_all(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "BIG = np.zeros(10)",
            "def win(i):\n    time.sleep(0.3)\n    return BIG[i:i+3]",
            "w = win(2)",
            "w[:] = 5",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("BIG.sum()") in ("15.0", "np.float64(15.0)")
    assert nb_runner.peek("np.shares_memory(BIG, w)") == "True"
