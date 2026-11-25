"""A seeded loop that draws, re-run by cash for a cell below, draws the
numbers a run from the top gives it.

The loop cell is not run by hand: the cell below it is, after a restart or
after an edit above, and the upstream check runs the loop again. Before it
does, the global stream goes where a run from the top has it there: the bare
seed above is run again, or the position recorded under the stream's lineage
is put back.

Oracle: the notebook's cells run top to bottom in a plain interpreter.
"""

import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"
HELPER = "import time\ndef f(x):\n    time.sleep(0.12)\n    return x * 2\n"
SETUP = HELPER + "import numpy as np\nnp.random.seed(3)\nitems = [1, 2, 3]"
LOOP = "draws = []\nfor i in items:\n    draws.append(round(float(np.random.rand()), 4) + f(i))"
READER = "r = list(draws)"


def _plain(cells: list[str], name: str = "r") -> str:
    ns: dict = {"__name__": "__main__"}
    for cell in cells:
        exec(cell.replace("time.sleep(0.12)", "pass"), ns)
    return repr(ns[name])


@pytest.mark.fresh_kernel
@pytest.mark.timeout(180)
def test_after_a_restart(nb_runner):
    cells = [SETUP, LOOP, READER]
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = _plain(cells)
    assert nb_runner.peek("r") == expected

    nb_runner.restart()
    nb_runner.run_cells([1, 4])
    assert nb_runner.peek("r") == expected
    assert "RANDOM-" not in nb_runner.get_output(4)


@pytest.mark.timeout(180)
def test_after_an_edit_above(nb_runner):
    cells = [SETUP, LOOP, READER]
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()

    edited = SETUP.replace("[1, 2, 3]", "[1, 2, 3, 4]")
    nb_runner.set_cell_source(2, edited)
    nb_runner.run_cell(4)
    assert nb_runner.peek("r") == _plain([edited, LOOP, READER])
    assert "RANDOM-" not in nb_runner.get_output(4)


@pytest.mark.timeout(180)
def test_the_seed_in_a_cell_of_its_own_and_the_random_module(nb_runner):
    cells = [
        HELPER + "import random",
        "random.seed(8)",
        "items = [1, 2, 3]",
        "draws = []\nfor i in items:\n    draws.append(round(random.random(), 4) + f(i))",
        READER,
    ]
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()

    edited = "items = [1, 2, 3, 4]"
    nb_runner.set_cell_source(4, edited)
    nb_runner.run_cell(6)
    assert nb_runner.peek("r") == _plain([cells[0], cells[1], edited, cells[3], READER])


@pytest.mark.timeout(180)
def test_a_while_loop_is_put_back_too(nb_runner):
    loop = "draws = []\nk = 0\nwhile k < n:\n    draws.append(round(float(np.random.rand()), 4) + f(k))\n    k += 1"
    setup = HELPER + "import numpy as np\nnp.random.seed(5)\nn = 2"
    nb_runner.create_notebook([C_ON, setup, loop, READER])
    nb_runner.start_kernel()
    nb_runner.run_all()

    edited = setup.replace("n = 2", "n = 3")
    nb_runner.set_cell_source(2, edited)
    nb_runner.run_cell(4)
    assert nb_runner.peek("r") == _plain([edited, loop, READER])


@pytest.mark.fresh_kernel
@pytest.mark.timeout(180)
def test_a_statement_above_that_seeds_and_draws_is_run_again_first(nb_runner):
    """The statement that moved the stream last binds a name from a seed and
    a draw: after a restart it runs again ahead of the loop."""
    setup = HELPER + "import numpy as np\nitems = [1, 2, 3]\nfirst = np.random.seed(3) or float(np.random.rand())"
    nb_runner.create_notebook([C_ON, setup, LOOP, READER])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.restart()
    nb_runner.run_cells([1, 4])
    assert nb_runner.peek("r") == _plain([setup, LOOP, READER])
