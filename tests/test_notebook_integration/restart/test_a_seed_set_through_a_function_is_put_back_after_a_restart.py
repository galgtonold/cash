"""A seed set through a function of the notebook (``set_seed(5)``) holds for
a cell run on its own after a restart.

The new kernel has run neither the function nor the call. Before the cell
draws, cash runs the call again when the function only seeds, so the cell
draws what Run All gives it. When the function does more than seed, cash
does not run it for the stream alone and says the stream was not put back
(``RANDOM-STREAM-UNRESTORED``); it does not call the draw unseeded.

Oracle: the cells run top to bottom in a plain interpreter.
"""

import pytest

pytestmark = [pytest.mark.libraries, pytest.mark.fresh_kernel]

C_ON = "import cash\n%cash_on"
SEEDER = "import numpy as np, random, time\ndef set_seed(s):\n    random.seed(s)\n    np.random.seed(s)\nset_seed(5)"
SLOW = "time.sleep(0.3)\nn = 3"
DRAW = "r = np.random.rand(n).round(4).tolist()"


def _plain(cells: list[str], name: str = "r") -> str:
    ns: dict = {"__name__": "__main__"}
    for cell in cells:
        exec(cell.replace("time.sleep(0.3)", "pass").replace("time.sleep(0.12)", "pass"), ns)
    return repr(ns[name])


@pytest.mark.timeout(180)
def test_the_draw_below_gets_the_run_all_numbers(nb_runner):
    cells = [SEEDER, SLOW, DRAW]
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = _plain(cells)
    assert nb_runner.peek("r") == expected

    nb_runner.restart()
    nb_runner.run_cells([1, 4])
    assert nb_runner.peek("r") == expected
    assert "RANDOM-" not in nb_runner.get_output(4)
    # And the kernel now knows the stream as seeded: a second run agrees.
    nb_runner.run_cell(4)
    assert nb_runner.peek("r") == expected
    assert "RANDOM-" not in nb_runner.get_output(4)


@pytest.mark.timeout(180)
def test_a_seeder_that_seeds_through_another_function(nb_runner):
    seeder = (
        "import random, time\nimport numpy as np\n"
        "def seed_numpy(s):\n"
        "    np.random.seed(s)\n"
        "def set_seed(s=0):\n"
        '    """Seed every stream."""\n'
        "    random.seed(s)\n"
        "    if np.__version__:\n"
        "        seed_numpy(s)\n"
        "SEED = 11\n"
        "set_seed(SEED)"
    )
    draw = "r = [round(random.random(), 4), *np.random.rand(n).round(4).tolist()]"
    cells = [seeder, SLOW, draw]
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = _plain(cells)

    nb_runner.restart()
    nb_runner.run_cells([1, 4])
    assert nb_runner.peek("r") == expected
    assert "RANDOM-" not in nb_runner.get_output(4)


@pytest.mark.timeout(180)
def test_a_seeded_loop_below_gets_the_run_all_numbers(nb_runner):
    helper = "def f(x):\n    time.sleep(0.12)\n    return x * 2"
    loop = "items = [1, 2, 3]\ndraws = []\nfor i in items:\n    draws.append(round(float(np.random.rand()), 4) + f(i))"
    cells = [SEEDER, helper, loop, "r = list(draws)"]
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = _plain(cells)

    nb_runner.restart()
    nb_runner.run_cells([1, 5])
    assert nb_runner.peek("r") == expected


@pytest.mark.timeout(180)
def test_a_function_that_does_more_than_seed_is_not_run_again(nb_runner):
    noisy = SEEDER.replace("    random.seed(s)\n", "    print('seeding', s)\n    random.seed(s)\n")
    cells = [noisy, SLOW, DRAW]
    nb_runner.create_notebook([C_ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = _plain(cells)

    nb_runner.restart()
    nb_runner.run_cells([1, 4])
    out = nb_runner.get_output(4)
    assert "RANDOM-STREAM-UNRESTORED" in out
    assert "RANDOM-UNSEEDED" not in out, "the stream is seeded above: 'seed the source' is the wrong advice"
    assert "seeding" not in out

    # What the warning says to do: run from the top.
    nb_runner.run_cells([2, 3, 4])
    assert nb_runner.peek("r") == expected
    assert "RANDOM-" not in nb_runner.get_output(4)
