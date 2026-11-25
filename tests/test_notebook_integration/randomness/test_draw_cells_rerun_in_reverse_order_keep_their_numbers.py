"""Two seeded draw cells, re-run lower one first, keep their numbers.

A cell re-run on its own draws where a run from the top draws. The check
before a draw cell puts the stream back and names it by its lineage there;
that name belongs to the cells from the last draw above on, never to the
seed's cell above them. Given to the seed's cell too, the first draw, re-run
after the second, started from where the second starts and took its number.

Oracle: the same cells run top to bottom in a plain interpreter.
"""

import random

import numpy as np
import pytest

pytestmark = pytest.mark.libraries

C_ON = "import cash\n%cash_on"


def _pair(runner):
    return runner.peek("(a, b)")


@pytest.mark.timeout(180)
def test_the_upper_draw_run_after_the_lower_one_keeps_its_number(nb_runner):
    nb_runner.create_notebook(
        [
            C_ON,
            "import numpy as np\nnp.random.seed(0)",
            "a = np.random.randint(1000)",
            "b = np.random.randint(1000)",
            "c = (a, b)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    np.random.seed(0)
    expected = repr((np.random.randint(1000), np.random.randint(1000)))
    assert _pair(nb_runner) == expected

    nb_runner.run_cell(4)
    nb_runner.run_cell(3)
    assert _pair(nb_runner) == expected, "the upper draw took the lower draw's number"
    nb_runner.run_cell(5)
    assert _pair(nb_runner) == expected, "a cell below moved the lower draw"
    assert nb_runner.peek("c") == expected
    # And once more in each order.
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    nb_runner.run_cell(4)
    nb_runner.run_cell(3)
    assert _pair(nb_runner) == expected


@pytest.mark.timeout(180)
def test_the_random_module_draws_keep_their_numbers(nb_runner):
    nb_runner.create_notebook(
        [
            C_ON,
            "import random\nrandom.seed(4)",
            "a = random.randint(0, 999)",
            "b = random.randint(0, 999)",
            "c = (a, b)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    random.seed(4)
    expected = repr((random.randint(0, 999), random.randint(0, 999)))

    nb_runner.run_cell(4)
    nb_runner.run_cell(3)
    nb_runner.run_cell(5)
    assert _pair(nb_runner) == expected


@pytest.mark.timeout(180)
def test_array_draws_with_a_cell_between_keep_their_numbers(nb_runner):
    nb_runner.create_notebook(
        [
            C_ON,
            "import numpy as np\nnp.random.seed(2)",
            "a = float(np.random.rand(3).sum())",
            "k = 5",
            "b = float(np.random.rand(3).sum())",
            "c = (a, b)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    np.random.seed(2)
    expected = repr((float(np.random.rand(3).sum()), float(np.random.rand(3).sum())))

    nb_runner.run_cell(5)
    nb_runner.run_cell(3)
    nb_runner.run_cell(6)
    assert _pair(nb_runner) == expected


@pytest.mark.timeout(180)
def test_a_stored_upper_draw_keeps_its_number_too(nb_runner):
    """The upper draw also does slow work, so its entry is written."""
    nb_runner.create_notebook(
        [
            C_ON,
            "import time\nimport numpy as np\nnp.random.seed(0)",
            "time.sleep(0.3); a = np.random.randint(1000)",
            "b = np.random.randint(1000)",
            "c = (a, b)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    np.random.seed(0)
    expected = repr((np.random.randint(1000), np.random.randint(1000)))

    nb_runner.run_cell(4)
    nb_runner.run_cell(3)
    nb_runner.run_cell(5)
    assert _pair(nb_runner) == expected
