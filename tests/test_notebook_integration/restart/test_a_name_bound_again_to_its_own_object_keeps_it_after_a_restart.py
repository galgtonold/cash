"""A frame a statement writes into and binds back to its name stays the
object found by identity above it after a restart.

``sales = slow(add(sales))``, where ``add`` writes a column and returns the
frame: plain Python keeps the frame, so ``key = id(sales)`` and
``ref = weakref.ref(sales)``, taken before it, still find it. Served from
disk after a restart, ``sales`` was bound to the stored copy, and both
lost it.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(240)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"

HELPERS = (
    "import time\nimport weakref\nimport pandas as pd\n"
    "def slow(v):\n    time.sleep(0.3)\n    return v\n"
    "def add(df):\n    df['c'] = df['a'] * 2\n    return df"
)

READ = "# @cash:no-cache\nprint('R', key == id(sales), ref() is sales, list(sales.columns))"
EXPECTED = "R True True ['a', 'c']"


@pytest.mark.parametrize("split", [False, True], ids=["one cell", "across cells"])
def test_what_holds_the_frame_by_identity_finds_it_after_a_restart(nb_runner, split):
    make = "sales = slow(pd.DataFrame({'a': range(300)}))"
    hold = "key = id(sales)\nref = weakref.ref(sales)"
    last = "sales = slow(add(sales))"
    body = [make, hold, last] if split else [f"{make}\n{hold}\n{last}"]
    cells = [SETUP, HELPERS, *body, READ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert EXPECTED in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))

    nb_runner.restart()
    nb_runner.run_all()

    assert EXPECTED in nb_runner.get_output(len(cells)), (
        nb_runner.get_raw_output(len(cells) - 1),
        nb_runner.get_raw_output(len(cells)),
    )
    # Not vacuous: the statement binding sales again was served from disk.
    assert "CACHED: sales" in nb_runner.get_raw_output(len(cells) - 1), nb_runner.get_raw_output(len(cells) - 1)
