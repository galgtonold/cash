"""``for col in df:`` runs each column once, on every Run All and after a restart.

A short cheap loop is learned as a split: ``for col in df[:5]`` then
``for col in df[5:]``. The loop was sized by ``len(df)``, the rows, and a
frame slices rows, so each half iterated all the columns: from the second Run
All on, a 60-row, 8-column frame's loop collected ``abcdefghabcdefgh`` and
summed every column twice, silently, also after a restart. Only a value that
slices as it iterates is split now.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(180)]

# Any eligible loop is learned as a split on its first run: the verdict is a
# wall-clock measurement, which these thresholds take out of the test.
SETUP = "import cash\ncash.configure(loop_split_max_iter_seconds=1.0, loop_split_min_remaining_seconds=0.0)\n%cash_on"
FRAME = (
    "import pandas as pd, numpy as np\n"
    "df = pd.DataFrame(np.arange(60 * 8).reshape(60, 8) % 7, columns=list('abcdefgh'))"
)
ORDER = "order = []\nfor col in df:\n    order.append(col)\nprint(len(order), ''.join(order))"
TOTAL = "total = 0\nfor col in df:\n    total += int(df[col].sum())\nprint(total)"
ROWS = "seen = []\nfor r in rows:\n    seen.append(r)\nprint(len(seen), sum(seen))"


def test_each_column_is_visited_once(nb_runner):
    pytest.importorskip("pandas")
    nb_runner.create_notebook([SETUP, FRAME, ORDER, TOTAL])
    nb_runner.start_kernel()
    for step in ("first Run All", "second Run All", "Run All after a restart"):
        if step.endswith("restart"):
            nb_runner.restart()
        nb_runner.run_all()
        assert nb_runner.get_output(3).strip() == "8 abcdefgh", f"{step}: {nb_runner.get_raw_output(3)}"
        assert nb_runner.get_output(4).strip() == "1434", f"{step}: {nb_runner.get_raw_output(4)}"


def test_a_list_is_still_split_and_still_right(nb_runner):
    """Positive control: the split still runs, for a value it is right for."""
    nb_runner.create_notebook([SETUP, "rows = list(range(60))", ROWS])
    nb_runner.start_kernel()
    for _ in range(3):
        nb_runner.run_all()
        assert nb_runner.get_output(3).strip() == "60 1770", nb_runner.get_raw_output(3)
    assert nb_runner.peek("len(seen)").strip() == "60"
