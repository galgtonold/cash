"""Re-running a cell alone above a cell that changed its input in place.

The re-run reads the input as the cells above leave it, as a top-to-bottom
run does: it computes without the change below, and the variable no longer
holds it until that cell runs again. cash says so (NOTEBOOK-CHANGED-BELOW).
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

CELLS = [
    "import numpy as np, pandas as pd, time\ns = pd.Series(np.arange(200_000, dtype=float))",
    "def summarize(x):\n    time.sleep(0.2)\n    return float(x.iloc[0])\nout = summarize(s)\nprint(out)",
    "s.iloc[0] = 1e9",
]


def test_the_rerun_says_which_cell_changed_it(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "NOTEBOOK-CHANGED-BELOW" not in nb_runner.get_raw_output(2)
    nb_runner.run_cell(2)
    raw = nb_runner.get_raw_output(2)
    assert "NOTEBOOK-CHANGED-BELOW" in raw
    assert "'s'" in raw and "cell 3" in raw
    assert nb_runner.peek("out") == "0.0"
    assert nb_runner.peek("float(s.iloc[0])") == "0.0"


def test_a_run_from_the_top_is_quiet(nb_runner):
    """Run All twice: quiet. The re-run alone after it is the positive control."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert "NOTEBOOK-CHANGED-BELOW" not in nb_runner.get_raw_output(2)
    nb_runner.run_cell(2)
    assert "NOTEBOOK-CHANGED-BELOW" in nb_runner.get_raw_output(2)
