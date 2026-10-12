"""``X = data[FEATS].to_numpy()`` on a table restored from the cache is as
writable as on the table a plain kernel reads.

The restored table was one block for all its float columns, so pandas handed
out a read-only view of the cached data, and the user's own standardisation
``X -= X.mean(axis=0)`` raised "output array is read-only" on every Run All
after the feature list changed, and after a restart.
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

CELLS = [
    "import pandas as pd, numpy as np\nFEATS = ['f1', 'f2', 'f3']",
    "data = pd.read_csv('clf.csv')\nX = data[FEATS].to_numpy()",
    "X -= X.mean(axis=0)\nprint('standardised', X.shape)",
]


@pytest.mark.fresh_kernel
def test_standardising_the_array_works_on_a_restored_table(nb_runner, tmp_path):
    frame = pd.DataFrame({"f1": np.arange(50.0), "f2": np.arange(50.0) ** 0.5, "f3": np.ones(50)})
    (tmp_path / "clf.csv").write_text(frame.to_csv(index=False), newline="\n", encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, CELLS[0].replace("['f1', 'f2', 'f3']", "['f1', 'f2']"))
    nb_runner.run_all()
    assert "standardised (50, 2)" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(1, CELLS[0])
    nb_runner.run_all()
    assert "standardised (50, 3)" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
