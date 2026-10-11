"""A value rebuilt before an edited cell takes what shares its object along.

``g = df.groupby('k')`` holds ``df``; a loop below changes ``df`` in place.
After the loop is edited and only that cell and the one below are run, the
groupby reads ``df`` as the edited loop leaves it, as in a plain kernel.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.timeout(300)]

LOOP = "for i in range(1):\n    df.loc[i, 'v'] = {}"
CELLS = [
    "import pandas as pd\ndf = pd.DataFrame({'k': [1, 1, 2], 'v': [1., 2., 3.]})",
    "g = df.groupby('k')",
    LOOP.format(100),
    "print(g['v'].sum().tolist())",
]


def test_a_groupby_reads_the_frame_after_the_edited_loop(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, LOOP.format(500))
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "[502.0, 3.0]"
    nb_runner.set_cell_source(3, LOOP.format(7))
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert nb_runner.get_output(4).strip() == "[9.0, 3.0]"
    assert nb_runner.peek("g['v'].sum().tolist()").strip() == "[9.0, 3.0]"
