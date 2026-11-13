"""A cell that raised, whose loop holds a blank line with the indent an
editor leaves on it, is known as failed when a cell below runs.

IPython blanks that line before it runs the cell, so cash filed the failure
under another text than the notebook holds, took the cell for one that ran
through, and the next cell's repair ran its last line, which the failed run
never reached, on the lists it had half filled. The value is the one plain
IPython leaves, and the one Restart & Run All leaves up to the error.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(180)]

CELLS = [
    "vals = [1, 2, 0, 4]\nrows = []",
    "xs, ys = [], []\nfor v in vals:\n    xs.append(v)\n    \n    ys.append(10 / v)\nrows = list(zip(xs, ys))",
    "print('rows', len(rows), 'xs', len(xs))",
]


def _run(nb_runner, with_cash: bool) -> str:
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel(with_cash=with_cash)
    nb_runner.run_cell(1)
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    nb_runner.shutdown()
    return out


def test_the_cell_below_sees_what_the_failed_cell_left(nb_runner):
    plain = _run(nb_runner, with_cash=False)
    assert "rows 0 xs 3" in plain
    assert "rows 0 xs 3" in _run(nb_runner, with_cash=True)
