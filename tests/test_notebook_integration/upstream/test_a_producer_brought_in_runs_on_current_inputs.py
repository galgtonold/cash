"""A statement the plan brings in for a shared list runs on current inputs.

``rows = [[SCALE, 2], [3, 4]]`` is brought into the plan because a later
cell changes, through ``holder``, a list it made. It then ran on the
``SCALE`` of an earlier edit: the name is bound once, so no pass asked
whether the kernel's ``SCALE`` was the one the statement reads in a run from
the top. It printed ``5 2`` where that run prints ``5 3``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on",
    "holder = []",
    "SCALE = 1\nrows = [[SCALE, 2], [3, 4]]",
    "for r, label in zip(rows, ['a', 'b']):\n    holder.append(r)",
    "holder[0].append(99)",
    "print(sum(map(len, rows)), rows[0][0])",
]


def test_the_edited_constant_reaches_the_rebuilt_list(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "SCALE = 2\nrows = [[SCALE, 2], [3, 4]]")
    nb_runner.run_cells([4, 5])
    nb_runner.set_cell_source(3, "SCALE = 3\nrows = [[SCALE, 2], [3, 4]]")

    nb_runner.run_cells([6])

    assert "5 3" in nb_runner.get_output(6), nb_runner.get_output(6)
