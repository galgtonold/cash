"""After a restart, a statement whose entry stores another variable holding
its objects restores while that variable's cell has not run again.

``hgb`` and ``lr`` both hold the column list ``cols`` (p11: two fitted
pipelines whose ColumnTransformers hold the same column-name lists). After a
restart the cells up to ``hgb`` run again, but not ``lr``'s: ``hgb`` comes
from the cache, holding the live ``cols``; ``lr``'s cell run afterwards
shares that list too, as in a plain kernel.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import time\ncols = ['a', 'b']",
    "lr = {'cols': cols, 'w': (time.sleep(0.5), 1)[1]}",
    "hgb = {'cols': cols, 'w': (time.sleep(0.5), 2)[1]}",
    "print('BOUND', 'lr' in globals(), hgb['cols'] is cols)",
    "cols.append('c')\nprint('SHARED', hgb['cols'] is cols is lr['cols'], lr['cols'], hgb['cols'])",
]


def test_the_entry_restores_without_the_holder_and_shares_its_list_after(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "SHARED True ['a', 'b', 'c'] ['a', 'b', 'c']" in nb_runner.get_output(6)

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 4, 5])

    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert "BOUND False True" in nb_runner.get_output(5), nb_runner.get_output(5)

    nb_runner.run_cells([3, 6])
    assert "SHARED True ['a', 'b', 'c'] ['a', 'b', 'c']" in nb_runner.get_output(6), nb_runner.get_output(6)
