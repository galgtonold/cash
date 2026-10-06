"""Variables sharing an object come back from disk as one object.

``b = a`` before ``a['x'] = ...``, and ``cfg = {'model': clf}`` before
``clf.w = ...``: each statement is stored together with the variable that
holds its object, so after a restart the hit restores both as one graph --
``a is b`` and ``cfg['model'] is clf`` again, as running the cells does, and
a later in-place change shows through both names.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import time\nclass Model:\n    def __init__(self):\n        self.w = 0",
    "a = {'v': 1}\nb = a",
    "a['x'] = (time.sleep(0.5), 41)[1]",
    "clf = Model()\ncfg = {'model': clf}",
    "clf.w = (time.sleep(0.5), 4)[1]",
    "b['y'] = 2\ncfg['model'].w += 1\nprint('SAME', a is b, a.get('y'), cfg['model'] is clf, clf.w)",
]

EXPECTED = "SAME True 2 True 5"


def test_after_a_restart_the_aliases_are_one_object_again(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert EXPECTED in nb_runner.get_output(7)

    nb_runner.restart()
    nb_runner.run_all()

    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert shows_cached(nb_runner.get_output(6)), nb_runner.get_output(6)
    assert EXPECTED in nb_runner.get_output(7), nb_runner.get_output(7)


def test_a_cell_run_alone_after_a_restart_gets_the_aliases_back(nb_runner):
    """Only the last cell runs: the upstream check brings back what it reads,
    the stored statements with the variables they were stored with."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.restart()
    nb_runner.run_cells([1, 7])

    assert EXPECTED in nb_runner.get_output(7), nb_runner.get_output(7)
