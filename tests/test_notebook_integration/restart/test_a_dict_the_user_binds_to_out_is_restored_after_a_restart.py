"""A dict the user binds to ``Out`` is restored after a restart.

IPython's history (``Out`` while it is ``_oh``) is never a statement's
output: ``Out[2].set_title('x')`` changes the axes it holds, not the
history. But the check asked before the statement ran, when ``Out`` was
still IPython's, and dropped it from ``Out = ({'a': [1]}, slow())[0]`` as
well: stored without ``Out``, the entry served after a restart left
``Out`` as the history, and ``Out['a'][0]`` below raised.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLEEP = ABOVE_PERSISTENCE_FLOOR_S * 2
CELLS = [
    "import cash\n%cash_on\nimport time",
    f"Out = ({{'a': [1]}}, time.sleep({SLEEP}))[0]",
    f"r = (Out['a'][0], time.sleep({SLEEP}))[0]\nprint('R', r, Out is _oh)",
]


def test_the_cell_reading_it_sees_the_users_dict(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 1 False" in nb_runner.get_output(3), "control: the first run"

    nb_runner.restart()
    nb_runner.run_all()

    assert "R 1 False" in nb_runner.get_output(3), nb_runner.get_output(3)
