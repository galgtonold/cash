"""A loop body statement reading what the same loop fills follows an edit above.

``loss = score(history, epoch)`` reads the list ``history.append(loss * lr)``
fills on the passes before it. After ``lr`` is edited and the notebook runs
again, the later passes must not be served the losses the old ``lr`` gave.
Compared with plain Python: ``[0, 1, 3, 7]``, then ``[0, 2, 8, 26]``.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SETUP = "import time\ndef score(history, epoch):\n    time.sleep(0.1)\n    return sum(history) + epoch\nlr = {lr}"
LOOP = "history = []\nfor epoch in range(4):\n    loss = score(history, epoch)\n    history.append(loss * lr)\nprint(history)"


def test_an_edit_above_reaches_every_pass(nb_runner):
    nb_runner.create_notebook([SETUP.format(lr=1), LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(2).strip() == "[0, 1, 3, 7]"

    nb_runner.set_cell_source(1, SETUP.format(lr=2))
    nb_runner.run_all()
    assert nb_runner.get_output(2).strip() == "[0, 2, 8, 26]"

    # Unchanged, the notebook runs again to the same list.
    nb_runner.run_all()
    assert nb_runner.get_output(2).strip() == "[0, 2, 8, 26]"
