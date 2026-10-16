"""A cell that starts a thread and ends while it runs is named once
(``NOTEBOOK-THREAD-ALIVE``); a cell that joins its thread is not.

What the thread changes after its cell finished is outside cash's view.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

CODE = "[NOTEBOOK-THREAD-ALIVE]"

CELLS = [
    "import threading\nstop = threading.Event()\nrows = []\ndef fill():\n    stop.wait(5)\n    rows.append(1)",
    "t = threading.Thread(target=fill, name='filler')\nt.start()",
    "stop.set()\nt.join()\ndone = threading.Thread(target=fill, name='joined')\ndone.start()\ndone.join()\n"
    "print(len(rows))",
]


def test_a_thread_left_running_is_named_once_and_a_joined_one_is_not(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    second = nb_runner.get_raw_output(2)
    assert CODE in second, second
    assert "cell 2" in second and "'filler'" in second and "fill" in second, second
    third = nb_runner.get_raw_output(3)
    assert CODE not in third, third
    assert "2" in nb_runner.get_output(3)

    nb_runner.run_all()
    assert CODE not in nb_runner.get_raw_output(2), "named again for the same cell"
