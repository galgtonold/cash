"""An object holding a lock, changed inside a lambda given to ``map``, is seen as changed.

Cell 2 is ``k = 1; list(map(lambda s: s["tracker"].inc(), [state] * k))``.
``state`` holds an object with a ``threading.Lock``, which cash cannot read.
After an edit of cell 2 to ``k = 2`` and Run All, cell 3 (``out =
slow_report(state)``) was served from the cache with the old count: the
statement keeps nothing, and still its unreadable argument was taken as
only read because it was handed on inside a list.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]


def _setup(with_lock: bool) -> str:
    return (
        "import threading, time\n"
        "class Tracker:\n"
        "    def __init__(self):\n"
        "        self.n = 0\n"
        f"        {'self.lock = threading.Lock()' if with_lock else 'pass'}\n"
        "    def inc(self):\n"
        "        self.n += 1\n"
        "def slow_report(s):\n"
        "    time.sleep(0.3)\n"
        "    return s['tracker'].n\n"
        "state = {'tracker': Tracker()}"
    )


ONCE = "k = 1\nlist(map(lambda s: s['tracker'].inc(), [state] * k))"
TWICE = "k = 2\nlist(map(lambda s: s['tracker'].inc(), [state] * k))"


@pytest.mark.parametrize("with_lock", [True, False], ids=["holding_a_lock", "control_without_the_lock"])
def test_the_reader_runs_again_after_the_changing_cell_is_edited(nb_runner, with_lock):
    nb_runner.create_notebook([_setup(with_lock), ONCE, "out = slow_report(state)\nprint('out =', out)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out = 1" in nb_runner.get_output(3), "control: the first run"

    nb_runner.set_cell_source(2, TWICE)
    nb_runner.run_all()

    assert "out = 2" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
