"""An object holding a lock, changed by a lambda or comprehension whose
result is kept, is seen as changed.

Cell 2 is ``k = 1; r = list(map(lambda s: s["tracker"].inc(), [state] * k))``.
``state`` holds an object with a ``threading.Lock``, which cash cannot
read. After an edit of cell 2 to ``k = 2`` and Run All, cell 3 (``out =
slow_report(state)``) was served from the cache with the old count: the
statement keeps its result, so its unreadable argument was taken as only
read, as ``rows = fetch(conn)`` takes it. A statement that only hands the
object to a function still is: it is served from the cache on a second run.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

SETUP = (
    "%cash_badge print\n"
    "import threading, time\n"
    "class Tracker:\n"
    "    def __init__(self):\n"
    "        self.n = 0\n"
    "        self.lock = threading.Lock()\n"
    "    def inc(self):\n"
    "        self.n += 1\n"
    "        return self.n\n"
    "def slow_report(s):\n"
    "    time.sleep(0.3)\n"
    "    return s['tracker'].n\n"
    "state = {'tracker': Tracker()}\n"
    "trackers = [state['tracker']]"
)

CHANGERS = {
    "a_lambda_given_to_map": "r = list(map(lambda s: s['tracker'].inc(), [state] * k))",
    "a_comprehension": "r = [t.inc() for t in trackers * k]",
}


@pytest.mark.parametrize("form", list(CHANGERS))
def test_the_reader_runs_again_after_the_changing_cell_is_edited(nb_runner, form):
    changer = CHANGERS[form]
    nb_runner.create_notebook([SETUP, "k = 1\n" + changer, "out = slow_report(state)\nprint('out =', out)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out = 1" in nb_runner.get_output(3), "control: the first run"

    nb_runner.set_cell_source(2, "k = 2\n" + changer)
    nb_runner.run_all()

    assert "out = 2" in nb_runner.get_output(3), nb_runner.get_raw_output(3)


@pytest.mark.parametrize("form", list(CHANGERS))
def test_the_change_is_made_on_a_run_all_after_a_restart(nb_runner, form):
    """Slow enough to be worth storing: served from the cache in a new
    kernel, ``r`` came back and the tracker was never counted up."""
    slow = SETUP.replace("        self.n += 1\n", "        time.sleep(0.3)\n        self.n += 1\n")
    assert slow != SETUP
    nb_runner.create_notebook([slow, "k = 1\n" + CHANGERS[form]])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("state['tracker'].n").strip() == "1"
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()

    assert nb_runner.peek("state['tracker'].n").strip() == "1", nb_runner.get_raw_output(2)
    assert nb_runner.peek("r").strip() == "[1]"


def test_a_statement_that_only_hands_the_object_on_is_served_from_the_cache(nb_runner):
    nb_runner.create_notebook([SETUP, "rows = slow_report(state)\nprint('rows =', rows)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "rows = 0" in nb_runner.get_output(2)

    nb_runner.run_cells([2])

    assert shows_cached(nb_runner.get_output(2)), nb_runner.get_raw_output(2)
