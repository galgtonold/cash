"""A cell's echoed result reaches IPython's output history with cash on.

cash runs a cell's statements itself and shows the last expression with
``display``, so IPython's display hook never saw it: ``_`` stayed ``''``,
``Out`` stayed empty and ``res = _`` silently bound an empty string.
"""

import os

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

SLOW = (
    "import os, time\n"
    "def slow(v):\n"
    "    os.write(os.open('runs.log', os.O_WRONLY | os.O_APPEND | os.O_CREAT), b'r')\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n"
    "    return v"
)


def test_underscore_and_out_hold_the_last_result(nb_runner):
    nb_runner.create_notebook(["6 * 7", "res = _"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("res") == "42"
    assert nb_runner.peek("Out[max(Out)] if Out else None") == "42"


def test_a_result_served_from_the_cache_reaches_the_history(nb_runner):
    """After a restart the result is a hit, and ``_`` still holds it."""
    nb_runner.create_notebook([SLOW, "slow(42)", "res = _"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()

    assert nb_runner.peek("res") == "42"
    log = os.path.join(nb_runner.work_dir, "runs.log")
    with open(log) as f:
        assert f.read() == "r", "the second run was not served from the cache"
