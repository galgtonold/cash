"""A ``%time`` call that changes its argument in place changes it for cash too.

``%time model.fit()`` changes ``model``: the receiver of a method call is
counted. ``%time train(model)`` and ``%time np.copyto(arr, k)`` change their
argument just as much, but were invisible: after an edit of the timed line
the cell below was a cache hit on the old training; after an edit above it,
re-running the cell below alone rebuilt ``model = M(k)`` without the training
(an untrained model); after an edit of ``k`` the reader was a hit on the old
``arr``. The plain line, without ``%time``, gave the right answer each time.
What the magic's Python hands to a call is now watched as a plain statement's
is, and a name it changed gets the magic's lineage.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

HELPERS = (
    "import time\n"
    "import numpy as np\n"
    "class M:\n"
    "    def __init__(self, k):\n"
    "        self.k = k; self.w = None\n"
    "def train(m, f=10):\n"
    "    m.w = m.k * f\n"
    "def slow(v):\n"
    "    time.sleep(0.3); return v\n"
)
K2 = HELPERS + "k = 2"
K3 = HELPERS + "k = 3"
READER = "out = slow(model.w)\nprint('out', out)"


def test_an_edit_of_the_timed_call_reaches_the_reader(nb_runner):
    nb_runner.create_notebook([K2, "model = M(k)", "%time train(model)", READER])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "out 20" in nb_runner.get_output(4)
    nb_runner.set_cell_source(3, "%time train(model, 100)")
    nb_runner.run_all()
    assert "out 200" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_an_edit_above_reruns_the_timed_call_for_the_reader(nb_runner):
    nb_runner.create_notebook([K2, "model = M(k)", "%time train(model)", READER])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, K3)
    nb_runner.run_cell(4)
    assert "out 30" in nb_runner.get_output(4), (
        "the reader got the model rebuilt without its training:\n" + nb_runner.get_raw_output(4)
    )


def test_a_library_call_that_fills_an_array(nb_runner):
    nb_runner.create_notebook([K2, "arr = np.zeros(3)", "%time np.copyto(arr, k)", "print('r', slow(arr.tolist()))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, K3)
    nb_runner.run_all()
    assert "r [3.0, 3.0, 3.0]" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_after_a_restart_an_edit_above_reruns_the_timed_call_for_the_reader(nb_runner):
    """What the timed call changed is known in the next kernel too."""
    nb_runner.create_notebook([K2, "model = M(k)", "%time train(model)", READER])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(1, K3)
    nb_runner.run_cell(4)
    assert "out 30" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_a_timed_call_that_only_reads_leaves_the_reader_cached(nb_runner):
    """Control: ``%time show(model)`` changes nothing, so the reader below
    stays a cache hit on the next Run All."""
    show = "def show(m):\n    return m.w"
    cells = ["%cash_badge print", K2 + "\n" + show, "model = M(k)\ntrain(model)", "%time show(model)", READER]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(5)
    assert "out 20" in raw, raw
    assert shows_cached(raw), raw
