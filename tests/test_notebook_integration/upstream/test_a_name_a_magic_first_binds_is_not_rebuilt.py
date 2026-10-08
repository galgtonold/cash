"""A name a line magic binds first is not rebuilt by the cell that reads it.

``%time n = bump()`` alone in a cell, with nothing binding ``n`` above it,
was run again by the next cell that reads ``n`` on every Run All: its side
effects and its cost doubled, and a timed seeded draw came out swapped with
the draw below it. A shell capture binding a name first (``p = !echo a b``)
made the cell reading it warn NOTEBOOK-MAGIC-STALE although it had just run.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

COUNT = (
    "import os\ndef bump():\n    os.write(os.open('calls', os.O_WRONLY | os.O_APPEND | os.O_CREAT), b'c')\n    return 1"
)


def test_a_timed_line_runs_once_per_run_all(nb_runner):
    nb_runner.create_notebook([COUNT, "%time n = bump()", "print('r', n)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()

    assert (nb_runner.work_dir / "calls").read_text(encoding="utf-8") == "cc"


def test_a_shell_capture_read_below_does_not_warn(nb_runner):
    nb_runner.create_notebook(["k = 2", "p = !echo a b", "print('C', p)"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out = nb_runner.get_raw_output(3)  # the warning is filtered from get_output
    assert "C ['a b']" in out
    assert "NOTEBOOK-MAGIC-STALE" not in out


def test_a_timed_seeded_draw_and_the_draw_below_keep_their_order(nb_runner):
    np = pytest.importorskip("numpy")
    nb_runner.create_notebook(
        [
            "import numpy as np, time\ndef slow(a):\n    time.sleep(0.4)\n    return a\nnp.random.seed(42)",
            "%time a = np.random.rand(2)",
            "b = slow(np.random.rand(2))\nprint('r', a.round(3).tolist(), b.round(3).tolist())",
        ]
    )
    np.random.seed(42)
    expected = f"r {np.random.rand(2).round(3).tolist()} {np.random.rand(2).round(3).tolist()}"
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(3)
    nb_runner.run_all()
    assert expected in nb_runner.get_output(3)
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(3)
