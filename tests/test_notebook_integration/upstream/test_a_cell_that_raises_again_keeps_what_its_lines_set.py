"""Running a cell again while it still raises leaves what a plain run leaves.

``x = load(); y = x * 2; x = heavy(x); w = y + 1; z = boom(x)``: on the
second run cash skips ``y`` and ``w`` (current) and restores ``x``. After the
error it ran the skipped lines again, at the end, so ``y = x * 2`` read the x
that ``x = heavy(x)`` had rebound and a scratch cell, the variable explorer
or a console saw y and w built from the wrong x.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytest.importorskip("numpy")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

HELPERS = """\
import time, numpy as np
def load():
    time.sleep(0.2)
    return np.arange(10.0)
def heavy(x):
    time.sleep(0.2)
    return x + 100
def boom(x):
    raise ValueError("boom")
"""

CELL = "x = helpers.load()\ny = x * 2\nx = helpers.heavy(x)\nw = y + 1\nz = helpers.boom(x)\nx = x / 2"
SUMS = "(float(x.sum()), float(y.sum()), float(w.sum()))"


def test_a_second_failing_run_leaves_the_lines_before_the_error_as_they_ran(nb_runner):
    (nb_runner.work_dir / "helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.create_notebook(["import helpers\nimport numpy as np", CELL])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(2)
    # a plain kernel: x = heavy(load()), y and w from load()
    assert nb_runner.peek(SUMS) == "(1045.0, 90.0, 100.0)"
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(2)
    assert nb_runner.peek(SUMS) == "(1045.0, 90.0, 100.0)"
