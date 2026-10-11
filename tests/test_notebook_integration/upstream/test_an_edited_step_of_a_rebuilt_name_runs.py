"""Editing a cell that rebuilds a name in steps runs the edited line.

``x = load(); x = heavy(x); y = ok(x); x = x + 1``, where ``ok`` returns its
argument, edited to ``y = triple(x)`` and ``x = x + 2``: cash restored
``heavy``, ran the last line and skipped the new ``y`` line as covered, so
``y`` kept the removed line's value. With ``y = boom(x)`` the cell did not
raise, and ``x`` held the value of a line a plain run never reaches.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

HELPERS = """
import time, numpy as np
def load():
    time.sleep(0.2); return np.arange(10.0)
def heavy(x):
    time.sleep(0.2); return x + 100
def ok(x):
    time.sleep(0.05); return x
def triple(x):
    time.sleep(0.05); return x * 3
def boom(x):
    raise ValueError("boom")
"""
SETUP = "import cash\n%cash_on\nimport h"
FIRST = "x = h.load()\nx = h.heavy(x)\ny = h.ok(x)\nx = x + 1"


def _edit(nb_runner, tmp_path, edited):
    (tmp_path / "h.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.create_notebook([SETUP, FIRST])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, edited)


def test_the_edited_line_runs(nb_runner, tmp_path):
    _edit(nb_runner, tmp_path, "x = h.load()\nx = h.heavy(x)\ny = h.triple(x)\nx = x + 2")
    nb_runner.run_cells([2])
    assert nb_runner.peek("float(y.sum())") == "3135.0", "y kept the value of the removed line"
    assert nb_runner.peek("float(x.sum())") == "1065.0"


def test_an_edited_line_that_raises_stops_the_cell(nb_runner, tmp_path):
    _edit(nb_runner, tmp_path, "x = h.load()\nx = h.heavy(x)\ny = h.boom(x)\nx = x + 2")
    with pytest.raises(CellExecutionError, match="boom"):
        nb_runner.run_cells([2])
    assert nb_runner.peek("float(x.sum())") == "1045.0"  # the line after the raise never ran
    assert nb_runner.peek("float(y.sum())") == "1045.0"  # as the first run left it
