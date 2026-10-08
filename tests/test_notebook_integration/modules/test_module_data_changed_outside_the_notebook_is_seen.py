"""Module data changed outside the notebook's cells reaches the statements that read it.

A statement is keyed on the module data it read when it ran, so that
``mylib.K = 7`` in a cell below the reader does not send the reader back to
run. A change no cell made -- set from a debugger or a console attached to
the kernel, or the module's file edited -- is not the notebook's doing, and
running only the last cell must rebuild what was computed from the old value.
"""

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = "import time\nK = {k}\ndef from_k(x):\n    time.sleep(0.3)\n    return x * K\n"


def test_a_value_set_from_outside(nb_runner, tmp_path):
    (tmp_path / "statelib.py").write_text(LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(["import statelib", "b = statelib.from_k(10)", "u = b + 1\nprint('U', u)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "U 21" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.peek("setattr(__import__('statelib'), 'K', 5)")
    nb_runner.run_cell(3)
    assert nb_runner.peek("u") == "51", nb_runner.get_raw_output(3)


def test_a_value_set_from_outside_with_a_setting_below(nb_runner, tmp_path):
    """A notebook setting below the reader does not reach it; one from
    outside does, even when the notebook sets the same value elsewhere."""
    (tmp_path / "statelib.py").write_text(LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(
        ["import statelib", "b = statelib.from_k(10)", "statelib.K = 7", "u = b + 1\nprint('U', u)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "21", nb_runner.get_raw_output(4)

    nb_runner.peek("setattr(__import__('statelib'), 'K', 5)")
    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "51", nb_runner.get_raw_output(4)


def test_a_value_set_from_outside_over_a_setting_above(nb_runner, tmp_path):
    """The reader runs again with the module as the cells above it leave it,
    as a variable is rebuilt: ``statelib.K = 3`` above it runs again over the
    value set from outside, and ``statelib.K = 7`` below it after it, as a
    top-to-bottom run has them."""
    (tmp_path / "statelib.py").write_text(LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(
        ["import statelib\nstatelib.K = 3", "b = statelib.from_k(10)", "statelib.K = 7", "u = b + 1\nprint('U', u)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "31", nb_runner.get_raw_output(4)

    nb_runner.peek("setattr(__import__('statelib'), 'K', 5)")
    nb_runner.run_cell(4)
    assert nb_runner.peek("u") == "31", nb_runner.get_raw_output(4)
    assert nb_runner.peek("statelib.K") == "7"


def test_the_value_edited_in_the_file(nb_runner, tmp_path):
    path = tmp_path / "statelib.py"
    path.write_text(LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(["import statelib", "b = statelib.from_k(10)", "u = b + 1\nprint('U', u)"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    before = os.stat(path).st_mtime_ns
    path.write_text(LIB.format(k=4), encoding="utf-8")
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))
    nb_runner.run_cell(3)
    assert nb_runner.peek("u") == "41", nb_runner.get_raw_output(3)


BIG_LIB = (
    "import time\nimport numpy as np\nTABLE = np.full(1_000_000, {k}.0)\n"
    "def from_k(x):\n    time.sleep(0.3)\n    return float(x * TABLE[0])\n"
)


def test_a_big_table_changed_in_place_is_seen_by_the_cell_that_reads_it(nb_runner, tmp_path):
    """A table of 8 MB is not hashed before every cell that only uses what
    its reader built (a 256 MB one cost 0.25 s a cell). A change made to it
    in place from outside is seen by the next cell that reads it; the cells
    below answer, until then, what a plain kernel answers. One rebinding it
    is seen by every cell, as for a small value."""
    (tmp_path / "statelib.py").write_text(BIG_LIB.format(k=2), encoding="utf-8")
    nb_runner.create_notebook(["import statelib", "b = statelib.from_k(10)", "u = b + 1\nprint('U', u)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert nb_runner.peek("u") == "21.0"

    nb_runner.peek("__import__('statelib').TABLE.__setitem__(0, 5.0)")
    nb_runner.run_cell(3)
    assert nb_runner.peek("u") == "21.0", "a plain kernel answers from the b it has"
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert nb_runner.peek("u") == "51.0", nb_runner.get_raw_output(3)

    nb_runner.peek("setattr(__import__('statelib'), 'TABLE', __import__('numpy').full(1_000_000, 4.0))")
    nb_runner.run_cell(3)
    assert nb_runner.peek("u") == "41.0", nb_runner.get_raw_output(3)
