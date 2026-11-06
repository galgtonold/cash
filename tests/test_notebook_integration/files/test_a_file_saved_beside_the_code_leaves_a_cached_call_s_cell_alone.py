"""A file saved in the notebook's folder leaves a cached call's cell alone.

p08: ``raw = etl.load_logs()`` calls a ``@cash.cache`` function over polars.
Keying it found polars' installed version, which lists every folder on
``sys.path`` -- the notebook's own folder first -- and that listing ran
under the statement's file tracker: the statement recorded the folder as
an input, keyed on its stat, and every cell taking ``raw`` inherited it. On
a machine where the notebook file is saved next to the code at each run,
those cells re-ran on every pass (``file changed: .../proj``, 32.9 s).

A fresh kernel, so the library lookup is made in this test. What must
still hold: a change to the file the call reads re-runs the cell, and
the cell below sees the new value.
"""

import pytest

from tests._nbharness.badge import shows_cached, shows_executed

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120), pytest.mark.fresh_kernel]

LIB = (
    "import cash\n"
    "import numpy as np\n"
    "@cash.cache\n"
    "def total(path):\n"
    "    with open(path) as fh:\n"
    "        return float(np.asarray([float(x) for x in fh.read().split()]).sum())\n"
)


def _notebook(nb_runner, tmp_path):
    (tmp_path / "p08lib.py").write_text(LIB, encoding="utf-8")
    data = tmp_path / "data.txt"
    data.write_text("1 2 3\n", encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print",
            f"import sys\nsys.path.insert(0, r'{tmp_path}')\nimport p08lib",
            f"x = p08lib.total(r'{data}')",
            "y = [x * i for i in range(3)]\nprint('y', y)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "y [0.0, 6.0, 12.0]" in nb_runner.get_output(4)
    return data


def _save_beside(tmp_path):
    """A file written in the folder by replacing it, as an editor saves."""
    tmp = tmp_path / ".report.ipynb.tmp"
    tmp.write_text("{}", encoding="utf-8")
    tmp.replace(tmp_path / "report.ipynb")


def test_a_file_saved_in_the_folder_leaves_the_call_s_cell_cached(nb_runner, tmp_path):
    _notebook(nb_runner, tmp_path)
    _save_beside(tmp_path)
    nb_runner.run_cells([3, 4])
    out = nb_runner.get_raw_output(3)
    assert shows_cached(out) and not shows_executed(out), out
    assert "y [0.0, 6.0, 12.0]" in nb_runner.get_output(4)


def test_a_changed_input_file_still_reruns_the_call_s_cell(nb_runner, tmp_path):
    data = _notebook(nb_runner, tmp_path)
    data.write_text("1 2 3 4\n", encoding="utf-8")
    nb_runner.run_cells([3, 4])
    out = nb_runner.get_raw_output(3)
    assert shows_executed(out), out
    assert "y [0.0, 10.0, 20.0]" in nb_runner.get_output(4)
