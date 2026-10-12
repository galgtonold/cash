"""An array rebuilt with the same values but read-only still refuses the
write a cell below makes, as in plain Python.

``arr = np.array(df["a"])`` edited to ``arr = df["a"].to_numpy()`` (read-only
in pandas 3, same values) took the old array's lineage, so the next cell's
``arr[arr < 2] = 0`` was a cache hit: no "assignment destination is
read-only", and ``arr`` came back clipped and writeable. ``np.frombuffer``
makes the same read-only array on every numpy and pandas version.
"""

import pytest

pytest.importorskip("numpy")

pytestmark = [pytest.mark.integration]

ON = "import cash\n%cash_on"
SETUP = (
    "import cash\n"
    "cash.configure(min_execution_time_to_cache_seconds=0.0, call_cost_floor_seconds=0.0)\n"
    "import numpy as np\n"
    "raw = np.array([1.0, 5.0, 3.0]).tobytes()"
)
WRITE = "try:\n    arr[arr < 2] = 0.0\n    status = 'clipped'\nexcept ValueError as e:\n    status = 'error: ' + str(e)"


def test_the_write_to_the_read_only_array_raises(nb_runner):
    nb_runner.create_notebook([ON, SETUP, "arr = np.frombuffer(raw).copy()", WRITE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("status") == "'clipped'"

    nb_runner.set_cell_source(3, "arr = np.frombuffer(raw)")  # same values, read-only
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)

    assert nb_runner.peek("status") == "'error: assignment destination is read-only'"
    assert nb_runner.peek("arr.flags.writeable") == "False"
    assert nb_runner.peek("arr.tolist()") == "[1.0, 5.0, 3.0]"
