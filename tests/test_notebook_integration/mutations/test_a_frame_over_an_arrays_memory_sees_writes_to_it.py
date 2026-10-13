"""A frame built over an array's memory without a copy sees writes to it.

``s = pl.Series(arr)``, ``pd.Series(arr, copy=False)``, ``pa.table({'a':
arr})`` and ``df['a'].to_numpy()`` read the memory of another variable, so
``arr[:10] = 1000.0`` (or ``df.loc[...] = ...``) changes them as well. A cached
call over the frame kept answering with the old data after such a write: the
frame's lineage did not move, so the same call in a cell below the write
was served the result from before it. And a call that returns such a frame,
restored after a restart, came back as a copy that no longer saw the write.

In plain Python, ``arange(100_000.0)`` sums to 4999950000.0 and 4999959955.0 once its
first ten values are 1000.0.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

pytest.importorskip("polars")
pytest.importorskip("pyarrow")

SETUP = "import cash\n%cash_on\nimport time\nimport numpy as np, pandas as pd, polars as pl, pyarrow as pa"
TOTAL = (
    "def total(x):\n"
    "    time.sleep(0.3)\n"
    "    if isinstance(x, pa.Table): return float(pa.compute.sum(x.column(0)).as_py())\n"
    "    if isinstance(x, (pl.DataFrame, pd.DataFrame)): x = x[x.columns[0]]\n"
    "    return float(np.asarray(x).sum())"
)
BEFORE, AFTER = "4999950000.0", "4999959955.0"

SHAPES = {
    "polars": ("arr = np.arange(100_000.0)", "s = pl.DataFrame({'a': arr})", "arr[:10] = 1000.0"),
    "pandas-copy-false": ("arr = np.arange(100_000.0)", "s = pd.Series(arr, copy=False)", "arr[:10] = 1000.0"),
    "arrow": ("arr = np.arange(100_000.0)", "s = pa.table({'a': arr})", "arr[:10] = 1000.0"),
    "numpy-view-of-pandas": (
        "p = pd.DataFrame({'a': np.arange(100_000.0)})",
        "s = p['a'].to_numpy()",
        "p.loc[:9, 'a'] = 1000.0",
    ),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_same_call_below_the_write_sees_it(nb_runner, shape):
    """The call's cell, copied below the write: the same statement over the
    same name, which a plain kernel answers with the written data."""
    make_data, make_frame, write = SHAPES[shape]
    nb_runner.create_notebook(
        [SETUP, TOTAL, make_data, make_frame, "r = total(s)", write, "# again, after the write\nr = total(s)"]
    )
    nb_runner.start_kernel()
    for run in ("first", "second"):
        nb_runner.run_all()
        assert nb_runner.peek("r") == AFTER, f"{run} Run All"


@pytest.mark.fresh_kernel
@pytest.mark.parametrize(
    "make", ["s = make(arr)", "s = (time.sleep(0.3), pl.Series(arr))[1]"], ids=["a-call", "a-slow-statement"]
)
def test_the_frame_made_by_a_slow_step_still_reads_the_array_after_a_restart(nb_runner, make):
    """Stored, it would come back after the restart as a copy of its own."""
    nb_runner.create_notebook(
        [
            SETUP,
            TOTAL,
            "def make(a):\n    time.sleep(0.3)\n    return pl.Series(a)",
            "arr = np.arange(100_000.0)",
            make,
            "arr[:10] = 1000.0",
            "r = total(s)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("float(s[0])") == "1000.0"
    assert nb_runner.peek("r") == AFTER


def test_a_frame_with_its_own_copy_keeps_its_cached_call(nb_runner, tmp_path):
    """Control: ``pd.Series(arr)`` copies, so the write leaves it alone, and
    the second call over it is the first one's hit: the function runs once."""
    calls = tmp_path / "calls"
    counted = TOTAL.replace(
        "    time.sleep(0.3)\n",
        f"    import os\n    fd = os.open({str(calls)!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # @cash:assume-safe\n"
        "    os.write(fd, b'x'); os.close(fd)\n    time.sleep(0.3)\n",
    )
    nb_runner.create_notebook(
        [
            SETUP,
            counted,
            "arr = np.arange(100_000.0)",
            "s = pd.Series(arr)",
            "r1 = total(s)",
            "arr[:10] = 1000.0",
            "r2 = total(s)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r2") == BEFORE
    assert calls.read_bytes() == b"x"
