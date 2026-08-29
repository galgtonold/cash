"""An in-place edit past the sample window must still invalidate.

The notebook path's ``compute_hash`` SAMPLES: ndarray -> first 100 elements,
DataFrame -> first 5 rows. Two values that agree on the sample hash
identically, so content hashing alone cannot see an edit further in.

What covers it is mutation tracking, not hashing: ``arr[5000] = 2.0``
registers ``arr`` as mutated, its lineage bumps, and consumers recompute
regardless of what the sampled digest says. This pins that, because it is the
mechanism actually doing the work -- an earlier version of this file claimed
to be testing the sampled hash and was passing for this reason instead.

If mutation tracking ever stops covering a shape, these go red, and the
sampled hash will not save it.
"""
import pytest

pytestmark = pytest.mark.core


def test_ndarray_edited_past_the_sample_window(nb_runner):
    """Element 5000 -- far past the 100 the sampled digest reads."""
    nb_runner.create_notebook([
        "import numpy as np",
        "# @cash:no-cache\narr = np.zeros(10000)",
        "# @cash:no-cache\narr[5000] = 1.0",
        "total = float(arr.sum())\nprint('TOTAL', total)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "TOTAL 1.0" in nb_runner.get_output(4), nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "# @cash:no-cache\narr[5000] = 2.0")
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)

    out = nb_runner.get_output(4)
    assert "TOTAL 2.0" in out, (
        f"got {out!r}. The array differs only at element 5000, which the "
        f"sampled digest never reads, so mutation tracking is what has to "
        f"catch this."
    )


def test_dataframe_edited_past_the_sample_window(nb_runner):
    """Row 900 -- far past the 5 rows the sampled digest reads."""
    nb_runner.create_notebook([
        "import pandas as pd",
        "# @cash:no-cache\ndf = pd.DataFrame({'x': list(range(1000))})",
        "# @cash:no-cache\ndf.loc[900, 'x'] = -1",
        "total = int(df['x'].sum())\nprint('TOTAL', total)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "TOTAL 498599" in nb_runner.get_output(4), nb_runner.get_output(4)

    nb_runner.set_cell_source(3, "# @cash:no-cache\ndf.loc[900, 'x'] = -2")
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)

    out = nb_runner.get_output(4)
    assert "TOTAL 498598" in out, (
        f"got {out!r}. The frame differs only at row 900, past the 5-row sample."
    )
