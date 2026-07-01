"""In-place mutation through a numpy slice-VIEW or a pandas df.loc row-append
must reset on isolated re-run instead of accumulating (CAS-74)."""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


# --- numpy slice-view augmented assign -------------------------------------

def test_numpy_view_augassign(nb_runner):
    _rerun(nb_runner, "import numpy as np\narr = np.array([0, 10, 20, 30])",
           "v = arr[1:]\nv += 1\nprint(arr.tolist())", "[0, 11, 21, 31]")


def test_numpy_view_full_slice(nb_runner):
    _rerun(nb_runner, "import numpy as np\narr = np.array([1, 2, 3])",
           "v = arr[:]\nv *= 2\nprint(arr.tolist())", "[2, 4, 6]")


def test_numpy_view_subscript_assign(nb_runner):
    _rerun(nb_runner, "import numpy as np\narr = np.array([1, 2, 3, 4])",
           "v = arr[::2]\nv[0] = 99\nprint(arr.tolist())", "[99, 2, 3, 4]")


def test_list_slice_is_copy_preserved(nb_runner):
    # a list slice is a COPY, not a view -> mutating it must NOT affect the base,
    # and the base stays unchanged across re-runs (no spurious behaviour).
    _rerun(nb_runner, "base = [1, 2, 3]",
           "v = base[1:]\nv.append(99)\nprint(base, v)", "[1, 2, 3] [2, 3, 99]")


# --- pandas df.loc row append ----------------------------------------------

def test_pandas_loc_row_append(nb_runner):
    _rerun(nb_runner, "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
           "df.loc[len(df)] = 99\nprint('len', len(df))", "len 4")


def test_pandas_loc_new_col_preserved(nb_runner):
    # df.loc[:, 'b'] = df['a'] * 2 is a new column from another -> idempotent,
    # should keep working (not a row-grow).
    _rerun(nb_runner, "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
           "df.loc[:, 'b'] = df['a'] * 2\nprint('cols', list(df.columns))", "cols ['a', 'b']")
