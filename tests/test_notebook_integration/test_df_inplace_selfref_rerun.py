"""Self-referential in-place DataFrame mutation must not accumulate on an
isolated re-run (CAS-54).

`df['a'] = df['a'] * 2` (or `df['a'] += 1`, `df.iloc[i,j] += x`) re-run in
isolation -- especially when the mutation and its display are in the SAME cell --
previously DOUBLED again because in-place-mutated lineage-carrying receivers were
excluded from the cell-entry reset (to preserve per-statement cache, the CAS-42
design). The fix detects SELF-REFERENTIAL subscript/attr writes (target also read
in the RHS) and routes them through the same reset used for method receivers
(CAS-53). A write to a NEW column read from OTHER columns (`df['VolAdj'] =
df.groupby('Close')...`) is NOT self-referential and keeps its cache (CAS-42).
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _last(out: str) -> str:
    return out.strip().splitlines()[-1].strip() if out.strip() else ""


def _rerun_one_cell(nb_runner, setup, mutate_and_show, expect):
    """The realistic pattern: mutation and display in ONE cell, re-run that cell."""
    nb_runner.create_notebook([setup, mutate_and_show])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first run: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


DF8 = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8]})"
DF100 = "import pandas as pd\ndf = pd.DataFrame({'a': list(range(100))})"


def test_column_scale_same_cell(nb_runner):
    _rerun_one_cell(nb_runner, DF8, "df['a'] = df['a'] * 2\nprint(df['a'].tolist())",
                    "[2, 4, 6, 8, 10, 12, 14, 16]")


def test_column_augmented_same_cell(nb_runner):
    _rerun_one_cell(nb_runner, DF8, "df['a'] += 100\nprint(df['a'].tolist())",
                    "[101, 102, 103, 104, 105, 106, 107, 108]")


def test_column_method_self_ref_same_cell(nb_runner):
    _rerun_one_cell(nb_runner, "import pandas as pd\ndf = pd.DataFrame({'a': [1.0, None, 3.0]})",
                    "df['a'] = df['a'].fillna(0)\nprint(df['a'].tolist())", "[1.0, 0.0, 3.0]")


def test_iloc_scalar_increment_same_cell(nb_runner):
    _rerun_one_cell(nb_runner, DF100,
                    "df.iloc[50, 0] = df.iloc[50, 0] + 1000\nprint(df.iloc[50, 0])", "1050")


def test_iloc_augmented_same_cell(nb_runner):
    _rerun_one_cell(nb_runner, DF100, "df.iloc[2, 0] += 1000\nprint(df.iloc[2, 0])", "1002")


def test_loc_scalar_self_ref_same_cell(nb_runner):
    _rerun_one_cell(nb_runner, DF100,
                    "df.loc[10, 'a'] = df.loc[10, 'a'] * 3\nprint(df.loc[10, 'a'])", "30")


def test_loc_masked_self_ref_same_cell(nb_runner):
    """CAS-55: masked .loc write reads the SAME column spelled differently than the
    target (df.loc[mask,'a'] = df['a']*2). Self-referential -> must not double."""
    _rerun_one_cell(nb_runner, DF100,
                    "df.loc[df['a'] >= 50, 'a'] = df['a'] * 2\nprint(df['a'].iloc[50])", "100")


def test_loc_masked_augmented_same_cell(nb_runner):
    """CAS-55 control: masked .loc AugAssign df.loc[mask,'a'] += 1000."""
    _rerun_one_cell(nb_runner, DF100,
                    "df.loc[df['a'] >= 50, 'a'] += 1000\nprint(df['a'].iloc[50])", "1050")


def test_loc_masked_new_column_preserved(nb_runner):
    """CAS-42 guard: a masked write to a DIFFERENT column read from another is
    idempotent and must keep working (not over-reset)."""
    _rerun_one_cell(
        nb_runner, "import pandas as pd\ndf = pd.DataFrame({'a': list(range(100))})",
        "df.loc[df['a'] >= 50, 'b'] = df['a'] * 2\nprint(df['b'].iloc[50])", "100")


def test_new_column_from_other_columns_preserved(nb_runner):
    """CAS-42 guard: a NEW column read from OTHER columns is idempotent and must
    keep working (not over-reset, not broken)."""
    _rerun_one_cell(nb_runner, "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})",
                    "df['c'] = df['a'] + df['b']\nprint(df['c'].tolist())", "[11, 22, 33]")


def test_split_cell_still_works(nb_runner):
    """Regression: mutation and display in SEPARATE cells still resets correctly."""
    nb_runner.create_notebook([DF8, "df['a'] = df['a'] * 2", "print(df['a'].tolist())"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[2, 4, 6, 8, 10, 12, 14, 16]" in nb_runner.get_output(3)
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "[2, 4, 6, 8, 10, 12, 14, 16]" in nb_runner.get_output(3), nb_runner.get_output(3)
