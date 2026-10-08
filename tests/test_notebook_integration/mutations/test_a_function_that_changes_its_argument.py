"""A function that changes its argument in place is not "restored" as a no-op.

A wrong answer that blocked scanpy use, 4/4 plain + 4/4 scanpy + 1/1 in a
real pipeline. All of scanpy is
written this way -- ``sc.pp.calculate_qc_metrics(adata, inplace=True)``,
``sc.tl.leiden(hv)`` -- and so is a project helper like

    def add_qc(df):
        df["qc"] = df["a"] * 2

``im.add_qc(df)`` is a statement with no output. Cash cached it, and a hit
"restored" nothing: the column was never added, and the next cell raised
``KeyError: 'qc'`` -- on the second Run All in the same kernel and after
every restart. The badge said the line produced ``sc`` (the module).

Method calls on an object (``df.sort_values(inplace=True)``) were already
routed as in-place mutations. An object passed AS AN ARGUMENT was not
considered at all when the callee is reached through a module.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

#: The reported module and cells exactly. Two
#: simpler versions of this file passed on the broken build.
MODULE = (
    "import numpy as np\n"
    "def add_qc(df):\n"
    '    """mutates its argument in place, returns None -- the scanpy style"""\n'
    "    s = 0.0\n"
    "    for _ in range(40):\n"
    '        s += float(np.sqrt(df["a"].to_numpy() + 1.0).sum())\n'
    '    df["qc"] = df["a"] * 2 + s * 0\n'
)

SETUP = "import cash\n%cash_on\nimport numpy as np, pandas as pd"
MAKE = "df = pd.DataFrame({'a': np.arange(3_000_000, dtype=float)})"
USE = "print('R', list(df.columns), float(df['qc'].sum()))"
EXPECT = "R ['a', 'qc'] 8999997000000.0"


def _nb(nb_runner, tmp_path, call):
    (tmp_path / "inplacemod.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook([SETUP + "\nimport inplacemod as im", MAKE, call, USE])
    nb_runner.start_kernel()


def test_run_all_twice_in_one_kernel(nb_runner, tmp_path):
    _nb(nb_runner, tmp_path, "im.add_qc(df)")
    nb_runner.run_all()
    assert EXPECT in nb_runner.get_output(4), nb_runner.get_raw_output(4)
    nb_runner.run_all()
    assert EXPECT in nb_runner.get_output(4), (
        "the second Run All skipped the in-place change:\n" + nb_runner.get_raw_output(3) + nb_runner.get_raw_output(4)
    )


def test_after_a_restart(nb_runner, tmp_path):
    _nb(nb_runner, tmp_path, "im.add_qc(df)")
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    assert EXPECT in nb_runner.get_output(4), (
        "after a restart the in-place change was skipped:\n" + nb_runner.get_raw_output(3) + nb_runner.get_raw_output(4)
    )


def test_a_call_that_only_reads_its_argument_changes_nothing(nb_runner, tmp_path):
    """The control: `print(df)` must not be taken for a mutation, or every
    cell reading `df` would recompute on every run."""
    (tmp_path / "inplacemod.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook(
        [
            SETUP + "\n%cash_badge print",
            MAKE,
            "print(df)",
            "total = float(df['a'].sum()) + sum(i * i for i in range(2_000_000)) % 1\nprint('R', total)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(4)
    assert "CACHED: total" in raw, "print(df) was treated as changing df, and the cell below re-ran:\n" + raw


#: A helper of the notebook's that changes its argument and returns a summary:
#: the result is kept, and the change in place must happen on every run.
HELPER = (
    "import time\n"
    "def add_features(df):\n"
    "    time.sleep(0.3)\n"
    "    df['x2'] = df['a'] * 2\n"
    "    return df[['x2']].describe()"
)
FRAME = "df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})"
COLUMNS = "print('C', list(df.columns))"


@pytest.mark.parametrize(
    "call",
    [
        "summary = add_features(df)",
        "summaries = [add_features(f) for f in [df]]",
        "for f in [df]:\n    s = add_features(f)",
    ],
)
def test_a_helper_whose_result_is_kept_changes_its_argument_on_every_run_all(nb_runner, call):
    nb_runner.create_notebook([SETUP, HELPER, FRAME, call, COLUMNS])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "C ['a', 'x2']" in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    nb_runner.run_all()
    assert nb_runner.peek("list(df.columns)") == "['a', 'x2']", (
        "the second Run All restored the result and skipped the change:\n" + nb_runner.get_raw_output(4)
    )


@pytest.mark.fresh_kernel
def test_a_helper_whose_result_is_kept_changes_its_argument_after_a_restart(nb_runner):
    nb_runner.create_notebook([SETUP, HELPER, FRAME, "summary = add_features(df)", COLUMNS])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("list(df.columns)") == "['a', 'x2']", (
        "after a restart the change was skipped:\n" + nb_runner.get_raw_output(4)
    )
