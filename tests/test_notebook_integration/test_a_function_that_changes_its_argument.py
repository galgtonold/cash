"""A function that changes its argument in place is not "restored" as a no-op.

Round 28, r28s4, WRONG and BLOCKING for scanpy, 4/4 plain + 4/4 scanpy +
1/1 in their pipeline (``r28s4/repro/restart_inplace``). All of scanpy is
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

#: r28s4's module and cells exactly (``r28s4/repro/restart_inplace``). Two
#: simpler versions of this file passed on the broken build.
MODULE = ('import numpy as np\n'
          'def add_qc(df):\n'
          '    """mutates its argument in place, returns None -- the scanpy style"""\n'
          '    s = 0.0\n'
          '    for _ in range(40):\n'
          '        s += float(np.sqrt(df["a"].to_numpy() + 1.0).sum())\n'
          '    df["qc"] = df["a"] * 2 + s * 0\n')

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
        "the second Run All skipped the in-place change:\n"
        + nb_runner.get_raw_output(3) + nb_runner.get_raw_output(4))


def test_after_a_restart(nb_runner, tmp_path):
    _nb(nb_runner, tmp_path, "im.add_qc(df)")
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_all()
    assert EXPECT in nb_runner.get_output(4), (
        "after a restart the in-place change was skipped:\n"
        + nb_runner.get_raw_output(3) + nb_runner.get_raw_output(4))


def test_a_call_that_only_reads_its_argument_changes_nothing(nb_runner, tmp_path):
    """The control: `print(df)` must not be taken for a mutation, or every
    cell reading `df` would recompute on every run."""
    (tmp_path / "inplacemod.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook([
        SETUP + "\n%cash_badge print", MAKE, "print(df)",
        "total = float(df['a'].sum()) + sum(i * i for i in range(2_000_000)) % 1\n"
        "print('R', total)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(4)
    assert "CACHED: total" in raw, (
        "print(df) was treated as changing df, and the cell below re-ran:\n" + raw)
