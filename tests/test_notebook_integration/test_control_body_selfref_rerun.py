"""Self-referential in-place mutation nested in a control structure must not
accumulate on an isolated re-run (CAS-57).

`if cond: df['a'] = df['a']*2` (also for/while/with bodies) re-run in isolation
previously DOUBLED again: the simulator treats a control structure as an opaque
unit and the downstream-advancement fallback collapsed the recorded lineage back
to the cell-entry base, so the stale-value guard saw base == current and declined
to reset. The fix compares the live VALUE's `_cash_lineage_hash` (which survives
the `reset_to` collapse) against the cell-entry base, so the receiver is restored.
A new column derived in a control body (`if c: df['b'] = df['a']+1`) is idempotent
and keeps its per-statement cache (CAS-42 preserved).
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

DF = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})"


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


def test_conditional_self_mutation(nb_runner):
    _rerun(nb_runner, DF,
           "if True:\n    df['a'] = df['a'] * 2\nprint(df['a'].tolist())", "[2, 4, 6]")


def test_conditional_augmented_self_mutation(nb_runner):
    _rerun(nb_runner, DF,
           "if True:\n    df['a'] += 100\nprint(df['a'].tolist())", "[101, 102, 103]")


def test_with_block_self_mutation(nb_runner):
    _rerun(nb_runner, DF,
           "import pandas as pd\nwith pd.option_context('mode.chained_assignment', None):\n"
           "    df['a'] = df['a'] * 2\nprint(df['a'].tolist())", "[2, 4, 6]")


def test_for_loop_self_mutation(nb_runner):
    _rerun(nb_runner, DF,
           "for col in ['a']:\n    df[col] = df[col] * 2\nprint(df['a'].tolist())", "[2, 4, 6]")


def test_nested_control_self_mutation(nb_runner):
    _rerun(nb_runner, DF,
           "for col in ['a']:\n    if True:\n        df[col] = df[col] * 2\n"
           "print(df['a'].tolist())", "[2, 4, 6]")


def test_conditional_del_column(nb_runner):
    nb_runner.create_notebook([
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
        "if True:\n    del df['b']\nprint(list(df.columns))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "['a']" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_cell(2)
    assert "['a']" in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


def test_conditional_new_column_preserved(nb_runner):
    """CAS-42 guard: a new column derived in a conditional is idempotent and must
    keep working (not over-reset)."""
    _rerun(nb_runner, DF,
           "if True:\n    df['b'] = df['a'] + 1\nprint(df['b'].tolist())", "[2, 3, 4]")


def test_top_level_still_works(nb_runner):
    """Regression: the top-level (non-nested) self-mutation path still resets."""
    _rerun(nb_runner, DF,
           "df['a'] = df['a'] * 2\nprint(df['a'].tolist())", "[2, 4, 6]")
