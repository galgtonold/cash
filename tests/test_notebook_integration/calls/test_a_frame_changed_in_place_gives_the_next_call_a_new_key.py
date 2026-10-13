"""A polars frame, or a pandas frame of Python strings, changed in place is seen by the calls after it.

A call statement's argument is now read once per statement, its digest
handed to the call and back (`DigestHandoff`), also when a known in-place
verdict is checked again on new inputs (`MutationClassifier.start_rewatch`);
a polars frame is kept by its buffers' addresses within a statement, and a
pandas frame of Python strings by copy-on-write. None of it may hide a
change: one made by an earlier cell, or by the call itself on inputs where
it did not change anything before.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_on"
MODULE = """import polars as pl


def total(df):
    if isinstance(df, pl.DataFrame):
        return float(df["a"].sum())
    return float(df["a"].sum()) + len(df["s"].str.cat())


def trim(df, limit):
    # Changes the frame only when it holds a value over the limit.
    if isinstance(df, pl.DataFrame):
        if df["a"].max() > limit:
            df[0, "a"] = 0.0
    elif df["a"].max() > limit:
        df.loc[0, "a"] = 0.0
        df.loc[0, "s"] = ""
    return len(df)
"""
POLARS = "events = pl.DataFrame({'a': pl.Series(range(200_000), dtype=pl.Float64) + START})"
TEXT = (
    "with pd.option_context('mode.string_storage', 'python'):\n"
    "    events = pd.DataFrame({'a': np.arange(200_000.0) + START,\n"
    "                           's': pd.Series(['ab'] * 200_000, dtype='str')})"
)


@pytest.mark.parametrize("make", [POLARS, TEXT], ids=["polars", "python-strings"])
def test_a_change_by_an_earlier_cell_or_by_the_call_is_seen(nb_runner, make):
    (nb_runner.work_dir / "inplace_lib.py").write_text(MODULE, encoding="utf-8")
    text = 0 if make == POLARS else 2 * 200_000
    nb_runner.create_notebook(
        [
            SETUP,
            "import numpy as np, pandas as pd, polars as pl, inplace_lib\nSTART = 0.0\nLIMIT = 10**9\n" + make,
            "edits = 0",
            "print('T', inplace_lib.total(events))",
            "n = inplace_lib.trim(events, LIMIT)",
            "print('U', inplace_lib.total(events))",
        ]
    )
    nb_runner.start_kernel()
    base = sum(range(200_000))
    nb_runner.run_all()
    assert f"T {float(base + text)}" in nb_runner.get_output(4)
    assert f"U {float(base + text)}" in nb_runner.get_output(6)

    nb_runner.set_cell_source(3, "events[0, 'a'] = 7.0" if make == POLARS else "events.loc[0, 'a'] = 7.0")
    nb_runner.run_cells([3, 4])
    assert f"T {float(base + text + 7)}" in nb_runner.get_output(4), "an edit by an earlier cell went unseen"

    # The same statement on new inputs: the call that changed nothing before
    # now trims the frame, and its known verdict is checked again.
    nb_runner.set_cell_source(3, "edits = 0")
    nb_runner.set_cell_source(2, nb_runner.get_cell_source(2).replace("LIMIT = 10**9", "LIMIT = 5"))
    nb_runner.run_all()
    trimmed = base + (text - 2 if make != POLARS else 0)
    assert f"U {float(trimmed)}" in nb_runner.get_output(6), "a change by the call went unseen"
