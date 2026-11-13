"""A frame with a column of lists is share-checked by count, statement after
statement, and its identity check stops at the walk's budget without
opening a list at a time.

Cash's frame memo held every list of a hashed frame once more, and after
``df.sort_values(..., inplace=True)`` its copy of the old block held each
again: to the share check every list read as held by a variable it could
not find. Each cheap statement on such a frame (a sort, a ``cumcount``, a
filter) searched for that holder past its budget, a second and more beside
a big notebook, was refused and re-ran. The memo now holds no list, and a
copy of a block of lists the frame replaced is let go before the check.

A column of more lists than the identity walk looks through is refused as
before, told from one read of the column, not after opening each list up
to the budget.

Pins the work, not only the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd

from cash.decorator.arg_hashing import cow_frame_digest, is_cow_pandas
from cash.notebook import shared_objects as so
from cash.notebook.statement import identity_holders
from tests._cell_driver import run_cash_cell

#: Big enough for the frame memo (a megabyte of blocks).
ROWS = 70_000

#: Past the identity walk's budget (200,000 objects) by a quarter.
PAST_THE_WALK = 250_000


def _frame(rows: int = ROWS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k": [(i * 7919) % rows for i in range(rows)],
            "items": pd.Series([["p", "q", "r"][: i % 3 + 1] for i in range(rows)], dtype=object),
        }
    )


def test_hashing_a_frame_adds_no_reference_to_its_lists():
    df = _frame()
    before = list(map(sys.getrefcount, df["items"]))
    first = cow_frame_digest(df)
    assert list(map(sys.getrefcount, df["items"])) == before
    if not is_cow_pandas(df):
        # Without copy-on-write (pandas 2) there is no frame memo: the
        # frame is read each time, and nothing of it is kept.
        assert first is None
        return
    assert first is not None
    # The memo answers the second hash: what it keeps still vouches for it.
    assert cow_frame_digest(df) == first
    df.loc[5, "items"].append("z")
    assert cow_frame_digest(df) != first


def _count_excess(monkeypatch) -> list[int]:
    """How many objects each count of the share check found held from
    outside what it walked."""
    found: list[int] = []
    real = so.excess_refs

    def counted(nodes, inbound, ids):
        shared = real(nodes, inbound, ids)
        found.append(len(shared))
        return shared

    monkeypatch.setattr(so, "excess_refs", counted)
    return found


def test_statements_changing_the_frame_find_no_list_held_elsewhere(cash_magics, monkeypatch):
    run_cash_cell(
        cash_magics,
        "import pandas as pd\n"
        f"df = pd.DataFrame({{'k': [(i * 7919) % {ROWS} for i in range({ROWS})], "
        f"'items': pd.Series([['p', 'q', 'r'][: i % 3 + 1] for i in range({ROWS})], dtype=object)}})",
    )
    excess = _count_excess(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "df.sort_values('k', inplace=True)")
        run_cash_cell(cash_magics, "df['n'] = df.groupby('k').cumcount()")
    codes = [getattr(w.message, "code", None) for w in caught]
    assert "NOTEBOOK-SHARE-UNCHECKED" not in codes, codes
    # Positive control: the check ran, and counted the lists.
    assert excess, "the share check never counted"
    assert max(excess) < 10, excess
    assert cash_magics.shell.user_ns["df"]["k"].is_monotonic_increasing


def test_a_column_past_the_walk_is_refused_without_opening_each_list(monkeypatch):
    df = _frame(PAST_THE_WALK)
    opened = [0]
    real = identity_holders.builtin_items

    def counted(value):
        opened[0] += 1
        return real(value)

    monkeypatch.setattr(identity_holders, "builtin_items", counted)
    reason = identity_holders.identity_held_reason("df", df, {"df": df})
    assert reason is not None and "more objects than cash looks through" in reason, reason
    assert opened[0] < 1000, opened[0]
