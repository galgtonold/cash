"""A frame's column of a million one-item lists, one of them also bound to a
variable, is share-checked by count, not a list at a time.

With nothing else holding a list, the column was read at C speed; once one
list had a second holder, the walk took every list as a node of its own and
the closure check asked each one for a function: 18 s a check. A list held
by its column alone and holding numbers alone is no part of anything else,
so it is left out by its count; the one held elsewhere is still found, and
so is a list or dict inside a cell.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cash.notebook import shared_objects as so


def _frame():
    rows = [[i] if i % 3 else i * 1.5 for i in range(3000)] + [{"k": 1}, [[1]], "s"]
    return pd.DataFrame({"a": pd.Series(rows, dtype=object)})


@pytest.mark.parametrize(
    ("extra", "holders"),
    [
        (None, set()),
        (lambda df: df["a"].iloc[4], {"held"}),
        (lambda df: df["a"].iloc[3001][0], {"held"}),
        (lambda df: df["a"].iloc[3000], {"held"}),
    ],
    ids=["nothing held", "a list of the column", "a list inside a cell", "a dict of the column"],
)
def test_what_else_holds_a_cell_is_found(extra, holders):
    df = _frame()
    ns = {"df": df}
    if extra is not None:
        ns["held"] = extra(df)
    found, shared = so.share_group(["df"], ns, ns)
    assert (set(found), shared) == (holders, set())


def test_the_lone_lists_are_not_taken_one_at_a_time(monkeypatch):
    df = pd.DataFrame({"a": pd.Series([[i] for i in range(3000)], dtype=object)})
    ns = {"df": df, "held": df["a"].iloc[4]}
    asked = [0]
    children_of = so.children_of

    def counted(value):
        asked[0] += 1
        return children_of(value)

    monkeypatch.setattr(so, "children_of", counted)
    found, shared = so.share_group(["df"], ns, ns)
    assert set(found) == {"held"} and not shared
    assert asked[0] < 500, asked[0]
