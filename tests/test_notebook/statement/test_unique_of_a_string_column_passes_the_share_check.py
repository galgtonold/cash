"""``schemes = df["card_scheme"].unique()`` is not refused by the share check.

The array ``unique()`` gives back holds the column's ``StringDtype``, which
every string column and pandas' dtype cache hold too. The search for
variables holding what a library object holds climbed its referrers, read
the heap past its budget, and the statement ran uncached every time. A
pandas dtype is a description pandas treats as immutable, as numpy's dtypes
are values here, so it is not looked for
(`shared_objects._is_extension_dtype`).
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import share_group, walk_budget


def _namespace() -> dict:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"card_scheme": ["GlobalCard", "NexPay", "SwiftCharge"] * 2000})
    return {"pd": pd, "df": df, "schemes": df["card_scheme"].unique()}


def test_the_dtype_a_unique_array_shares_with_its_column_is_not_looked_for():
    ns = _namespace()
    assert isinstance(ns["schemes"].dtype, ns["pd"].api.extensions.ExtensionDtype)
    assert ns["schemes"].dtype is ns["df"]["card_scheme"].dtype
    # No time to climb the heap: the check must not need to.
    with walk_budget(0.0):
        holders, shared = share_group(["schemes"], {"schemes": ns["schemes"]}, ns)
    assert not holders
    assert not shared
