"""A column of a million Timestamps is share-checked by count, not one by one,
once a pandas scalar has had an attribute set.

From then on pandas scalars are objects to the share check
(`scalar_identity`), and each Timestamp of an object column was walked as
its own node: 12 s a check for a million rows, against 0.2 s for a column
of lists. A Timestamp is a leaf there, so only its own count matters, and
those are read at C speed as a lone record's are. One held by a variable
is still found.

Run in a fresh interpreter: the switch is for the whole process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pandas")

CHECK = textwrap.dedent(
    """
    import pandas as pd
    from cash.notebook import shared_objects as so

    so.library_value_types()
    so.turn_on_scalar_identity()
    asked = [0]
    children_of = so.children_of

    def counted(value):
        asked[0] += 1
        return children_of(value)

    so.children_of = counted

    def frame():
        start = pd.Timestamp("2023-01-01")
        return pd.DataFrame({"at": pd.Series([start + pd.Timedelta(seconds=i) for i in range(50_000)], dtype=object)})

    df = frame()
    ns = {"df": df}
    assert so.share_group(["df"], ns, ns) == ({}, set())
    assert asked[0] < 1_000, asked[0]

    held = df["at"].iloc[7]
    ns = {"df": df, "held": held}
    holders, shared = so.share_group(["df"], ns, ns)
    assert set(holders) == {"held"} and not shared, (holders, shared)
    print("ok")
    """
)


def test_the_column_is_read_by_count_and_a_held_item_is_found():
    done = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
