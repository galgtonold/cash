"""``for day, counts in day_counts.items(): weekday_df[day] = pd.Series(counts)``
does not fingerprint ``day_counts`` around the loop.

The pandas constructors were not among the library calls known to only
read what they are handed, so the loop counted as handing ``day_counts``'
lists to unknown code: a Series of a million dates in lists was hashed
before and after the loop, 7 s plain against 34 s. ``pd.Series``,
``pd.DataFrame`` and ``pd.Index`` build a new object from their data and
never write into it.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from cash.analysis.mutation_effects import loop_watch_candidates
from cash.analysis.read_only_callees import reads_its_arguments_only

NAMESPACE = {
    "pd": pd,
    "day_counts": pd.Series([[1, 2], [3]], index=["Monday", "Tuesday"]),
    "weekday_df": pd.DataFrame(),
    "frames": [],
}


@pytest.mark.parametrize(
    "loop",
    [
        "for day, counts in day_counts.items():\n    weekday_df[day] = pd.Series(counts)",
        "for day, counts in day_counts.items():\n    frames.append(pd.DataFrame({day: counts}))",
        "for day, counts in day_counts.items():\n    frames.append(pd.Index(counts))",
    ],
    ids=["Series", "DataFrame", "Index"],
)
def test_the_data_a_constructor_is_handed_is_not_watched(loop):
    node = ast.parse(loop).body[0]
    assert "day_counts" not in loop_watch_candidates(node, NAMESPACE, lambda name: name == "pd")


def test_a_class_of_the_same_name_is_not_taken_for_pandas():
    class Series:
        def __init__(self, data):
            data.append(0)

    assert reads_its_arguments_only(pd.Series, ([1],))
    assert not reads_its_arguments_only(Series, ([1],))
