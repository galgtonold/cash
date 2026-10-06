"""A ``state_token()`` that returns an array or a frame is keyed by its content.

The token was keyed by ``str(token)``: numpy and pandas print a long array
with its middle elided and floats rounded to 8 digits, so a change in the
middle of an array, or in the 9th digit of a float, kept the key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash import DataSource

TOKEN: dict = {}


class Partitions(DataSource):
    def get_id(self) -> str:
        return "partitions"

    def state_token(self):
        return TOKEN["value"]


def _moved(base):
    if isinstance(base, pd.Series):
        moved = base.copy()
        moved.iloc[1000] += 60.0
        return moved
    moved = base.copy()
    moved[len(moved) // 2] = moved[len(moved) // 2] + 1e-10 if len(moved) == 1 else moved[len(moved) // 2] + 60.0
    return moved


@pytest.mark.parametrize(
    "base",
    [
        np.arange(2000, dtype=float) + 1.7e9,
        pd.Series(np.arange(2000, dtype=float) + 1.7e9),
        np.array([0.123456789]),
    ],
    ids=["long-array", "series", "ninth-digit"],
)
@pytest.mark.parametrize("how", ["depends_on", "dynamic_depends_on"])
def test_a_change_the_printed_form_hides_recomputes(cash_instance, base, how):
    TOKEN["value"] = base
    runs: list = []
    kwargs = {"depends_on": [Partitions()]} if how == "depends_on" else {"dynamic_depends_on": lambda: Partitions()}

    @cash_instance.cache(assume_safe=True, **kwargs)
    def total():
        runs.append(1)
        return len(runs)

    total()
    total()
    assert len(runs) == 1
    TOKEN["value"] = _moved(base)
    assert str(TOKEN["value"]) == str(base)  # what the old key saw: no change
    total()
    assert len(runs) == 2
