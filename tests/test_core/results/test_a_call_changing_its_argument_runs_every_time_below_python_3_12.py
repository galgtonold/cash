"""Below Python 3.12 a call that changes its argument in place runs every
time, and says so.

A hit writes back the state a call left its argument in only when nothing
else the call sees shares memory with that argument. What else it sees is
read from the record of what the call ran, which sys.monitoring (3.12+)
makes. Without it there is no exact answer: reading the function's text
misses a global read through a method, another module or a decorated
helper, and a hit would serve a result the body never computed. So the
result is not stored, the call runs every time, and the warning names why.
The 3.12+ behaviour is pinned in
`test_the_first_call_that_changes_its_argument_in_place_is_stored.py`.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(sys.version_info >= (3, 12), reason="the record of what a call ran is there")


def _sort(rows):
    rows.sort()
    return rows[0]


def _clip(a):
    np.clip(a, 0, 1, out=a)
    return float(a.sum())


def _bump(df):
    df.iloc[0, 0] = df.iloc[0, 0] + 1
    return float(df["a"].sum())


CASES = {
    "list": (_sort, lambda: [3, 1, 2], lambda v: v),
    "array": (_clip, lambda: np.array([-1.0, 0.5, 2.0]), lambda v: v.tolist()),
    "frame": (_bump, lambda: pd.DataFrame({"a": [1.0, 2.0]}), lambda v: v["a"].tolist()),
}
#: Changed inside a library, which the source does not show. ``rows.sort()``
#: and ``df.iloc[0, 0] = ...`` are read in the source and said as found there.
IN_A_LIBRARY = {"array"}


@pytest.mark.parametrize("case", list(CASES))
def test_it_runs_every_time_with_the_warning(disk_cash, case):
    body, make, seen = CASES[case]
    runs = []

    def changes(value):
        runs.append(1)  # @cash:assume-safe
        return body(value)

    # A name of its own per case: a warning is said once per function.
    changes.__qualname__ = changes.__name__ = f"changes_{case}"
    changes = disk_cash.cache(changes)

    plain = make()
    expected = body(plain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            value = make()
            assert changes(value) == expected
            assert seen(value) == seen(plain), "the argument is not left as a plain call leaves it"
    assert len(runs) == 3, "a call that changes its argument in place was served"
    assert changes.cache_info()["hits"] == 0
    # What it said, kept on the function (a warning is shown once).
    said = [w["message"] for w in changes.cache_info()["warnings"]]
    assert any("runs every time" in m for m in said), f"it ran every time without saying why: {said}"
    if case in IN_A_LIBRARY:
        # A change made inside a library: only running it shows it, and
        # the reason it is not stored names the missing record.
        (observed,) = [m for m in said if "IMPURE-OBSERVED-EFFECTS" in m]
        assert "'value' in place" in observed
        assert "recorded on Python 3.12 or newer only" in observed
