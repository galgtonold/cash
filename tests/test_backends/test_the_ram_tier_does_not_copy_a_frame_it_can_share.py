"""The RAM tier shares a pandas frame instead of copying it, when that is safe.

Measured after round 28 (cash's own first-run cost): 20 statements each
producing a 1M-row frame took 0.44 s under cash against 0.09 s plain, and
0.28 s of it was the RAM tier deep-copying every frame on store -- and again on
every restore (0.125 s of the warm run). The copy exists so that a later
in-place edit cannot reach the cached value. With pandas copy-on-write, which
is always on from pandas 3, a SHALLOW copy already guarantees that: the first
write to either side copies the data then, and only the part written.
"""

import numpy as np
import pandas as pd
import pytest

from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".")[0]) < 3, reason="copy-on-write is always on only from pandas 3"
)


def _frame():
    return pd.DataFrame({"x": np.arange(100_000, dtype=float), "s": ["a"] * 100_000})


def test_a_stored_frame_shares_its_data_until_written():
    df = _frame()
    copy = InMemoryBackend._safe_deep_copy(df)
    assert np.shares_memory(copy["x"].to_numpy(), df["x"].to_numpy()), (
        "the frame was copied in full; under copy-on-write a shallow copy is safe"
    )


def test_writing_either_side_does_not_reach_the_other():
    df = _frame()
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"df": df}}, {"execution_time": 1.0})
    df.loc[0, "x"] = -1.0  # the user edits their frame
    _meta, stored = backend.get("k")
    restored = stored["variables"]["df"]
    assert restored.loc[0, "x"] == 0.0, "an edit after the store reached the cache"
    restored.loc[1, "x"] = -2.0  # the user edits the restored frame
    _meta, again = backend.get("k")
    assert again["variables"]["df"].loc[1, "x"] == 1.0, "an edit to a restored frame reached the cache"


def test_a_frame_inside_an_entry_is_shared_too():
    df = _frame()
    copy = InMemoryBackend._safe_deep_copy({"variables": {"df": df, "n": [1, 2]}})
    assert np.shares_memory(copy["variables"]["df"]["x"].to_numpy(), df["x"].to_numpy())
    assert copy["variables"]["n"] == [1, 2]


def test_a_frame_inside_a_returned_tuple_is_shared_too():
    """A call's result is often a tuple: ``frame, n = build()``. deepcopy
    copied the frame in it deep."""
    df = _frame()
    copy = InMemoryBackend._safe_deep_copy((df, 3, "x"))
    assert np.shares_memory(copy[0]["x"].to_numpy(), df["x"].to_numpy())
    assert copy[1:] == (3, "x") and copy[0] is not df
