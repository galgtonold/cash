"""A write into a stored or returned pandas frame never reaches the RAM entry.

Copy-on-write covers writes made through pandas, not the writable handles
pandas hands out: ``s.array`` of any column and ``s.values`` of a nullable or
categorical column point at the block itself. The RAM tier copied frames
shallowly, so ``df["score"].values[0] = 100`` on a returned frame changed
what every later hit returned. It copies them deep on store and on hit.
"""

import numpy as np
import pandas as pd

from cash.backends.memory_backend import InMemoryBackend


def _frame():
    return pd.DataFrame(
        {
            "score": pd.array([1, 2, 3], dtype="Int64"),
            "grade": pd.Categorical(["a", "b", "a"]),
            "w": [1.0, 2.0, 3.0],
        }
    )


def _write_through_handles(df):
    df["score"].values[0] = 100
    df["grade"].values[0] = "b"
    df["w"].array[0] = 50.0


def _unchanged(df):
    return (int(df["score"][0]), df["grade"][0], float(df["w"][0])) == (1, "a", 1.0)


def test_a_write_to_the_stored_original_does_not_reach_the_entry():
    df = _frame()
    backend = InMemoryBackend()
    backend.set("k", df, {"execution_time": 1.0})
    _write_through_handles(df)
    assert _unchanged(backend.get("k")[1])


def test_a_write_to_a_hit_does_not_reach_later_hits():
    backend = InMemoryBackend()
    backend.set("k", _frame(), {"execution_time": 1.0})
    _write_through_handles(backend.get("k")[1])
    assert _unchanged(backend.get("k")[1])


def test_a_frame_inside_an_entry_is_isolated_too():
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"df": _frame(), "n": [1, 2]}}, {"execution_time": 1.0})
    _write_through_handles(backend.get("k")[1]["variables"]["df"])
    _meta, again = backend.get("k")
    assert _unchanged(again["variables"]["df"]) and again["variables"]["n"] == [1, 2]


def test_a_returned_frame_does_not_share_its_data():
    df = pd.DataFrame({"x": np.arange(1000, dtype=float)})
    for copy in (InMemoryBackend._safe_deep_copy(df), InMemoryBackend._safe_deep_copy((df, 3, "x"))[0]):
        assert not np.shares_memory(copy["x"].to_numpy(), df["x"].to_numpy())


def test_writing_either_side_does_not_reach_the_other():
    df = pd.DataFrame({"x": np.arange(100, dtype=float)})
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"df": df}}, {"execution_time": 1.0})
    df.loc[0, "x"] = -1.0  # the user edits their frame
    _meta, stored = backend.get("k")
    restored = stored["variables"]["df"]
    assert restored.loc[0, "x"] == 0.0, "an edit after the store reached the cache"
    restored.loc[1, "x"] = -2.0  # the user edits the restored frame
    _meta, again = backend.get("k")
    assert again["variables"]["df"].loc[1, "x"] == 1.0, "an edit to a restored frame reached the cache"
