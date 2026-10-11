"""A table whose object cells are a mix of immutable values is not copied
through a pickle round trip.

``infer_dtype`` calls ints beside strings "mixed-integer" and floats (NaN)
beside dates "mixed", which read as "may hold a list": ``df2.loc['action_time']
= ...`` added a row by label to a 1.7-million-row frame, and the RAM tier
copied it through pickle, 12 s against 4.4 s plain. The cells' exact types
answer instead; a cell that can change still sends the table through pickle.
"""

from __future__ import annotations

import datetime

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.backends import memory_backend  # noqa: E402 - imported after the importorskip above
from cash.backends.memory_backend import InMemoryBackend, immutable_cells  # noqa: E402


def test_a_mix_of_immutable_cells_is_immutable():
    assert immutable_cells(np.array([1, "a", 2.5, None, datetime.date(2020, 1, 1)], dtype=object))
    assert immutable_cells(np.array([pd.Timestamp(1), float("nan"), pd.NaT, np.int64(3)], dtype=object))


def test_a_changeable_cell_or_a_subclass_is_not():
    class Tag(str):
        pass

    assert not immutable_cells(np.array([1, "a", [1]], dtype=object))
    assert not immutable_cells(np.array([1, Tag("a")], dtype=object))


def test_a_row_added_by_label_keeps_the_table_off_the_pickle_round_trip(monkeypatch):
    df = pd.DataFrame({"user_id": ["a", "b"], "time": pd.to_datetime(["2023-01-01", "2023-01-02"])})
    df.loc["time"] = ["c", pd.Timestamp("2023-01-03")]
    round_trips = []
    real = memory_backend._round_trip
    monkeypatch.setattr(memory_backend, "_round_trip", lambda v: round_trips.append(1) or real(v))

    backend = InMemoryBackend()
    backend.set("k", df, {"execution_time": 1.0})
    hit = backend.get("k")[1]

    assert not round_trips
    pd.testing.assert_frame_equal(hit, df)


def test_a_list_among_mixed_cells_is_still_copied():
    df = pd.DataFrame({"x": [1, "a", [1, 2]]})
    backend = InMemoryBackend()
    backend.set("k", df, {"execution_time": 1.0})
    df["x"].iloc[2].append(3)

    assert backend.get("k")[1]["x"].iloc[2] == [1, 2]


def test_a_list_of_timestamps_is_copied_without_pickling():
    """``list(df['action_time'])``: 1.7 million Timestamps went through a
    pickle round trip for the RAM tier, 16 s after a 2.9 s statement. A new
    list holding the same immutable Timestamps is a complete copy."""
    stamps = list(pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02"] * 50)))

    backend = InMemoryBackend()
    backend.set("k", {"variables": {"stamps": stamps}}, {"execution_time": 1.0})
    stamps.append(pd.Timestamp("2024-01-01"))
    hit = backend.get("k")[1]["variables"]["stamps"]

    assert len(hit) == 100
    assert hit[0] is stamps[0], "the Timestamps are shared, not pickled"
