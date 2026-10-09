"""A table whose object columns hold lists or dicts is kept with those cells as bytes.

Copying such cells one container at a time on the store and on every hit
cost more than building the table (1.4 s per hit for a million rows of
two-item lists). Kept as marshal bytes, a hit reads new cells out at C
speed. The cells still come back as new objects every time, and sharing
the copy must keep -- a cell list another name also holds -- still makes
the tier copy them the old way, which keeps it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend, _TableWithBytes
from cash.backends.tiered_backend import TieredBackend


def _frame(n: int = 2000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "tags": [[i % 7, i % 11] for i in range(n)],
            "label": [f"x{i}" for i in range(n)],
            "meta": [{"score": i * 0.5, "seen": [i]} for i in range(n)],
            "when": [pd.Timestamp("2024-01-01").date()] * n,
        },
        index=pd.Index(np.arange(n) * 3, name="row"),
    )


def _stored(backend, key="k"):
    return backend._store[key][1]


def test_its_list_columns_are_kept_as_bytes_and_come_back_equal():
    df = _frame()
    backend = InMemoryBackend()
    backend.set("k", df, {"execution_time": 1.0})
    assert type(_stored(backend)) is _TableWithBytes
    hit = backend.get("k")[1]
    pd.testing.assert_frame_equal(hit, _frame())
    assert list(hit.columns) == list(df.columns) and hit.index.name == "row"


def test_every_hit_gets_cells_of_its_own():
    df = _frame()
    backend = InMemoryBackend()
    backend.set("k", df, {"execution_time": 1.0})
    first, second = backend.get("k")[1], backend.get("k")[1]
    assert first["tags"].iloc[0] is not second["tags"].iloc[0]
    assert first["tags"].iloc[0] is not df["tags"].iloc[0]
    first["tags"].iloc[0].append(99)
    first["meta"].iloc[1]["seen"].append(99)
    df["tags"].iloc[2].append(99)  # the caller's own table, after the store
    pd.testing.assert_frame_equal(backend.get("k")[1], _frame())
    pd.testing.assert_frame_equal(second, _frame())


@pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")
def test_its_other_columns_are_shared_frozen():
    backend = InMemoryBackend()
    backend.set("k", pd.DataFrame({"tags": [[1], [2], [3]], "w": [1.0, 2.0, 3.0]}), {"execution_time": 1.0})
    hit, other = backend.get("k")[1], backend.get("k")[1]
    assert np.shares_memory(hit["w"].to_numpy(), other["w"].to_numpy())
    with pytest.raises(ValueError):
        hit["w"].array[0] = -1.0  # read-only: the memory is shared
    hit.loc[0, "w"] = -1.0  # pandas copies first
    hit["tags"].iloc[0].append(9)
    again = backend.get("k")[1]
    assert again["w"].iloc[0] == 1.0 and again["tags"].iloc[0] == [1]


def test_a_series_of_lists_keeps_its_name_index_and_attrs():
    s = pd.Series([[1, 2], [3], []], index=["a", "b", "c"], name="tags")
    s.attrs["source"] = "x"
    backend = InMemoryBackend()
    backend.set("k", s, {"execution_time": 1.0})
    assert type(_stored(backend)) is _TableWithBytes
    hit = backend.get("k")[1]
    pd.testing.assert_series_equal(hit, s)
    assert hit.attrs == {"source": "x"}
    hit.iloc[0].append(5)
    assert backend.get("k")[1].iloc[0] == [1, 2]


def test_a_table_inside_an_entry_comes_back_with_the_rest():
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"df": _frame(), "n": [1, 2]}}, {"execution_time": 1.0})
    got = backend.get("k")[1]
    pd.testing.assert_frame_equal(got["variables"]["df"], _frame())
    assert got["variables"]["n"] == [1, 2]
    pd.testing.assert_frame_equal(backend.peek_entry("k")[1]["variables"]["df"], _frame())


class TestSharingTheCopyMustKeep:
    def test_a_cell_another_name_holds_stays_shared_with_it(self):
        df = _frame(50)
        backend = InMemoryBackend()
        backend.set("k", {"variables": {"df": df, "first": df["tags"].iloc[0]}}, {"execution_time": 1.0})
        got = backend.get("k")[1]["variables"]
        assert got["first"] is got["df"]["tags"].iloc[0]

    def test_one_list_in_two_cells_stays_one_list(self):
        shared = [1, 2]
        df = pd.DataFrame({"tags": [shared, shared, [3]]})
        backend = InMemoryBackend()
        backend.set("k", df, {"execution_time": 1.0})
        hit = backend.get("k")[1]
        assert hit["tags"].iloc[0] is hit["tags"].iloc[1] and hit["tags"].iloc[0] is not shared

    @pytest.mark.parametrize("cell", [{1, 2}, np.int64(3), bytearray(b"x")], ids=["set", "numpy_scalar", "bytearray"])
    def test_a_cell_marshal_cannot_write_is_copied_the_old_way(self, cell):
        df = pd.DataFrame({"c": [[cell], [1]]})
        backend = InMemoryBackend()
        backend.set("k", df, {"execution_time": 1.0})
        assert type(_stored(backend)) is not _TableWithBytes
        hit = backend.get("k")[1]
        assert hit["c"].iloc[0] == [cell] and hit["c"].iloc[0] is not df["c"].iloc[0]


def test_it_goes_to_disk_and_back_as_it_was(tmp_path):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])
    tiers.set("k", _frame(), {"execution_time": 5.0, "decorator_entry": True})
    tiers.shutdown()
    reader = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])
    meta, got = reader.get("k")
    assert meta["source"] == "DISK"
    pd.testing.assert_frame_equal(got, _frame())
