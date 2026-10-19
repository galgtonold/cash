"""The RAM tier keeps a big pandas table uncopied, checked on each hit.

It copied every table it stored, though a cleaning cell rebinds its table
statement after statement (``df = df[df.fare > 0]``) and the caller's copy
is gone a moment later. On a 2.9-million-row taxi table each copy took
0.2-2.9 s, most of it fresh pages, and doubled the memory the table took.
Now a big table of numbers and dates is kept as a shallow copy with the
digest of its data (`frame_sharing.share_checked`): pandas copies a block
before writing it on either side, a write past pandas changes the digest
and makes the next hit a miss, and once nothing but the tier reaches the
data the tier freezes it as its own (`frame_sharing.adopt`).
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

pytestmark = [
    pytest.mark.skipif(not frame_sharing._checked_works(), reason="needs pandas copy-on-write"),
    # A write past pandas drops the entry and says so (STORE-RESULT-WRITTEN).
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheStoreFailedWarning"),
]

#: Rows enough for 16 MiB of numbers and dates: kept uncopied.
ROWS = frame_sharing.CHECKED_MIN_BYTES // 24 * 11 // 10


def _table(rows: int = ROWS) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "fare": np.linspace(-1.0, 100.0, rows),
            "n": np.arange(rows),
            "when": pd.date_range("2024-01-01", periods=rows, freq="s"),
            "zone": pd.array(["a", "b", "c", "d"] * (rows // 4) + ["a"] * (rows % 4), dtype="str"),
        }
    )
    return raw[raw["fare"] > 0]  # blocks over fresh arrays, as a filter leaves them


def _store(backend, value, key="k"):
    backend.set(key, value, {"execution_time": 1.0})


def _numpy(frame, column):
    return frame[column].to_numpy()


def test_a_store_copies_none_of_its_numbers_and_dates(monkeypatch):
    copied = []
    real = frame_sharing._copy_values
    monkeypatch.setattr(frame_sharing, "_copy_values", lambda values, *a: copied.append(values) or real(values, *a))
    backend = InMemoryBackend()
    df = _table()
    _store(backend, df)
    assert backend._checked["k"]
    assert all(str(values.dtype) == "str" for values in copied)  # its Arrow text only
    stored = backend._store["k"][1]
    assert np.shares_memory(_numpy(stored, "fare"), _numpy(df, "fare"))


def test_a_hit_after_the_caller_rebinds_is_shared_and_frozen():
    backend = InMemoryBackend()
    df = _table()
    expected = df.copy()
    _store(backend, {"variables": {"df": df}})
    df = df[df["fare"] > 50]  # the cell's next statement: the stored version is the tier's alone
    gc.collect()
    first = backend.get("k")[1]["variables"]["df"]
    assert "k" not in backend._checked  # taken over
    second = backend.get("k")[1]["variables"]["df"]
    pd.testing.assert_frame_equal(first, expected)
    assert np.shares_memory(_numpy(first, "fare"), _numpy(second, "fare"))
    first.iloc[0, 0] = -7.0
    first["fare"] = 0.0
    pd.testing.assert_frame_equal(backend.get("k")[1]["variables"]["df"], expected)
    with pytest.raises(ValueError):
        second["fare"].array[0] = 5.0  # frozen: no handle writes it


MUTATORS = {
    "iloc": lambda df: df.iloc.__setitem__((0, 0), -5.0),
    "loc_column": lambda df: df.loc.__setitem__((slice(None), "n"), 0),
    "replace_column": lambda df: df.__setitem__("fare", 0.0),
    "add_column": lambda df: df.__setitem__("new", 1.0),
    "fillna_inplace": lambda df: df["fare"].fillna(0.0, inplace=True),
    "update": lambda df: df.update(pd.DataFrame({"fare": [9.0]}, index=df.index[:1])),
    "drop_inplace": lambda df: df.drop(columns=["n"], inplace=True, errors="ignore"),
}


@pytest.mark.parametrize("mutate", MUTATORS.values(), ids=MUTATORS.keys())
def test_what_the_caller_does_through_pandas_never_reaches_a_hit(mutate):
    backend = InMemoryBackend()
    df = _table()
    expected = df.copy()
    _store(backend, df)
    mutate(df)
    pd.testing.assert_frame_equal(backend.get("k")[1], expected)
    mutate(df)  # and again, after the hit
    pd.testing.assert_frame_equal(backend.get("k")[1], expected)


def test_a_write_past_pandas_after_the_store_is_a_miss():
    backend = InMemoryBackend()
    df = _table()
    _store(backend, df)
    df["fare"].array[0] = 123.0  # writes the block itself
    assert backend.get("k") == (None, None)
    assert "k" not in backend._store


def test_a_table_the_caller_still_holds_is_copied_by_its_hit():
    backend = InMemoryBackend()
    df = _table()
    _store(backend, df)
    hit = backend.get("k")[1]
    assert "k" in backend._checked  # still the caller's data
    assert not np.shares_memory(_numpy(hit, "fare"), _numpy(df, "fare"))
    df["n"].array[0] = -1  # the caller writes past pandas after that hit
    assert backend.get("k") == (None, None)


def test_a_numpy_handle_the_caller_keeps_stops_the_takeover():
    backend = InMemoryBackend()
    df = _table()
    _store(backend, df)
    handle = df["fare"].array
    del df
    gc.collect()
    hit = backend.get("k")[1]
    assert "k" in backend._checked  # the handle can still write the data
    handle[0] = 5.0
    assert hit["fare"].iloc[0] != 5.0  # the hit was a copy
    assert backend.get("k") == (None, None)


def test_a_table_with_an_object_column_is_copied():
    backend = InMemoryBackend()
    df = _table()
    df["tags"] = pd.Series([("a",)] * len(df), index=df.index, dtype=object)
    _store(backend, df)
    assert "k" not in backend._checked
    assert not np.shares_memory(_numpy(backend._store["k"][1], "fare"), _numpy(df, "fare"))


def test_a_small_table_is_copied():
    backend = InMemoryBackend()
    df = _table(1000)
    _store(backend, df)
    assert "k" not in backend._checked


def test_the_disk_tier_gets_the_table_with_its_check(tmp_path):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    df = _table()
    expected = df.copy()
    metadata = {"execution_time": 50.0}
    tiers.set("k", df, metadata)
    assert "DISK" in metadata["storage"]
    ram = tiers.backends[0]
    assert ram.private_copy("k", None) is None  # not the tier's own: written with a check
    stored, check = ram.checked_copy("k", ram._store["k"][0])
    assert check()
    df["fare"].array[0] = -1.0
    assert not check()
    ram.delete("k")
    _meta, read = tiers.get("k")  # written before the change, or not at all
    tiers.shutdown()
    assert read is None or read.equals(expected)
