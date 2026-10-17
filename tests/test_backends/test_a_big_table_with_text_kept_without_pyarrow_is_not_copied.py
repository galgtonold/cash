"""Without pyarrow, a big table of numbers and dates is kept uncopied and
checked, as with it.

pandas 3 keeps text in a numpy array of str when pyarrow is missing, and
column names are text. The start-up check of table sharing froze a table
with a text column, failed on that kind, and turned sharing off: every
table was copied when stored and again on each hit. In a retail pipeline
the RAM tier's copies took peak memory to 3.2 GB (plain Python 1.2 GB).
Now text labels are frozen like other labels, and the check tries only the
column kinds frozen here. A table with a text column is still copied, as
before: kept uncopied, its digest would be taken again on every hit.
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

pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".", 1)[0]) < 3, reason="pandas 3 keeps text as str: an older one keeps objects"
)

ROWS = frame_sharing.CHECKED_MIN_BYTES // 24 * 11 // 10


@pytest.fixture(autouse=True)
def text_without_arrow(monkeypatch):
    """pandas keeping text as it does without pyarrow, and sharing decided
    again under it."""
    monkeypatch.setattr(frame_sharing, "_ENABLED", [])
    monkeypatch.setattr(frame_sharing, "_CHECKED_OK", [])
    with pd.option_context("mode.string_storage", "python"):
        yield


def _table(rows: int = ROWS, text: bool = False) -> pd.DataFrame:
    columns = {
        "fare": np.linspace(-1.0, 100.0, rows),
        "n": np.arange(rows),
        "when": pd.date_range("2024-01-01", periods=rows, freq="s"),
    }
    if text:
        columns["zone"] = ["a", "b", "c", "d"] * (rows // 4) + ["a"] * (rows % 4)
    raw = pd.DataFrame(columns)
    table = raw[raw["fare"] > 0]
    assert type(table.columns._data) is pd.arrays.StringArray  # text as pandas keeps it here
    return table


def _numpy(frame, column):
    return frame[column].to_numpy()


def test_tables_are_shared_without_pyarrow():
    assert frame_sharing.enabled()
    assert frame_sharing._checked_works()


def test_the_store_keeps_its_numbers_and_dates_uncopied():
    backend = InMemoryBackend()
    df = _table()
    backend.set("k", df, {"execution_time": 1.0})
    assert backend._checked["k"]
    stored = backend._store["k"][1]
    assert np.shares_memory(_numpy(stored, "fare"), _numpy(df, "fare"))


def test_a_table_with_a_text_column_is_copied_as_before():
    backend = InMemoryBackend()
    df = _table(text=True)
    expected = df.copy()
    backend.set("k", df, {"execution_time": 1.0})
    assert "k" not in backend._checked
    assert not np.shares_memory(_numpy(backend._store["k"][1], "fare"), _numpy(df, "fare"))
    df["zone"].array[0] = "q"
    hit = backend.get("k")[1]
    pd.testing.assert_frame_equal(hit, expected)
    hit["zone"].array[1] = "r"
    pd.testing.assert_frame_equal(backend.get("k")[1], expected)


MUTATORS = {
    "iloc": lambda df: df.iloc.__setitem__((0, 0), -5.0),
    "label": lambda df: df.rename(columns={"n": "count"}, inplace=True),
    "replace_column": lambda df: df.__setitem__("fare", 0.0),
    "fillna_inplace": lambda df: df.fillna({"fare": 0.0}, inplace=True),
    "rename_inplace": lambda df: df.rename(columns={"fare": "price"}, inplace=True),
}


@pytest.mark.parametrize("mutate", MUTATORS.values(), ids=MUTATORS.keys())
def test_a_change_through_pandas_never_reaches_a_hit(mutate):
    """pandas copies a block before it writes one the tier shares."""
    backend = InMemoryBackend()
    df = _table()
    expected = df.copy()
    backend.set("k", df, {"execution_time": 1.0})
    mutate(df)
    pd.testing.assert_frame_equal(backend.get("k")[1], expected)


WRITES_PAST_PANDAS = {
    "array": lambda df: df["fare"].array.__setitem__(0, 123.0),
    "numpy_handle": lambda df: df["n"].array._ndarray.__setitem__(0, -1),
    "dates": lambda df: df["when"].array._ndarray.__setitem__(0, np.datetime64("1999-01-01")),
}


@pytest.mark.parametrize("write", WRITES_PAST_PANDAS.values(), ids=WRITES_PAST_PANDAS.keys())
def test_a_change_pandas_does_not_see_drops_the_entry(write):
    """Writes into the shared memory itself, which copy-on-write does not
    copy first: the digest no longer matches, the hit is a miss."""
    backend = InMemoryBackend()
    df = _table()
    backend.set("k", df, {"execution_time": 1.0})
    write(df)
    assert backend.get("k") == (None, None)
    assert "k" not in backend._store


def test_a_hit_is_a_copy_the_caller_can_change():
    backend = InMemoryBackend()
    df = _table()
    expected = df.copy()
    backend.set("k", df, {"execution_time": 1.0})
    hit = backend.get("k")[1]
    hit.iloc[0, 0] = -7.0
    hit["n"].array[2] = -3
    hit.columns = ["a", "b", "c"]
    pd.testing.assert_frame_equal(backend.get("k")[1], expected)
    pd.testing.assert_frame_equal(df, expected)


def test_a_hit_after_the_caller_lets_go_matches_and_can_be_changed():
    backend = InMemoryBackend()
    df = _table()
    expected = df.copy()
    backend.set("k", {"variables": {"df": df}}, {"execution_time": 1.0})
    df = None
    gc.collect()
    first = backend.get("k")[1]["variables"]["df"]
    pd.testing.assert_frame_equal(first, expected)
    first.iloc[0, 1] = -9
    with pytest.raises(ValueError):
        first["fare"].array[0] = -9.0  # frozen: shared by every hit, no handle writes it
    with pytest.raises(ValueError):
        first.columns.array[0] = "x"  # its labels too
    pd.testing.assert_frame_equal(backend.get("k")[1]["variables"]["df"], expected)


def test_with_the_callers_table_gone_the_disk_copy_matches(tmp_path):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    df = _table()
    expected = df.copy()
    metadata = {"execution_time": 50.0}
    tiers.set("k", df, metadata)
    assert "DISK" in metadata["storage"]
    del df
    gc.collect()
    tiers.shutdown()  # the disk write finished
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)  # a new session's
    _meta, read = disk.get("k")
    disk.shutdown()
    pd.testing.assert_frame_equal(read, expected)
