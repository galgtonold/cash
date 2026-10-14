"""Arrow data over the caller's numpy memory is copied when stored.

``pa.array(a)`` over a numpy array of numbers copies nothing, and neither
does ``pa.table({"x": a}).to_pandas(types_mapper=pd.ArrowDtype)``: the
column's Arrow buffer is ``a``'s memory. The RAM tier kept such a column as
it was (Arrow memory is not written through pandas), so ``a[:] = ...`` after
the store changed what every hit returned, and what the disk write, still
queued, pickled. pyarrow cannot tell a buffer its memory pool allocated from
one over memory someone else writes, so the stored copy copies Arrow data.
A polars value, Arrow data too, is copied the same way where it was cloned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from pandas.arrays import ArrowExtensionArray

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

ROWS = 50_000

VALUES = {
    "table.to_pandas": lambda a: pa.table({"x": a}).to_pandas(types_mapper=pd.ArrowDtype),
    "ArrowExtensionArray": lambda a: pd.DataFrame({"x": ArrowExtensionArray(pa.array(a))}),
    "series": lambda a: pd.Series(ArrowExtensionArray(pa.array(a)), name="x"),
    "with a nullable column": lambda a: pd.DataFrame(
        {"x": ArrowExtensionArray(pa.array(a)), "n": pd.array(np.ones(len(a), dtype=int), dtype="Int64")}
    ),
    "in a tuple": lambda a: (pd.DataFrame({"x": ArrowExtensionArray(pa.array(a))}), {"rows": len(a)}),
    "as the index": lambda a: pd.DataFrame({"y": np.zeros(len(a))}, index=pd.Index(ArrowExtensionArray(pa.array(a)))),
    "as categories": lambda a: pd.DataFrame(
        {
            "c": pd.Categorical.from_codes(
                np.arange(len(a)) % 3, categories=pd.Index(ArrowExtensionArray(pa.array(a[:3])))
            )
        }
    ),
}


def _seen(value) -> list:
    frame = value[0] if isinstance(value, tuple) else value
    if isinstance(frame, pd.Series):
        return frame.tolist()
    return [*frame.index.tolist(), *(frame[name].tolist() for name in frame.columns)]


CASES = [(name, True) for name in VALUES] + [
    # Too big for RAM: written from a copy made for the write. A categorical
    # column is not copied that way, but pickled before the store returns.
    (name, False)
    for name in VALUES
    if name != "as categories"
]


@pytest.mark.parametrize(
    ("name", "kept"), CASES, ids=[f"{name}, {'kept in RAM' if kept else 'too big for RAM'}" for name, kept in CASES]
)
def test_a_write_into_the_numpy_array_after_the_store_reaches_no_hit(name, kept, tmp_path):
    ram = InMemoryBackend(max_size_bytes=None if kept else 1000)
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    store = TieredBackend([ram, disk])
    a = np.arange(ROWS, dtype=np.int64)
    value = VALUES[name](a)
    expected = _seen(value)
    store.set("k", value, {"execution_time": 60.0, "force_persist": True})
    a[:] = -1  # the disk write may still be queued
    assert _seen(value) != expected, "the value does not view the array: the test checks nothing"
    if kept:
        assert _seen(ram.get("k")[1]) == expected
    disk.shutdown()
    assert _seen(FileBackend(str(tmp_path / "c")).get("k")[1]) == expected


def test_the_stored_copy_keeps_the_columns_chunks_and_type():
    a = np.arange(ROWS, dtype=np.int64)
    column = pa.chunked_array([pa.array(a[:10]), pa.array(a[10:])])
    frame = pd.DataFrame({"x": ArrowExtensionArray(column)})
    ram = InMemoryBackend()
    ram.set("k", frame, {"execution_time": 60.0})
    kept = ram.get("k")[1]["x"].array._pa_array
    assert kept.num_chunks == 2 and kept.type == column.type and kept.equals(column)
    for copied, original in zip(kept.chunks, column.chunks):
        assert copied.buffers()[1].address != original.buffers()[1].address


def test_a_dictionary_column_is_copied_with_its_dictionary():
    words = np.array(["lo", "hi"], dtype=object)
    column = pa.DictionaryArray.from_arrays(pa.array(np.arange(ROWS, dtype=np.int32) % 2), pa.array(words))
    frame = pd.DataFrame({"d": pd.Series(column, dtype=pd.ArrowDtype(column.type))})
    ram = InMemoryBackend()
    ram.set("k", frame, {"execution_time": 60.0})
    kept = ram.get("k")[1]["d"].array._pa_array.chunk(0)
    original = frame["d"].array._pa_array.chunk(0)
    assert kept.type == original.type and kept.equals(original)
    addresses = {buffer.address for buffer in original.buffers() + original.dictionary.buffers() if buffer is not None}
    assert not addresses & {b.address for b in kept.buffers() + kept.dictionary.buffers() if b is not None}


POLARS = {
    "DataFrame": lambda pl, a: pl.DataFrame({"x": a}),
    "Series": lambda pl, a: pl.Series("x", a),
    "from_numpy": lambda pl, a: pl.from_numpy(a.reshape(-1, 1), schema=["x"]),
    "with an Object column": lambda pl, a: pl.DataFrame({"o": pl.Series([[1]] * len(a), dtype=pl.Object), "x": a}),
    "LazyFrame": lambda pl, a: pl.DataFrame({"x": a}).lazy(),
}


def _polars_seen(pl, value) -> list:
    if isinstance(value, pl.LazyFrame):
        value = value.collect()
    return value.to_list() if isinstance(value, pl.Series) else value["x"].to_list()


@pytest.mark.parametrize("name", POLARS)
def test_a_polars_value_over_the_numpy_array_is_copied_too(name):
    """Polars data is Arrow data: ``pl.Series(a)`` copies nothing either."""
    pl = pytest.importorskip("polars")
    a = np.arange(ROWS, dtype=np.int64)
    value = POLARS[name](pl, a)
    expected = _polars_seen(pl, value)
    ram = InMemoryBackend()
    ram.set("k", value, {"execution_time": 60.0})
    a[:] = -1
    assert _polars_seen(pl, value) != expected, "the value does not view the array: the test checks nothing"
    assert _polars_seen(pl, ram.get("k")[1]) == expected
    assert _polars_seen(pl, ram.get("k")[1]) == expected
