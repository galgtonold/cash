"""Arrow memory only Arrow holds is shared with the stored copy.

The RAM tier's copy of a table copied every Arrow chunk, because Arrow data
can be the caller's numpy memory (``pa.array(a)`` copies nothing). Storing
a 1 GB Arrow-backed table then took seconds of fresh allocations. A buffer
a memory pool allocated -- read from a file, computed by Arrow or pandas --
is written by no one, so a chunk made only of such buffers is now shared,
and only a chunk with a buffer over someone else's memory is copied.
`frame_sharing.arrow_owned` tells the two apart by the C++ class of the
buffer; where it cannot, everything is copied as before.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from pandas.arrays import ArrowExtensionArray

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend

ROWS = 50_000


def _addresses(chunked) -> set[int]:
    return {b.address for chunk in chunked.chunks for b in chunk.buffers() if b is not None and b.size}


def _stored(value):
    ram = InMemoryBackend()
    ram.set("k", value, {"execution_time": 60.0})
    return ram.get("k")[1]


def test_buffers_are_told_apart_on_this_python():
    """Without the layout every chunk is copied: correct, but the slow path."""
    assert frame_sharing._arrow_layout() is not None


def test_pool_memory_is_arrow_s_own():
    pooled = pa.array([1, 2, 3])
    assert frame_sharing.arrow_owned(pooled.buffers()[1])
    assert frame_sharing.arrow_owned(pooled.buffers()[1].slice(8))
    assert frame_sharing.arrow_owned(pa.allocate_buffer(64))
    assert frame_sharing.arrow_owned(pa.concat_arrays([pa.array(np.arange(10))]).buffers()[1])


def test_numpy_python_and_imported_memory_is_not():
    a = np.arange(100)
    assert not frame_sharing.arrow_owned(pa.array(a).buffers()[1])
    assert not frame_sharing.arrow_owned(pa.py_buffer(a))
    assert not frame_sharing.arrow_owned(pa.py_buffer(a).slice(8))  # a slice of numpy's memory
    assert not frame_sharing.arrow_owned(pa.py_buffer(bytearray(64)))
    assert not frame_sharing.arrow_owned(pa.foreign_buffer(a.ctypes.data, a.nbytes, base=a))


def test_an_arrow_stream_over_python_bytes_is_not_arrow_s_own():
    sink = pa.BufferOutputStream()
    table = pa.table({"x": np.arange(100)})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    data = bytearray(sink.getvalue().to_pybytes())
    read = pa.ipc.open_stream(pa.py_buffer(data)).read_all()
    assert not frame_sharing._chunk_owned(read["x"].chunk(0))


def test_a_table_read_from_a_file_keeps_its_arrow_memory():
    """Read by Arrow's reader. pandas' own reader builds numbers in numpy
    and wraps them (``pa.array``), so those columns are copied."""
    text = "name,n\n" + "".join(f"w{i},{i}\n" for i in range(ROWS))
    frame = pd.read_csv(io.BytesIO(text.encode()), engine="pyarrow", dtype_backend="pyarrow")
    stored = _stored(frame)
    for name in frame.columns:
        mine, kept = frame[name].array._pa_array, stored[name].array._pa_array
        assert kept.equals(mine)
        assert _addresses(kept) == _addresses(mine), f"{name} was copied"


def test_a_pandas_string_column_keeps_its_arrow_memory():
    words = pd.Series([f"w{i}" for i in range(ROWS)], dtype=pd.ArrowDtype(pa.string()))
    kept = _stored(pd.DataFrame({"s": words}))["s"].array._pa_array
    assert _addresses(kept) == _addresses(words.array._pa_array)


def test_only_the_chunk_over_numpy_memory_is_copied():
    a = np.arange(ROWS, dtype=np.int64)
    column = pa.chunked_array([pa.array(a), pa.array(list(range(10)))])
    frame = pd.DataFrame({"x": ArrowExtensionArray(column)})
    kept = _stored(frame)["x"].array._pa_array
    assert kept.num_chunks == 2
    assert kept.chunk(0).buffers()[1].address != a.ctypes.data
    assert kept.chunk(1).buffers()[1].address == column.chunk(1).buffers()[1].address
    a[:] = -1
    assert kept.chunk(0).to_pylist()[:3] == [0, 1, 2]


def test_a_dictionary_over_numpy_codes_is_copied():
    words = pa.array(["lo", "hi"])
    column = pa.DictionaryArray.from_arrays(pa.array(np.arange(ROWS, dtype=np.int32) % 2), words)
    assert not frame_sharing._chunk_owned(column)
    owned = pa.DictionaryArray.from_arrays(pa.array([0, 1, 0], pa.int32()), words)
    assert frame_sharing._chunk_owned(owned)


def test_a_write_into_numpy_under_arrow_reaches_no_stored_copy():
    a = np.arange(ROWS, dtype=np.int64)
    frame = pa.table({"x": a, "s": pa.array([str(i) for i in range(ROWS)])}).to_pandas(types_mapper=pd.ArrowDtype)
    expected = frame["x"].tolist()
    stored = _stored(frame)
    a[:] = -1
    assert frame["x"].tolist() != expected, "the table does not view the array: the test checks nothing"
    assert stored["x"].tolist() == expected
    assert _addresses(stored["s"].array._pa_array) == _addresses(frame["s"].array._pa_array)


def test_a_polars_frame_over_arrow_over_numpy_is_copied():
    pl = pytest.importorskip("polars")
    a = np.arange(ROWS, dtype=np.int64)
    frame = pl.from_arrow(pa.table({"x": pa.array(a)}))
    expected = frame["x"].to_list()
    stored = _stored(frame)
    a[:] = -1
    assert frame["x"].to_list() != expected, "the frame does not view the array: the test checks nothing"
    assert stored["x"].to_list() == expected


def test_polars_memory_is_never_taken_for_arrow_s_own():
    """polars exports its own memory and numpy's alike: both are copied."""
    pl = pytest.importorskip("polars")
    computed = pl.Series("x", [1, 2, 3]) * 2
    assert not frame_sharing._chunk_owned(computed.to_arrow())
