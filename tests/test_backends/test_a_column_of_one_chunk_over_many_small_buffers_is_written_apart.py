"""A text column of one chunk over many small buffers goes out of band too.

A sorted polars frame has one chunk per column, and its text column points
into every small data buffer of the frames it came from: p08's cleaned log
frame, 6,151 buffers in one column. Only a column of two chunks or more had
its small buffers packed (`_reduce_chunked_array`); this one went to
pyarrow's own pickle, its 320 MB of text into the stream, and read back
from it as a copy on every hit (clean's hit 1.4-1.6 s, now 1.0-1.1 s).
"""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

from polars.testing import assert_frame_equal

from cash.backends.file_backend import FileBackend
from cash.backends.serialization import OUT_OF_BAND_MIN_BYTES, PickleSerializer

PIECES = 300
ROWS = 100


def _texts() -> pa.Array:
    return pa.concat_arrays(
        [
            pa.array(
                [
                    None if (i * ROWS + j) % 37 == 0 else f"line {i * ROWS + j} of a log, past twelve bytes"
                    for j in range(ROWS)
                ],
                type=pa.string_view(),
            )
            for i in range(PIECES)
        ]
    )


def _frame() -> pl.DataFrame:
    return pl.from_arrow(pa.table({"t": _texts(), "n": pa.array(range(PIECES * ROWS))}), rechunk=False)


def test_the_column_is_one_chunk_over_many_small_buffers():
    column = _frame().to_arrow(compat_level=pl.CompatLevel.newest()).column("t")

    assert len(column.chunks) == 1
    data = [b for b in column.chunks[0].buffers()[2:] if b is not None]
    assert len(data) == PIECES
    assert all(b.size < OUT_OF_BAND_MIN_BYTES for b in data)


def test_its_small_buffers_go_out_of_band_and_it_comes_back_as_it_was():
    frame = _frame()
    stream, buffers = PickleSerializer().serialize_split(frame, copy=False)
    back = PickleSerializer().deserialize_split(stream, buffers)

    assert len(stream) < 0.05 * sum(memoryview(b).nbytes for b in buffers)
    assert_frame_equal(back, frame)
    assert back.n_chunks("all") == frame.n_chunks("all")
    assert back.null_count().row(0) == frame.null_count().row(0)


def test_a_slice_of_it_comes_back_row_for_row():
    texts = _texts()
    for whole in (texts.slice(1_234, 20_000), texts.slice(7, 5)):
        column = pa.chunked_array([whole], type=whole.type)
        stream, buffers = PickleSerializer().serialize_split(column, copy=False)
        back = PickleSerializer().deserialize_split(stream, buffers)

        assert back.equals(column)
        assert back.null_count == column.null_count


def test_a_column_of_one_chunk_without_small_buffers_pickles_as_before():
    column = pa.chunked_array([pa.array(range(100_000), type=pa.int64())])
    stream, buffers = PickleSerializer().serialize_split(column, copy=False)

    assert PickleSerializer().deserialize_split(stream, buffers).equals(column)
    assert [memoryview(b).nbytes for b in buffers] == [800_000]


def test_the_file_tier_restores_the_frame(tmp_path):
    frame = _frame()
    writer = FileBackend(str(tmp_path), flush_interval=0)
    writer.set("k", frame, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    reader = FileBackend(str(tmp_path), flush_interval=0)
    _meta, back = reader.get("k")
    reader.shutdown()

    assert_frame_equal(back, frame)
