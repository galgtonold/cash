"""A column of many small chunks goes out of band as one buffer.

Each buffer under `OUT_OF_BAND_MIN_BYTES` is pickled into the stream. p08's
log frame has 1,280 chunks per column (30 files read on 32 threads): all of
its 708 MB went into the stream, copied as it grew, 6.6 s to store on
Windows, and read back from it again (1.1 s against 0.5 s here). The small
buffers of a column are now copied into one pack, which protocol 5 hands
out of band; each chunk comes back over its slice of it.
"""

from __future__ import annotations

import pickle

import pytest

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

from polars.testing import assert_frame_equal

from cash.backends import serialization
from cash.backends.file_backend import FileBackend
from cash.backends.serialization import OUT_OF_BAND_MIN_BYTES, PickleSerializer

CHUNKS = 400
ROWS = 1_000


def _texts(n: int, at: int = 0) -> pa.Array:
    return pa.array(
        [None if i % 37 == 0 else f"line {at + i} of a log, past twelve bytes" for i in range(n)],
        type=pa.string_view(),
    )


def _frame() -> pl.DataFrame:
    pieces = [
        pa.table(
            {
                "t": _texts(ROWS, i * ROWS),
                "n": pa.array(range(i * ROWS, (i + 1) * ROWS), type=pa.int64()),
                "f": pa.array([None if j % 5 == 0 else j / 3 for j in range(ROWS)], type=pa.float32()),
                "b": pa.array([j % 3 == 0 for j in range(ROWS)]),
            }
        )
        for i in range(CHUNKS)
    ]
    return pl.from_arrow(pa.concat_tables(pieces), rechunk=False)


def test_the_frame_has_the_small_chunks_the_test_is_about():
    frame = _frame()
    arrow = frame.to_arrow(compat_level=pl.CompatLevel.newest())

    assert frame.n_chunks("all") == [CHUNKS] * 4
    assert all(b.size < OUT_OF_BAND_MIN_BYTES for c in arrow.columns for ch in c.chunks for b in ch.buffers() if b)


def test_the_small_buffers_go_out_of_band_and_the_frame_comes_back_as_it_was():
    frame = _frame()
    stream, buffers = PickleSerializer().serialize_split(frame, copy=False)
    back = PickleSerializer().deserialize_split(stream, buffers)

    assert len(stream) < 0.05 * sum(memoryview(b).nbytes for b in buffers)
    assert_frame_equal(back, frame)
    assert back.schema == frame.schema
    assert back.n_chunks("all") == frame.n_chunks("all")
    assert back.null_count().row(0) == frame.null_count().row(0)


def test_slices_with_offsets_and_nulls_come_back_row_for_row():
    numbers = pa.array([None if i % 7 == 0 else i for i in range(5_000)], type=pa.int32())
    texts = _texts(5_000)
    for whole in (numbers, texts):
        column = pa.chunked_array([whole.slice(i * 37 + 3, 30) for i in range(100)], type=whole.type)
        stream, buffers = PickleSerializer().serialize_split(column, copy=False)
        back = PickleSerializer().deserialize_split(stream, buffers)

        assert back.equals(column)
        assert [len(c) for c in back.chunks] == [len(c) for c in column.chunks]
        assert [c.null_count for c in back.chunks] == [c.null_count for c in column.chunks]


def test_a_plain_pickle_of_the_column_loads():
    column = _frame().to_arrow(compat_level=pl.CompatLevel.newest()).column("t")

    assert pickle.loads(PickleSerializer().serialize(column)).equals(column)


def test_an_entry_written_before_the_packing_still_loads():
    column = pa.chunked_array([_texts(100).slice(i * 10, 10) for i in range(10)])
    rebuild, args = _unpacked(column)

    assert pickle.loads(pickle.dumps((rebuild, args)))[0](*args).equals(column)


def _unpacked(column):
    """The ``_chunked_array`` arguments of a column, as written before the packing."""
    index, buffers, layout = {}, [], []
    for chunk in column.chunks:
        refs = []
        for buffer in chunk.buffers():
            if buffer is None:
                refs.append(None)
                continue
            key = (buffer.address, buffer.size)
            if key not in index:
                index[key] = len(buffers)
                buffers.append(buffer)
            refs.append(index[key])
        layout.append((len(chunk), chunk.null_count, chunk.offset, tuple(refs)))
    return serialization._chunked_array, (column.type, buffers, layout)


def test_the_file_tier_restores_the_frame(tmp_path):
    frame = _frame()
    writer = FileBackend(str(tmp_path), flush_interval=0)
    writer.set("k", frame, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    reader = FileBackend(str(tmp_path), flush_interval=0)
    _meta, back = reader.get("k")
    reader.shutdown()

    assert_frame_equal(back, frame)
    assert back.n_chunks("all") == frame.n_chunks("all")


def test_the_key_is_not_the_written_bytes(monkeypatch):
    from cash.value_hash import compute_hash

    frame = _frame()
    table = frame.to_arrow(compat_level=pl.CompatLevel.newest())
    keys = (compute_hash(frame), compute_hash(table), compute_hash({"t": table}))
    real = serialization.stored_reducers
    monkeypatch.setattr(
        serialization,
        "stored_reducers",
        lambda **kw: {k: v for k, v in (real(**kw) or {}).items() if k is not pa.ChunkedArray},
    )

    assert keys == (compute_hash(frame), compute_hash(table), compute_hash({"t": table}))
    assert compute_hash(frame) == compute_hash(frame.rechunk())
