"""A polars frame whose text column's chunks share their data buffers stores
each buffer once, and comes back as it was.

Each chunk of a polars text column (Arrow ``string_view``) lists every data
buffer of the column, not only the ones its rows point into. Pickled chunk
by chunk, each buffer went into the entry once per chunk: p08's cleaned log
frame, filtered on 32 threads into 32 chunks, was 438 MB in memory and 10.3
GB in its entry on Windows, written for 39 s beside the next cell. A column
is now pickled with its shared buffers once (`_reduce_chunked_array`).
"""

from __future__ import annotations

import pickle

import pytest

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

from polars.testing import assert_frame_equal, assert_series_equal

from cash.backends import serialization
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import PickleSerializer
from cash.backends.tiered_backend import TieredBackend

ROWS = 64_000
CHUNKS = 32


def _texts(n: int) -> pa.Array:
    """Strings past the 12 bytes a view holds inline, over many data buffers."""
    return pa.array(
        [None if i % 97 == 0 else f"row {i} of the log, past twelve bytes" for i in range(n)], type=pa.string_view()
    )


def _sliced(array: pa.Array, chunks: int = CHUNKS) -> pa.ChunkedArray:
    """*array* cut into *chunks* slices, each over all of its buffers, as polars' chunks are."""
    step = len(array) // chunks
    return pa.chunked_array([array.slice(i * step, step) for i in range(chunks)], type=array.type)


def _frame() -> pl.DataFrame:
    texts = _texts(ROWS)
    table = pa.table(
        {
            "text": _sliced(texts),
            "binary": _sliced(
                pa.array([f"bytes {i} and some more of them".encode() for i in range(ROWS)], type=pa.binary_view())
            ),
            "n": _sliced(pa.array(range(ROWS), type=pa.int64())),
            "flag": _sliced(pa.array([i % 3 == 0 if i % 5 else None for i in range(ROWS)], type=pa.bool_())),
        }
    )
    return pl.from_arrow(table, rechunk=False)


def _sizes(value) -> tuple[int, int]:
    stream, buffers = PickleSerializer().serialize_split(value, copy=False)
    return len(stream), sum(memoryview(b).nbytes for b in buffers)


def _one_copy(frame: pl.DataFrame) -> int:
    """The bytes of every distinct buffer under *frame*'s columns."""
    seen = {}
    for column in frame.to_arrow(compat_level=pl.CompatLevel.newest()).columns:
        for chunk in column.chunks:
            for buffer in chunk.buffers():
                if buffer is not None:
                    seen[(buffer.address, buffer.size)] = buffer.size
    return sum(seen.values())


def test_the_frame_has_the_chunks_and_shared_buffers_the_test_is_about():
    frame = _frame()
    arrow = frame.to_arrow(compat_level=pl.CompatLevel.newest())

    assert frame.n_chunks("all") == [CHUNKS] * 4
    text = arrow.column("text")
    assert pa.types.is_string_view(text.type)
    assert len(text.chunks[0].buffers()) > 3, "the slices list several data buffers"
    assert {b.address for b in text.chunks[0].buffers()[2:]} == {b.address for b in text.chunks[-1].buffers()[2:]}


def test_shared_buffers_are_stored_once():
    frame = _frame()
    stream, apart = _sizes(frame)

    assert stream + apart < 1.3 * _one_copy(frame), (stream, apart, _one_copy(frame))


def test_the_frame_comes_back_equal_with_its_dtypes_and_chunks():
    frame = _frame()
    stream, buffers = PickleSerializer().serialize_split(frame, copy=False)
    back = PickleSerializer().deserialize_split(stream, buffers)

    assert_frame_equal(back, frame)
    assert back.schema == frame.schema
    assert back.n_chunks("all") == frame.n_chunks("all")
    assert back.null_count().row(0) == frame.null_count().row(0)


def test_a_series_and_a_plain_pickle_come_back_equal():
    series = pl.from_arrow(_sliced(_texts(ROWS)), rechunk=False).alias("s")
    stream, buffers = PickleSerializer().serialize_split(series, copy=False)

    assert_series_equal(PickleSerializer().deserialize_split(stream, buffers), series)
    assert_series_equal(pickle.loads(PickleSerializer().serialize(series)), series)


def test_slices_at_odd_offsets_with_nulls_come_back_row_for_row():
    texts = _texts(10_001)
    column = pa.chunked_array([texts.slice(0, 3), texts.slice(3, 4_000), texts.slice(4_003, 1), texts.slice(4_004)])
    stream, buffers = PickleSerializer().serialize_split(pa.table({"t": column}), copy=False)
    back = PickleSerializer().deserialize_split(stream, buffers).column("t")

    assert back.equals(column)
    assert [len(c) for c in back.chunks] == [len(c) for c in column.chunks]
    assert back.null_count == column.null_count


def test_a_column_of_one_chunk_or_a_nested_type_is_pickled_as_pyarrow_does():
    one = pa.chunked_array([_texts(1_000)])
    nested = pa.chunked_array([pa.array([[1, 2], None, [3]]), pa.array([[4]])])
    dictionary = pa.chunked_array([pa.array(["a", "b", "a"]).dictionary_encode()] * 2)

    for column in (one, nested, dictionary):
        stream, buffers = PickleSerializer().serialize_split(column, copy=False)
        assert PickleSerializer().deserialize_split(stream, buffers).equals(column)
        assert serialization._reduce_chunked_array(column)[0] is not serialization._chunked_array


def test_the_file_tier_restores_the_frame(tmp_path):
    frame = _frame()
    writer = FileBackend(str(tmp_path), flush_interval=0)
    writer.set("k", frame, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    reader = FileBackend(str(tmp_path), flush_interval=0)  # a new process
    _meta, back = reader.get("k")
    reader.shutdown()

    assert_frame_equal(back, frame)
    assert back.n_chunks("all") == frame.n_chunks("all")
    assert sum(f.stat().st_size for f in tmp_path.rglob("*") if f.is_file()) < 1.5 * _one_copy(frame)


# -- the key is not the stored bytes: only how the entry is written changed --


def _without_the_column_reducer(monkeypatch):
    real = serialization.stored_reducers
    monkeypatch.setattr(
        serialization,
        "stored_reducers",
        lambda **kw: {k: v for k, v in (real(**kw) or {}).items() if k is not pa.ChunkedArray},
    )


def test_a_frame_keys_alike_in_one_chunk_or_many_and_as_before(monkeypatch):
    from cash.value_hash import compute_hash

    chunked = _frame()
    whole = chunked.rechunk()
    keys = {compute_hash(chunked), compute_hash(whole)}
    table = chunked.to_arrow(compat_level=pl.CompatLevel.newest())
    arrow_key = compute_hash(table)
    nested_key = compute_hash({"frame": chunked, "table": table})
    _without_the_column_reducer(monkeypatch)

    assert whole.n_chunks("all") == [1] * 4
    assert keys == {compute_hash(chunked)} == {compute_hash(whole)}
    assert arrow_key == compute_hash(table)
    assert nested_key == compute_hash({"frame": chunked, "table": table})


def test_a_call_on_the_frame_in_many_chunks_hits_the_entry_of_one_chunk(tmp_path):
    import cash

    calls = []
    c = cash.Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def rows(frame):
        calls.append(1)
        return frame.height

    chunked = _frame()
    assert rows(chunked.rechunk()) == ROWS - ROWS % CHUNKS
    assert rows(chunked) == ROWS - ROWS % CHUNKS
    assert calls == [1]


def test_the_restored_frame_hashes_as_the_stored_one():
    """What keys on the frame (`hash_polars`) sees the same content."""
    from cash.content_hashers import hash_polars

    frame = _frame()
    digest = hash_polars(frame)
    stream, buffers = PickleSerializer().serialize_split(frame, copy=False)
    back = PickleSerializer().deserialize_split(stream, buffers)

    assert hash_polars(back) == digest


def test_a_frame_read_from_disk_is_served_from_ram_again(tmp_path):
    writer = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    writer.set("k", _frame(), {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "cache"), flush_interval=0)])
    _meta, first = tiers.get("k")
    meta, again = tiers.get("k")
    tiers.shutdown()

    assert_frame_equal(first, _frame())
    assert_frame_equal(again, _frame())
    assert meta["source"] == "RAM"
