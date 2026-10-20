"""A polars text column of many chunks hashes as the same column in one.

Every chunk of a polars text column lists all of the column's data buffers,
and the C kernel that reads the values took each of them on every call, once
per block: p08's cleaned log frame in 32 chunks took 0.78 s to hash against
0.50 s in one. The kernel now takes a buffer when a row first points into
it. And each chunk's lengths and bytes were hashed apart, a few KB at a time
on one thread: p08's raw frame in 1,280 chunks per column hashed in 0.92 s
against 0.41 s in one. They are now gathered into full blocks first, read
chunk after chunk in one call (``view_values_many``), as are the values of
its numbers (``copy_pieces``), so its columns are read side by side. The
columns come through each one's Arrow C stream, not a table of 1,280 record
batches. The digest is the one the export at the oldest compat level gives,
so keys made before still match.
"""

import pytest

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

from cash import _native, content_hashers

pytestmark = pytest.mark.skipif(_native.kernels is None, reason="no C extension")


def _exported(value):
    kernels = _native.kernels
    _native.kernels = None
    try:
        return content_hashers.hash_polars(value)
    finally:
        _native.kernels = kernels


def _texts(n, buffers):
    """*n* long strings spread over *buffers* data buffers, some null."""
    parts = []
    step = n // buffers
    for b in range(buffers):
        parts.append(
            pa.array(
                [None if i % 89 == 0 else f"row {i} of part {b}, past the inline bytes" for i in range(step)],
                type=pa.string_view(),
            )
        )
    return pa.concat_arrays(parts)


def _chunked(array, chunks):
    step = len(array) // chunks
    return pa.chunked_array([array.slice(i * step, step) for i in range(chunks)], type=array.type)


@pytest.mark.parametrize("chunks", [1, 3, 32])
def test_the_digest_is_the_exported_one_and_the_same_in_one_chunk(chunks):
    texts = _texts(40_000, buffers=8)
    frame = pl.from_arrow(
        pa.table({"t": _chunked(texts, chunks), "n": _chunked(pa.array(range(len(texts))), chunks)}), rechunk=False
    )

    assert frame.n_chunks("all") == [chunks, chunks]
    assert content_hashers.hash_polars(frame) == _exported(frame) == content_hashers.hash_polars(frame.rechunk())


def test_rows_that_point_back_to_an_earlier_buffer_hash_as_exported():
    texts = pl.from_arrow(_texts(9_000, buffers=6))
    back = texts.gather(list(range(len(texts) - 1, -1, -2)) + list(range(len(texts) - 2, -1, -2)))
    chunked = pl.concat([back.slice(i * 1_000, 1_000) for i in range(9)], rechunk=False)

    assert chunked.n_chunks() == 9
    assert content_hashers.hash_polars(chunked) == _exported(chunked) == content_hashers.hash_polars(back)
    assert content_hashers.hash_polars(chunked) != content_hashers.hash_polars(texts)


@pytest.mark.parametrize("block", [64, 1000, 4096])
def test_blocks_filled_across_chunks_and_rows_cut_at_a_block_end_hash_alike(monkeypatch, block):
    """Tiny blocks fill and are hashed many times within a chunk and across
    chunks, and cut long values in the middle."""
    texts = _texts(6_000, buffers=4)
    frame = pl.from_arrow(pa.table({"t": _chunked(texts, 300)}), rechunk=False)
    want = _exported(frame)
    monkeypatch.setattr(content_hashers._VIEW_BLOCKS, "blocks", (bytearray(block), bytearray(block)), raising=False)

    assert content_hashers.hash_polars(frame) == want
    assert content_hashers.hash_polars(frame.rechunk()) == want


def test_many_small_chunks_hash_as_one():
    texts = _texts(25_600, buffers=4)
    frame = pl.from_arrow(
        pa.table({"t": _chunked(texts, 1_280), "n": _chunked(pa.array(range(len(texts))), 1_280)}), rechunk=False
    )

    assert frame.n_chunks("all") == [1_280, 1_280]
    assert content_hashers.hash_polars(frame) == _exported(frame) == content_hashers.hash_polars(frame.rechunk())


def test_short_and_empty_values_need_no_buffer():
    series = pl.Series("s", ["", "a", None, "x" * 12] * 1000)

    assert content_hashers.hash_polars(series) == _exported(series)


@pytest.mark.parametrize("block", [7, 64, 4096])
def test_numbers_of_many_small_chunks_gathered_into_blocks_hash_as_one(monkeypatch, block):
    n = pa.array(range(30_000), type=pa.int64())
    f = pa.array([i / 7 for i in range(30_000)], type=pa.float32())
    frame = pl.from_arrow(pa.table({"n": _chunked(n, 300), "f": _chunked(f, 300)}), rechunk=False)
    frame = frame.slice(5, 29_000)
    want = _exported(frame)
    monkeypatch.setattr(content_hashers._VIEW_BLOCKS, "blocks", (bytearray(block), bytearray(block)), raising=False)

    assert content_hashers.hash_polars(frame) == want == content_hashers.hash_polars(frame.rechunk())


def _as_a_table(frame, pl):
    return frame.to_arrow(compat_level=pl.CompatLevel.newest()).columns


def _cut_apart():
    """A frame whose Null column polars cuts at other rows than the rest."""
    schema = {"l": pl.List(pl.Int64), "nul": pl.Null, "dec": pl.Decimal(10, 2), "i": pl.Int64}

    def part(n):
        return pl.DataFrame(
            {"l": [[1]] * n, "nul": [None] * n, "dec": pl.Series(["1.5"] * n).cast(pl.Decimal(10, 2)), "i": range(n)},
            schema=schema,
        )

    return pl.concat([part(50), part(150)], rechunk=False)


def test_a_frame_chunked_apart_keys_as_its_table(monkeypatch):
    """``to_arrow`` cuts every column at the same rows first, in the frame
    itself, and a list, Null or decimal column hashed after that is keyed by
    those cuts (its IPC stream): the frame is still exported as a table."""
    frame = _cut_apart()
    assert len({tuple(s.chunk_lengths()) for s in frame.get_columns()}) > 1
    got = [content_hashers.hash_polars(frame)] + [content_hashers.hash_polars(s) for s in frame.get_columns()]

    monkeypatch.setattr(content_hashers, "_newest_arrow_columns", _as_a_table)
    frame = _cut_apart()
    assert got == [content_hashers.hash_polars(frame)] + [content_hashers.hash_polars(s) for s in frame.get_columns()]


def test_a_column_is_read_from_its_own_stream_chunk_for_chunk():
    frame = pl.from_arrow(
        pa.table({"t": _chunked(_texts(3_000, buffers=2), 30), "n": _chunked(pa.array(range(3_000)), 30)}),
        rechunk=False,
    )
    columns = content_hashers._newest_arrow_columns(frame, pl)

    assert [c.num_chunks for c in columns] == [30, 30]
    assert [c.type for c in columns] == [c.type for c in _as_a_table(frame, pl)]
