"""A polars frame restored from disk is kept in RAM without a copy, and a
small frame cut from a big text column is copied by its rows.

p08's hourly table (5,760 rows, 23 chunks) is a group-by of a 3.7M-row log
frame. Each chunk of its text columns listed data buffers of ~1 GB in all:
the RAM tier copied the frame through the Arrow IPC stream, which writes
every buffer a chunk lists, 7.6 s on each hit of an etl script run (Linux;
16 s against 5.4 plain on Windows). A frame read from disk was meant to be
kept as a clone, but the restore only looked for pandas tables
(`memory_backend._frames_in`); and a text chunk is now copied by its rows
(`frame_sharing._owned_chunk`).
"""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

from polars.testing import assert_frame_equal

from cash.backends import frame_sharing, memory_backend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


def _big_texts(n: int = 200_000) -> pa.Array:
    return pa.array(
        [None if i % 101 == 0 else f"line {i} of a long log, past twelve bytes" for i in range(n)],
        type=pa.string_view(),
    )


def _bytes_of(chunk: pa.Array) -> int:
    return sum(b.size for b in chunk.buffers() if b is not None)


@pytest.mark.parametrize("view", [pa.string_view(), pa.binary_view()])
def test_a_text_chunk_is_copied_by_its_rows(view):
    big = _big_texts().cast(pa.large_string()).cast(view) if view == pa.binary_view() else _big_texts()
    chunk = pa.concat_arrays([big.slice(i * 50_000, 50_000) for i in range(4)]).slice(123_456, 300)

    owned = frame_sharing._owned_chunk(chunk)

    assert owned.equals(chunk)
    assert owned.type == chunk.type and owned.null_count == chunk.null_count
    assert not frame_sharing._overlapping(owned.buffers(), chunk.buffers())
    assert _bytes_of(owned) < 100 * len(chunk) < _bytes_of(chunk)


def _hourly_like() -> pl.DataFrame:
    """Few rows in many chunks, each over the big column's data buffers."""
    texts = _big_texts()
    pieces = [pa.table({"host": texts.slice(i * 7_000, 250), "n": pa.array(range(250))}) for i in range(23)]
    return pl.from_arrow(pa.concat_tables(pieces), rechunk=False)


def test_a_small_frame_over_big_buffers_is_stored_equal_and_apart():
    frame = _hourly_like()
    stored = memory_backend._copy_polars(frame)

    assert_frame_equal(stored, frame)
    assert stored.n_chunks("all") == frame.n_chunks("all")
    mine = [b for c in frame.to_arrow(compat_level=pl.CompatLevel.newest()).column("host").chunks for b in c.buffers()]
    theirs = [
        b for c in stored.to_arrow(compat_level=pl.CompatLevel.newest()).column("host").chunks for b in c.buffers()
    ]
    assert not frame_sharing._overlapping(theirs, mine)


def test_a_frame_read_from_disk_is_kept_in_ram_without_a_copy(tmp_path, monkeypatch):
    frame = _hourly_like()
    writer = FileBackend(str(tmp_path), flush_interval=0)
    writer.set("k", frame, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()

    copies = []
    real = memory_backend._owned_polars
    monkeypatch.setattr(memory_backend, "_owned_polars", lambda f: copies.append(f) or real(f))
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])
    _meta, first = tiers.get("k")
    meta, again = tiers.get("k")
    tiers.shutdown()

    assert copies == []
    assert meta["source"] == "RAM"
    assert_frame_equal(first, frame)
    assert_frame_equal(again, frame)


def test_what_the_reader_does_to_a_restored_frame_stays_out(tmp_path):
    frame = _hourly_like()
    writer = FileBackend(str(tmp_path), flush_interval=0)
    writer.set("k", frame, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])

    _meta, first = tiers.get("k")
    first[0, "n"] = -1
    first.insert_column(0, pl.Series("new", [0] * first.height))
    meta, again = tiers.get("k")
    tiers.shutdown()

    assert meta["source"] == "RAM"
    assert_frame_equal(again, frame)
