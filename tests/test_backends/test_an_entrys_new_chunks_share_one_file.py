"""The chunks one entry write adds share one file, a pack.

On Windows the virus scanner reads each new file the first time it is
opened: p08's first hit after a miss opened 415 chunk files at about 5 ms
each and took twice what later hits did. The chunks one write adds now go
into one file, so a hit opens one file per write that stored its chunks.
A read stays safe: the pack is renamed into place whole, each chunk is
checked against the crc32 its entry recorded, and a wrong place in the
metadata is a miss, never other data.
"""

from __future__ import annotations

import builtins
import textwrap

import pytest

np = pytest.importorskip("numpy")

from cash.backends.entry_format import PLACES_FIELD, read_entry, update_metadata_in_place
from cash.backends.file_backend import FileBackend
from tests._scripts import run_python

ROWS = 300_000  # 2.4 MB per float64 column: each one a chunk


def _backend(tmp_path):
    return FileBackend(str(tmp_path / "cache"), flush_interval=0)


def _store(backend, key, value):
    backend.set(key, value, {"execution_time": 1.0})
    backend._writes.wait_all()


def _files(tmp_path):
    return sorted((tmp_path / "cache" / "chunks").iterdir())


def _arrays(seed, n):
    rng = np.random.default_rng(seed)
    return [rng.random(ROWS) for _ in range(n)]


def _opened_chunk_files(monkeypatch, backend, key):
    opened = []
    real = builtins.open

    def counting(file, *args, **kwargs):
        if "/chunks/" in str(file).replace("\\", "/"):
            opened.append(str(file))
        return real(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting)
    try:
        got = backend.get(key)[1]
    finally:
        monkeypatch.setattr(builtins, "open", real)
    return got, opened


def test_twenty_columns_are_one_file_read_with_one_open(tmp_path, monkeypatch):
    value = _arrays(1, 20)
    _store(_backend(tmp_path), "k", value)
    (pack,) = _files(tmp_path)
    assert pack.suffix == ".pack"
    got, opened = _opened_chunk_files(monkeypatch, _backend(tmp_path), "k")
    assert opened == [str(pack)]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(got, value))


def test_an_entry_sharing_chunks_opens_the_packs_it_holds(tmp_path, monkeypatch):
    base = _arrays(2, 10)
    backend = _backend(tmp_path)
    _store(backend, "base", base)
    _store(backend, "more", [*base, *_arrays(3, 5)])
    assert len(_files(tmp_path)) == 2
    got, opened = _opened_chunk_files(monkeypatch, _backend(tmp_path), "more")
    assert len(opened) == 2
    assert all(a.tobytes() == b.tobytes() for a, b in zip(got, [*base, *_arrays(3, 5)]))


def test_another_process_reuses_the_chunks_stored_before_it(tmp_path):
    """It learns where they are from the entries' metadata, so it writes
    only what is new."""
    base = _arrays(4, 6)
    _store(_backend(tmp_path), "base", base)
    before = sum(p.stat().st_size for p in _files(tmp_path))
    np.save(tmp_path / "base.npy", np.stack(base))
    run_python(
        "-c",
        textwrap.dedent(
            f"""
            import numpy as np
            from cash.backends.file_backend import FileBackend
            base = list(np.load({str(tmp_path / "base.npy")!r}))
            backend = FileBackend({str(tmp_path / "cache")!r}, flush_interval=0)
            backend.set("more", [*base, np.ones({ROWS})], {{"execution_time": 1.0}})
            backend._writes.wait_all()
            """
        ),
        cwd=tmp_path,
    )
    grown = sum(p.stat().st_size for p in _files(tmp_path)) - before
    assert [p.suffix for p in _files(tmp_path)] == [".pack", ".pack"]
    assert grown == ROWS * 8 + 4
    got = _backend(tmp_path).get("more")[1]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(got, [*base, np.ones(ROWS)]))


def test_a_wrong_place_is_a_miss_never_other_data(tmp_path):
    value = _arrays(5, 3)
    backend = _backend(tmp_path)
    _store(backend, "k", value)
    path = backend._get_path("k")
    metadata = read_entry(path, with_payload=False)[0]
    places = metadata[PLACES_FIELD]
    first, second = sorted(places, key=lambda d: places[d][1])[:2]
    # The first chunk said to be where the second is: the same size, other bytes.
    places[first] = [*places[second][:3], places[first][3]]
    assert update_metadata_in_place(path, {**metadata, PLACES_FIELD: places})
    # update_metadata_in_place keeps the entry's own places: write them by hand.
    fresh = read_entry(path, with_payload=False)[0]
    assert fresh[PLACES_FIELD] != places
    from cash.backends import entry_format

    with open(path, "r+b") as fh:
        head = fh.read(entry_format.HEADER_SIZE)
        magic, meta_len, cap = entry_format.HEADER.unpack(head)
        stored = entry_format._load_metadata(fh.read(meta_len), path)
        stored[PLACES_FIELD] = places
        data = entry_format.pickle.dumps(stored)
        assert len(data) <= cap
        fh.seek(0)
        fh.write(entry_format.HEADER.pack(magic, len(data), cap) + data)
    assert _backend(tmp_path).get("k") == (None, None)


def test_evicting_one_of_two_entries_sharing_a_pack_keeps_the_others_chunks(tmp_path):
    """The cheap entry, whose pack the other still reads two chunks from, is
    evicted. The pack is half the evicted entry's own chunks: it is rewritten
    with the two the other still reads, so the eviction frees those bytes as
    a chunk file each did; the other still restores; the tier's byte count
    matches the folder; once no entry holds a chunk, no pack is left."""
    shared = _arrays(6, 2)
    first = [*shared, *_arrays(7, 2)]
    second = [*shared, np.ones(ROWS)]
    probe = FileBackend(str(tmp_path / "probe"), flush_interval=0)
    probe.set("first", first, {"execution_time": 0.01})
    probe.set("second", second, {"execution_time": 10.0})
    probe._writes.wait_all()
    both = probe.evictor.current_bytes
    # Room for one more entry of one array once first's own two are freed,
    # not before: evicting first alone must do.
    room = both + ROWS * 8
    backend = FileBackend(str(tmp_path / "cache"), flush_interval=0, max_size_bytes=room)
    backend.set("first", first, {"execution_time": 0.01})  # cheap: evicted first
    backend._writes.wait_all()
    (first_pack,) = _files(tmp_path)
    backend.set("second", second, {"execution_time": 10.0})
    backend._writes.wait_all()
    backend.set("third", _arrays(8, 1), {"execution_time": 10.0})
    backend._writes.wait_all()
    assert backend.get_metadata("first") is None, "the cap evicted something else"
    assert backend.get_metadata("second") is not None, "evicting first freed too little"
    assert not first_pack.exists(), "rewritten without first's own chunks"
    reader = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(reader.get("second")[1], second))
    on_disk = sum(p.stat().st_size for p in (tmp_path / "cache").rglob("*") if p.suffix in (".entry", ".pack"))
    assert backend.evictor.current_bytes == on_disk
    assert on_disk <= room
    backend.delete("second")
    backend.delete("third")
    assert _files(tmp_path) == [], "no entry holds a chunk any more"
