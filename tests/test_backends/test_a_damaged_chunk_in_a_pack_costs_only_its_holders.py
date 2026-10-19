"""A damaged chunk in a pack costs only the entries holding it.

A chunk that failed its crc32 used to remove its whole file: with a chunk
per file that lost one buffer, with packs it lost every chunk in the pack,
other entries' too. Now the entries holding that chunk are removed, the rest
of the pack stays, and the next write of the damaged bytes stores them
afresh instead of naming the damaged place again. Two buffers an entry
places at one offset must also be one chunk: the second is checked, never
handed the first one's bytes.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cash.backends import entry_format
from cash.backends.entry_format import PLACES_FIELD, read_entry
from cash.backends.file_backend import FileBackend

ROWS = 200_000  # 1.6 MB per float64 column: each one a chunk


def _backend(tmp_path):
    return FileBackend(str(tmp_path / "cache"), flush_interval=0)


def _arrays(seed, n):
    rng = np.random.default_rng(seed)
    return [rng.random(ROWS) for _ in range(n)]


def _store(backend, values):
    for key, value in values.items():
        backend.set(key, value, {"execution_time": 1.0})
        backend._writes.wait_all()


def _place(backend, key, array):
    """Where *key*'s entry places the chunk holding *array*."""
    from cash.bulk_digest import bulk_digest

    places = read_entry(backend._get_path(key), with_payload=False)[0][PLACES_FIELD]
    return places[bulk_digest(memoryview(array).cast("B")).hex()]


def _damage(tmp_path, place):
    path = tmp_path / "cache" / "chunks" / place[0]
    with open(path, "r+b") as fh:
        fh.seek(place[1] + 100)
        byte = fh.read(1)
        fh.seek(place[1] + 100)
        fh.write(bytes([byte[0] ^ 0xFF]))


def _same(got, value):
    return got != (None, None) and [a.tobytes() for a in got[1]] == [a.tobytes() for a in value]


def test_an_entrys_own_damaged_chunk_leaves_the_others_in_its_pack(tmp_path):
    shared = _arrays(1, 2)
    own = _arrays(2, 3)
    values = {"first": [*shared, *own], "second": [*shared, np.ones(ROWS)]}
    backend = _backend(tmp_path)
    _store(backend, values)
    _damage(tmp_path, _place(backend, "first", own[0]))
    reader = _backend(tmp_path)
    assert reader.get("first") == (None, None)
    assert reader.get_metadata("first") is None, "the entry holding the damaged chunk is removed"
    # The pack, rewritten without first's chunks, still has second's.
    assert _same(_backend(tmp_path).get("second"), values["second"])
    # Stored again, the damaged bytes are written afresh, not named again.
    _store(reader, {"first": values["first"]})
    assert _same(_backend(tmp_path).get("first"), values["first"])


def test_a_damaged_shared_chunk_removes_each_entry_holding_it(tmp_path):
    shared = _arrays(3, 2)
    values = {
        "first": [*shared, *_arrays(4, 2)],
        "second": [*shared, np.ones(ROWS)],
        "third": [shared[1], np.zeros(ROWS)],
    }
    backend = _backend(tmp_path)
    _store(backend, values)
    _damage(tmp_path, _place(backend, "second", shared[0]))
    reader = _backend(tmp_path)
    assert reader.get("second") == (None, None)
    assert reader.get_metadata("first") is None and reader.get_metadata("second") is None
    assert _same(reader.get("third"), values["third"]), "third does not hold the damaged chunk"
    _store(reader, {"second": values["second"]})
    assert _same(_backend(tmp_path).get("second"), values["second"])


def test_two_chunks_placed_at_one_offset_are_a_miss(tmp_path):
    value = _arrays(5, 3)
    backend = _backend(tmp_path)
    _store(backend, {"k": value})
    path = backend._get_path("k")
    with open(path, "r+b") as fh:
        magic, meta_len, cap = entry_format.HEADER.unpack(fh.read(entry_format.HEADER_SIZE))
        stored = entry_format._load_metadata(fh.read(meta_len), path)
        places = stored[PLACES_FIELD]
        first, second = sorted(places, key=lambda d: places[d][1])[:2]
        # The second placed where the first is, with its own crc32.
        places[second] = [*places[first][:3], places[second][3]]
        data = entry_format.pickle.dumps(stored)
        assert len(data) <= cap
        fh.seek(0)
        fh.write(entry_format.HEADER.pack(magic, len(data), cap) + data)
    assert _backend(tmp_path).get("k") == (None, None)
