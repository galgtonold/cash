"""A buffer a chunk file from before packs holds is not written again.

Entries written before packs name ``chunks/<digest>.chunk`` files. A new
write of the same buffer names that file too, as those entries did, rather
than storing the bytes a second time in its pack.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cash.backends import entry_format
from cash.backends.entry_format import CHUNKS_FIELD, PLACES_FIELD
from cash.backends.file_backend import FileBackend

ROWS = 200_000  # 1.6 MB per float64 column: each one a chunk


def _backend(tmp_path):
    return FileBackend(str(tmp_path / "cache"), flush_interval=0)


def _as_before_packs(tmp_path, path):
    """Rewrite the entry at *path* as one from before packs: each chunk in a
    ``<digest>.chunk`` file of its own, no places."""
    chunks = tmp_path / "cache" / "chunks"
    with open(path, "r+b") as fh:
        magic, meta_len, cap = entry_format.HEADER.unpack(fh.read(entry_format.HEADER_SIZE))
        stored = entry_format._load_metadata(fh.read(meta_len), path)
        held = []
        for digest, (pack, offset, nbytes, _crc) in stored.pop(PLACES_FIELD).items():
            with open(chunks / pack, "rb") as src:
                src.seek(offset)
                (chunks / f"{digest}.chunk").write_bytes(src.read(nbytes + 4))
            held.append((f"{digest}.chunk", nbytes + 4))
        for pack in {name for name, _size in stored[CHUNKS_FIELD]}:
            (chunks / pack).unlink()
        stored[CHUNKS_FIELD] = held
        data = entry_format.pickle.dumps(stored)
        fh.seek(0)
        fh.write(entry_format.HEADER.pack(magic, len(data), cap) + data)


def test_a_new_write_names_the_old_chunk_file(tmp_path):
    rng = np.random.default_rng(1)
    old = [rng.random(ROWS) for _ in range(2)]
    backend = _backend(tmp_path)
    backend.set("old", old, {"execution_time": 1.0})
    backend._writes.wait_all()
    _as_before_packs(tmp_path, backend._get_path("old"))
    assert _backend(tmp_path).get("old")[1][0].tobytes() == old[0].tobytes()

    writer = _backend(tmp_path)
    new = [old[0], np.ones(ROWS)]
    writer.set("new", new, {"execution_time": 1.0})
    writer._writes.wait_all()
    (pack,) = (tmp_path / "cache" / "chunks").glob("*.pack")
    assert pack.stat().st_size == ROWS * 8 + 4, "only the buffer no chunk file holds is in the pack"
    names = {
        name for name, _size in entry_format.read_entry(writer._get_path("new"), with_payload=False)[0][CHUNKS_FIELD]
    }
    assert sum(name.endswith(".chunk") for name in names) == 1
    writer.delete("old")
    got = _backend(tmp_path).get("new")[1]
    assert [a.tobytes() for a in got] == [a.tobytes() for a in new]
