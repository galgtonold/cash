"""A big chunk's crc32 is taken in parts on several threads.

A result kept on disk alone (a 400 MB array) is read back by every hit,
and each read checked the chunk's crc32 on one core: over 100 ms of a
120 ms hit. The parts are summed on several threads and combined, which
gives the number ``zlib.crc32`` gives, so a damaged byte anywhere in the
chunk is still a miss.
"""

from __future__ import annotations

import os
import random
import threading
import zlib

import pytest

np = pytest.importorskip("numpy")

from cash.backends import chunk_store
from cash.backends.chunk_store import CRC_PARALLEL_MIN_BYTES, crc32_combine, crc32_of
from cash.backends.file_backend import FileBackend

BIG = 2 * CRC_PARALLEL_MIN_BYTES + 12_345  # parts of unequal length


def test_combining_the_parts_gives_zlibs_number():
    rng = random.Random(7)
    for _ in range(100):
        a, b = os.urandom(rng.randrange(0, 300)), os.urandom(rng.randrange(0, 300))
        assert crc32_combine(zlib.crc32(a), zlib.crc32(b), len(b)) == zlib.crc32(a + b)
    data = np.random.default_rng(1).integers(0, 255, BIG, dtype=np.uint8)
    assert crc32_of(data) == zlib.crc32(data).to_bytes(4, "big")
    assert crc32_of(b"small") == zlib.crc32(b"small").to_bytes(4, "big")


def test_a_big_chunk_is_summed_on_several_threads(monkeypatch):
    monkeypatch.setattr(chunk_store, "hashing_threads", lambda: 4)
    names = []
    real = zlib.crc32

    def counted(data, *rest):
        names.append(threading.current_thread().name)
        return real(data, *rest)

    monkeypatch.setattr(chunk_store.zlib, "crc32", counted)
    data = np.arange(BIG // 8, dtype=np.int64)
    crc32_of(data)
    assert len(names) == 4
    assert names.count("cash-crc") == 3


def test_a_damaged_byte_in_the_last_part_of_a_big_chunk_is_a_miss(tmp_path):
    value = np.arange(BIG // 8, dtype=np.int64)
    backend = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    backend.set("k", value, {"execution_time": 1.0})
    backend._writes.wait_all()
    reader = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    got = reader.get("k")[1]
    assert np.array_equal(got, value)
    del got
    folder = tmp_path / "cache" / "chunks"
    (path,) = [p for p in folder.iterdir() if p.name.endswith((".pack", ".chunk"))]
    with open(path, "r+b") as fh:
        fh.seek(value.nbytes - 10)
        byte = fh.read(1)
        fh.seek(value.nbytes - 10)
        fh.write(bytes([byte[0] ^ 0xFF]))
    assert FileBackend(str(tmp_path / "cache"), flush_interval=0).get("k") == (None, None)
