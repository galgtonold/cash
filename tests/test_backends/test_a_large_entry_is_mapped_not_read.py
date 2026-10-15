"""A large split entry is mapped, not read into a fresh copy.

Pool workers restoring the same 200 MB frame each read it into new memory
and then passed over it again for the checksum: about 0.5 s a restore,
twice per chunk on a run that maps two functions over the chunks. Its
buffers are now views of a private copy-on-write mapping of the entry file:
checked in one pass, shared with the page cache, and the caller's own to
change -- a write lands in that process's copy of the page, never in the
file or in another hit.
"""

from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")

from cash.backends import entry_format
from cash.backends.entry_format import HEADER, HEADER_SIZE, SPLIT_ALIGN, read_entry
from cash.backends.file_backend import FileBackend

N = 500_000  # 4 MB of float64: above MAP_MIN_BYTES

pytestmark = pytest.mark.skipif(
    os.name != "posix" or sys.version_info < (3, 13), reason="entries are mapped on POSIX, Python 3.13+"
)


def _stored(tmp_path, value):
    backend = FileBackend(str(tmp_path), flush_interval=0)
    backend.set("k", value, {"execution_time": 1.0})
    backend._writes.wait_all()
    backend.shutdown()
    return backend._get_path("k")


def _fresh(tmp_path):
    return FileBackend(str(tmp_path), flush_interval=0)


@pytest.fixture
def copied(monkeypatch):
    """Bytes read into new memory by the split reader."""
    seen = []
    real = entry_format._read_into
    monkeypatch.setattr(entry_format, "_read_into", lambda fh, n, path: (seen.append(n), real(fh, n, path))[1])
    return seen


def test_a_large_entry_is_restored_without_copying_its_buffers(tmp_path, copied):
    arr = np.arange(N, dtype=np.float64)
    _stored(tmp_path, arr)
    _meta, got = _fresh(tmp_path).get("k")
    np.testing.assert_array_equal(got, arr)
    assert sum(copied) < 64 * 1024, f"read {sum(copied)} bytes into new memory for a {arr.nbytes}-byte array"


def test_the_payload_of_a_split_entry_starts_aligned(tmp_path):
    path = _stored(tmp_path, np.arange(N, dtype=np.float64))
    with open(path, "rb") as fh:
        _magic, _meta_len, cap = HEADER.unpack(fh.read(HEADER_SIZE))
    assert (HEADER_SIZE + cap) % SPLIT_ALIGN == 0
    _meta, got = _fresh(tmp_path).get("k")
    assert got.ctypes.data % SPLIT_ALIGN == 0
    assert got.flags.writeable


def test_a_write_to_a_restored_value_stays_with_its_caller(tmp_path):
    arr = np.arange(N, dtype=np.float64)
    path = _stored(tmp_path, arr)
    reader = _fresh(tmp_path)
    first = reader.get("k")[1]
    second = reader.get("k")[1]
    first[:10] = -1.0
    assert second[0] == 0.0, "a write reached another hit's copy"
    assert _fresh(tmp_path).get("k")[1][0] == 0.0, "a write reached the entry"
    meta, _ = read_entry(path, with_payload=True)  # the checksum still holds
    assert meta is not None


def test_a_value_outlives_its_entry_file(tmp_path):
    arr = np.arange(N, dtype=np.float64)
    _stored(tmp_path, arr)
    reader = _fresh(tmp_path)
    got = reader.get("k")[1]
    reader.delete("k")
    reader.shutdown()
    np.testing.assert_array_equal(got, arr)


def test_a_damaged_mapped_entry_is_not_served(tmp_path):
    path = _stored(tmp_path, np.arange(N, dtype=np.float64))
    data = bytearray(open(path, "rb").read())
    data[len(data) // 2] ^= 0x01
    with open(path, "wb") as fh:
        fh.write(data)
    assert _fresh(tmp_path).get("k") == (None, None)


def test_an_entry_whose_payload_is_not_aligned_is_read(tmp_path, copied):
    """As one written before the metadata region was padded: read into new
    memory, as it was."""
    # Buffers under a megabyte each, so they stay in the entry (not chunks).
    arrays = [np.arange(N // 5, dtype=np.float64) + i for i in range(5)]
    path = _stored(tmp_path, arrays)
    data = open(path, "rb").read()
    magic, meta_len, cap = HEADER.unpack(data[:HEADER_SIZE])
    new_cap = cap + 8
    assert (HEADER_SIZE + new_cap) % SPLIT_ALIGN != 0
    shifted = HEADER.pack(magic, meta_len, new_cap) + data[HEADER_SIZE : HEADER_SIZE + cap] + bytes(8)
    with open(path, "wb") as fh:
        fh.write(shifted + data[HEADER_SIZE + cap :])
    _meta, got = _fresh(tmp_path).get("k")
    for one, stored in zip(got, arrays):
        np.testing.assert_array_equal(one, stored)
    assert sum(copied) >= sum(a.nbytes for a in arrays)
