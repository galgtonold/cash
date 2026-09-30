"""A large array goes to disk and back without being copied again and again.

Pickled in one piece, a 100 MB array was copied into the pickle stream, into
a joined header-plus-payload blob, into a slice of that blob for the write,
and on a hit into a ``bytes`` read and out of it again into the array: a store
took 5x ``np.save`` and a disk hit 5x ``np.load``. Large buffers now go out of
band (pickle protocol 5): written from their own copy as they are, read
straight into the memory the array then uses.
"""

from __future__ import annotations

import os
import tracemalloc

import pytest

np = pytest.importorskip("numpy")

from cash.backends import file_backend
from cash.backends.cache_dir import VERSION_FILENAME
from cash.backends.entry_format import MAGIC, read_entry
from cash.backends.file_backend import FileBackend

N = 500_000  # 4 MB of float64: well above the out-of-band threshold


def _stored(tmp_path, value, **kwargs):
    backend = FileBackend(str(tmp_path), flush_interval=0, **kwargs)
    backend.set("k", value, {"execution_time": 1.0})
    backend._writes.wait_all()
    return backend


def _magic(backend, key="k"):
    with open(backend._get_path(key), "rb") as fh:
        return fh.read(4)


def _meta_cap(path):
    with open(path, "rb") as fh:
        return int.from_bytes(fh.read(12)[8:12], "little")


def _fresh(tmp_path, **kwargs):
    """Another process's view: nothing remembered from the write."""
    return FileBackend(str(tmp_path), flush_interval=0, **kwargs)


def test_a_large_array_is_written_without_joining_it_to_the_header(tmp_path, monkeypatch):
    writes = []
    real = file_backend.write_all
    monkeypatch.setattr(file_backend, "write_all", lambda fd, data: (writes.append(len(data)), real(fd, data)))
    arr = np.arange(N, dtype=np.float64)
    _stored(tmp_path, arr).shutdown()
    assert max(writes) == arr.nbytes, f"the largest write was {max(writes)} bytes for a {arr.nbytes}-byte array"


def test_a_disk_hit_holds_the_array_once(tmp_path):
    """Read into the memory the array then uses: the peak is one array, not
    the ``bytes`` read plus the array unpickled out of them."""
    arr = np.arange(N, dtype=np.float64)
    _stored(tmp_path, arr).shutdown()
    backend = _fresh(tmp_path)
    backend.get_metadata("k")  # the directory and the stamp, outside the measurement
    tracemalloc.start()
    try:
        _meta, got = backend.get("k")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    backend.shutdown()
    np.testing.assert_array_equal(got, arr)
    assert peak < 1.5 * arr.nbytes, f"a hit of a {arr.nbytes}-byte array peaked at {peak} bytes"
    got[0] = -1.0  # a hit is the caller's to change
    assert _fresh(tmp_path).get("k")[1][0] == 0.0


def test_arrays_frames_and_fortran_order_come_back_as_stored(tmp_path):
    pd = pytest.importorskip("pandas")
    value = {
        "a": np.arange(N, dtype=np.int64),
        "f": np.asfortranarray(np.arange(N, dtype=np.float32).reshape(1000, -1)),
        "df": pd.DataFrame({"x": np.arange(N // 4, dtype=float), "s": ["t"] * (N // 4)}),
        "small": np.arange(3),
    }
    _stored(tmp_path, value).shutdown()
    _meta, got = _fresh(tmp_path).get("k")
    np.testing.assert_array_equal(got["a"], value["a"])
    np.testing.assert_array_equal(got["f"], value["f"])
    assert got["f"].flags.f_contiguous
    assert got["df"].equals(value["df"])
    np.testing.assert_array_equal(got["small"], value["small"])


def test_only_values_with_a_large_buffer_are_split(tmp_path):
    from cash.backends.entry_format import MAGIC_SPLIT

    backend = _stored(tmp_path, np.arange(N, dtype=np.float64))
    backend.set("small", [1, 2, 3], {"execution_time": 1.0})
    backend._writes.wait_all()
    assert _magic(backend) == MAGIC_SPLIT
    assert _magic(backend, "small") == MAGIC
    backend.shutdown()


def test_a_compressed_tier_stores_the_value_in_one_piece(tmp_path):
    arr = np.zeros(N)
    backend = _stored(tmp_path, arr, compress=True)
    assert _magic(backend) == MAGIC
    backend.shutdown()
    np.testing.assert_array_equal(_fresh(tmp_path, compress=True).get("k")[1], arr)


@pytest.mark.parametrize("where", [0.3, 0.6, 0.99])
def test_a_damaged_split_entry_is_not_served(tmp_path, where):
    _stored(tmp_path, np.arange(N, dtype=np.float64)).shutdown()
    path = _fresh(tmp_path)._get_path("k")
    data = bytearray(open(path, "rb").read())
    data[int(len(data) * where)] ^= 0x01
    with open(path, "wb") as fh:
        fh.write(data)
    assert _fresh(tmp_path).get("k") == (None, None)


def test_a_truncated_split_entry_is_not_served(tmp_path):
    _stored(tmp_path, np.arange(N, dtype=np.float64)).shutdown()
    path = _fresh(tmp_path)._get_path("k")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 8)
    assert _fresh(tmp_path).get("k") == (None, None)


def test_a_damaged_length_asks_for_no_memory(tmp_path):
    """The buffer table is checked against the file before anything is
    allocated for it."""
    backend = _stored(tmp_path, np.arange(N, dtype=np.float64))
    backend.shutdown()
    path = backend._get_path("k")
    head = 12 + _meta_cap(path)
    with open(path, "r+b") as fh:
        fh.seek(head + 12)  # the first buffer's length
        fh.write((2**60).to_bytes(8, "little"))
    assert _fresh(tmp_path).get("k") == (None, None)


def test_recording_an_access_keeps_a_split_entry_readable(tmp_path):
    arr = np.arange(N, dtype=np.float64)
    _stored(tmp_path, arr).shutdown()
    reader = _fresh(tmp_path)
    reader.get("k")
    reader.shutdown()  # writes the access stamp back in place
    meta, _ = read_entry(reader._get_path("k"), with_payload=False)
    assert meta["access_count"] >= 1
    np.testing.assert_array_equal(_fresh(tmp_path).get("k")[1], arr)


def test_an_unstamped_directory_of_split_entries_is_kept(tmp_path):
    _stored(tmp_path, np.arange(N, dtype=np.float64)).shutdown()
    (tmp_path / VERSION_FILENAME).unlink()
    assert _fresh(tmp_path).get("k")[1] is not None
