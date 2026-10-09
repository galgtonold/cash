"""Tests for compression functionality in FileBackend."""

import gzip
import os
import pickle
import sys

import pytest

from cash import Cash
from cash.backends import FileBackend
from cash.backends.entry_format import ENTRY_SUFFIX


def test_compression_enabled(temp_cache_dir):
    """Test that compression reduces file size for compressible data."""
    # Use FileBackend directly to bypass smart persistence policy
    backend = FileBackend(temp_cache_dir, compress=True)
    app = Cash(backend=backend)

    @app.cache
    def large_data():
        return b"0" * 10000

    large_data()
    entries = app.backend.list_entries()  # list_entries drains pending writes
    assert len(entries) == 1

    files = os.listdir(temp_cache_dir)
    data_file = [f for f in files if f.endswith(ENTRY_SUFFIX)][0]
    data_path = os.path.join(temp_cache_dir, data_file)
    file_size = os.path.getsize(data_path)

    assert file_size < 1000


def test_compression_disabled(temp_cache_dir):
    """Test that disabling compression keeps files uncompressed."""
    backend = FileBackend(temp_cache_dir, compress=False)
    app = Cash(backend=backend)

    @app.cache
    def large_data():
        return b"0" * 10000

    large_data()
    # Drain pending writes so the on-disk file exists.
    backend.list_entries()
    files = os.listdir(temp_cache_dir)
    data_file = [f for f in files if f.endswith(ENTRY_SUFFIX)][0]
    data_path = os.path.join(temp_cache_dir, data_file)
    file_size = os.path.getsize(data_path)

    assert file_size > 10000


def _stored(backend, key):
    backend._writes.wait_all()
    return backend.get(key)


def test_a_compressed_entry_names_its_codec_and_reads_back(tmp_path):
    from cash.backends import compression

    backend = FileBackend(str(tmp_path), compress=True, flush_interval=0)
    value = ["the same line of text"] * 20_000
    backend.set("k", value)
    meta, got = _stored(backend, "k")
    assert got == value
    assert meta["compressed"] == compression.CODEC
    assert compression.CODEC == ("zstd" if sys.version_info >= (3, 14) else "zlib")
    assert meta["size"] < len(pickle.dumps(value, protocol=5)) / 10
    backend.shutdown()


def test_a_payload_that_does_not_shrink_is_stored_as_it_is(tmp_path):
    backend = FileBackend(str(tmp_path), compress=True, flush_interval=0)
    noise = os.urandom(3_000_000)  # past the 1 MB sample
    backend.set("k", noise)
    meta, got = _stored(backend, "k")
    assert got == noise
    assert meta["compressed"] is False
    assert meta["size"] >= len(noise)
    backend.shutdown()


def test_an_entry_gzipped_by_an_earlier_version_still_reads(tmp_path, monkeypatch):
    """Entries written before codecs were named say ``compressed: True`` and hold gzip."""
    from cash.backends import compression

    backend = FileBackend(str(tmp_path), compress=True, flush_interval=0)
    monkeypatch.setattr(compression, "compress", lambda data: (True, gzip.compress(bytes(data))))
    value = {"rows": list(range(50_000))}
    backend.set("old", value)
    backend._writes.wait_all()
    monkeypatch.undo()
    backend.shutdown()

    fresh = FileBackend(str(tmp_path), compress=True, flush_interval=0)
    meta, got = fresh.get("old")
    assert meta["compressed"] is True
    assert got == value
    fresh.shutdown()


@pytest.mark.parametrize("codec", ["zstd", "lz4"])
def test_an_entry_in_a_codec_this_python_cannot_read_is_a_miss(tmp_path, monkeypatch, codec):
    """A zstd entry read by Python 3.13 (no ``compression.zstd``) is computed again."""
    from cash.backends import compression

    backend = FileBackend(str(tmp_path), compress=True, flush_interval=0)
    monkeypatch.setattr(compression, "compress", lambda data: (codec, b"not what was pickled"))
    backend.set("k", "a" * 100_000)
    backend._writes.wait_all()
    monkeypatch.setattr(compression, "_zstd", None)
    backend.shutdown()

    fresh = FileBackend(str(tmp_path), compress=True, flush_interval=0)
    assert fresh.get_metadata("k")["compressed"] == codec
    assert fresh.get("k") == (None, None)
    fresh.shutdown()


def test_a_compressed_store_keeps_a_later_mutation_out(tmp_path):
    backend = FileBackend(str(tmp_path), compress=True, flush_interval=0)
    value = [[0] * 100 for _ in range(1_000)]
    backend.set("k", value)
    value[0][0] = 99  # after set: the entry holds what was set
    _, got = _stored(backend, "k")
    assert got[0][0] == 0
    backend.shutdown()
