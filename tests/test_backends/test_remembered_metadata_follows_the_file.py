"""Metadata a process remembered is never paired with another process's payload.

A `FileBackend` keeps the metadata of every entry it read, so that access
stamps can be counted in memory and flushed later. Another process may replace
the whole entry in the meantime. Serving the new payload under the remembered
metadata -- whose file hashes describe the OLD inputs -- is a wrong value: once
the input goes back to what it was, the new payload validates against it. And
flushing the remembered metadata over the new entry wrote that pairing to disk
for every other process.

Two backends over one directory stand in for two processes: each has its own
remembered metadata.
"""

from __future__ import annotations

from cash.backends.entry_format import read_entry
from cash.backends.file_backend import FileBackend


def _pair(tmp_path):
    writer = FileBackend(str(tmp_path), flush_interval=0)
    reader = FileBackend(str(tmp_path), flush_interval=0)
    return writer, reader


def _store(backend, key, value, deps):
    backend.set(key, value, {"file_deps": deps})
    backend._writes.wait_all()


def test_a_read_after_another_process_rewrote_the_entry_gets_the_new_metadata(tmp_path):
    writer, reader = _pair(tmp_path)
    _store(writer, "k", "A", "hash-of-A")
    meta, value = reader.get("k")
    assert (value, meta["file_deps"]) == ("A", "hash-of-A")

    _store(writer, "k", "B", "hash-of-B")

    meta, value = reader.get("k")
    assert (value, meta["file_deps"]) == ("B", "hash-of-B")


def test_a_metadata_read_sees_the_rewrite_too(tmp_path):
    writer, reader = _pair(tmp_path)
    _store(writer, "k", "A", "hash-of-A")
    assert reader.get_metadata("k")["file_deps"] == "hash-of-A"
    reader.get("k")

    _store(writer, "k", "B", "hash-of-B")

    assert reader.get_metadata("k")["file_deps"] == "hash-of-B"


def test_flushing_access_stamps_does_not_overwrite_a_newer_entry(tmp_path):
    writer, reader = _pair(tmp_path)
    _store(writer, "k", "A", "hash-of-A")
    reader.get("k")  # an unflushed read of the entry holding "A"

    _store(writer, "k", "B", "hash-of-B")
    reader._flush_metadata()

    on_disk, _ = read_entry(writer._get_path("k"), with_payload=False)
    assert on_disk["file_deps"] == "hash-of-B"
    fresh = FileBackend(str(tmp_path), flush_interval=0)
    meta, value = fresh.get("k")
    assert (value, meta["file_deps"]) == ("B", "hash-of-B")


def test_access_stamps_of_an_unchanged_entry_are_still_flushed(tmp_path):
    writer, reader = _pair(tmp_path)
    _store(writer, "k", "A", "hash-of-A")
    reader.get("k")
    reader.get("k")
    reader._flush_metadata()

    on_disk, _ = read_entry(writer._get_path("k"), with_payload=False)
    assert on_disk["access_count"] == 2
    # And the reader keeps counting on the dict it holds.
    assert reader.get_metadata("k") is reader._touched.metadata("k")
