"""A damaged entry is a miss, not a wrong value.

Found while attacking the decorator before round 26: entries carried no
integrity check, so a single flipped byte inside a stored payload came back as
data. Measured: a cached ``{"payload": "yyy..."}`` returned ``"yyyyZyyy..."``
with no warning. Truncation was already handled; corruption that leaves the
pickle structurally valid was not.

A crc32 of the payload costs 4.6 ms per 64 MB -- next to writing or reading the
same bytes, nothing -- and turns "silently wrong" into "recomputed".
"""
from __future__ import annotations

import pytest

from cash.backends.entry_format import (
    CorruptEntry,
    pack_entry,
    read_entry,
    unpack_entry,
    update_metadata_in_place,
)


def _entry(tmp_path, payload=b"y" * 512):
    path = tmp_path / "e.entry"
    path.write_bytes(pack_entry({"key": "k", "execution_time": 1.0}, payload))
    return path


def test_a_flipped_payload_byte_is_refused(tmp_path):
    path = _entry(tmp_path)
    blob = bytearray(path.read_bytes())
    blob[-8] = blob[-8] ^ 0xFF
    path.write_bytes(bytes(blob))

    with pytest.raises(CorruptEntry):
        read_entry(str(path), with_payload=True)
    # Metadata-only reads still work: they do not touch the payload.
    assert read_entry(str(path), with_payload=False)[0]["key"] == "k"


def test_an_intact_entry_reads_back(tmp_path):
    path = _entry(tmp_path)
    metadata, payload = read_entry(str(path), with_payload=True)
    assert payload == b"y" * 512
    assert metadata["key"] == "k"
    assert not [k for k in metadata if k.startswith("_payload")], metadata


def test_an_in_memory_blob_is_checked_too(tmp_path):
    blob = bytearray(pack_entry({"key": "k"}, b"payload-bytes"))
    assert unpack_entry(bytes(blob), with_payload=True)[1] == b"payload-bytes"
    blob[-3] = blob[-3] ^ 0xFF
    with pytest.raises(CorruptEntry):
        unpack_entry(bytes(blob), with_payload=True)


def test_a_metadata_rewrite_keeps_the_check(tmp_path):
    """`last_access` is rewritten in place every few seconds; the payload is
    untouched, so its checksum must survive."""
    path = _entry(tmp_path)
    assert update_metadata_in_place(str(path), {"key": "k", "last_access": 123.0})
    metadata, payload = read_entry(str(path), with_payload=True)
    assert payload == b"y" * 512 and metadata["last_access"] == 123.0

    blob = bytearray(path.read_bytes())
    blob[-8] = blob[-8] ^ 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(CorruptEntry):
        read_entry(str(path), with_payload=True)


def test_an_entry_written_before_the_check_still_reads(tmp_path):
    """Older entries carry no checksum; they are read, not refused."""
    import pickle
    import struct

    from cash.backends.entry_format import HEADER, MAGIC, META_SLACK

    meta = pickle.dumps({"key": "k"})
    path = tmp_path / "old.entry"
    path.write_bytes(b"".join((
        HEADER.pack(MAGIC, len(meta), len(meta) + META_SLACK),
        meta, bytes(META_SLACK), b"payload",
    )))
    assert read_entry(str(path), with_payload=True)[1] == b"payload"
