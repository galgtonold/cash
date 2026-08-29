"""One file per entry, without paying for the payload to read the metadata.

Entries used to be a ``.meta`` and a ``.data``, split for one good reason:
reading an entry's metadata must not cost deserializing its value. Merging
them is only safe while that stays true, so the arms below are mostly about
proving it -- by COUNTING bytes, never by timing, because a timing assertion
here would be a threshold test on a box whose first touch of a fresh file
costs 100x its warm cost.

The second property the merge depends on is that recording an access does not
rewrite the value. Every ``get`` bumps ``last_access`` and the flusher writes
it back every few seconds; if that rewrote the whole entry, reading a 100MB
frame would rewrite 100MB on a timer.
"""
from __future__ import annotations

import os
import pickle

import pytest

from cash.backends import FileBackend, entry_format
from cash.backends.entry_format import (
    CorruptEntry,
    pack_entry,
    packed_size,
    read_entry,
    update_metadata_in_place,
)


class _CountingHandle:
    """Wraps a file object and records how many bytes cross it."""

    def __init__(self, fh, reads, writes):
        self._fh, self._reads, self._writes = fh, reads, writes

    def read(self, n=-1):
        data = self._fh.read(n)
        self._reads.append(len(data))
        return data

    def write(self, data):
        self._writes.append(len(data))
        return self._fh.write(data)

    def seek(self, *a, **k):
        return self._fh.seek(*a, **k)

    def truncate(self, *a, **k):
        return self._fh.truncate(*a, **k)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._fh.__exit__(*exc)


@pytest.fixture
def counted(monkeypatch):
    """Byte counters for everything ``entry_format`` reads or writes.

    Module globals are consulted before builtins, so binding ``open`` on the
    module intercepts its file access and nobody else's.
    """
    reads: list[int] = []
    writes: list[int] = []
    real_open = open

    def counting_open(path, mode="r", *a, **k):
        return _CountingHandle(real_open(path, mode, *a, **k), reads, writes)

    monkeypatch.setattr(entry_format, "open", counting_open, raising=False)
    return reads, writes


BIG = 4 * 1024 * 1024


def _write_big(tmp_path, payload=b"z" * BIG):
    path = str(tmp_path / "big.entry")
    meta = {"key": "k", "size": len(payload), "created_at": 0.0,
            "last_access": 1.0, "access_count": 1}
    with open(path, "wb") as fh:
        fh.write(pack_entry(meta, payload))
    return path, meta, payload


# ---------------------------------------------------------------------------
# The property the two-file split existed to guarantee
# ---------------------------------------------------------------------------

def test_reading_metadata_does_not_read_the_payload(tmp_path, counted):
    """A 4MB entry must cost a few hundred bytes to inspect."""
    reads, _writes = counted
    path, meta, _payload = _write_big(tmp_path)

    got, payload = read_entry(path, with_payload=False)

    assert payload is None
    assert got["key"] == meta["key"]
    assert sum(reads) < 4096, (
        f"read {sum(reads)} bytes to fetch metadata from a {BIG:,}-byte entry; "
        f"the length-prefixed header exists to keep this O(metadata)"
    )


def test_reading_the_value_does_read_the_payload(tmp_path, counted):
    """The control. A read_entry that returned nothing would pass the arm above."""
    reads, _writes = counted
    path, _meta, payload = _write_big(tmp_path)

    _got, got_payload = read_entry(path, with_payload=True)

    assert got_payload == payload
    assert sum(reads) >= BIG


def test_metadata_read_cost_is_flat_in_payload_size(tmp_path, counted):
    """Same bytes read for a 4MB entry as for a 100-byte one.

    Identical metadata on both, deliberately: the claim is that the read cost
    tracks the METADATA and ignores the payload, so letting the two entries
    differ in metadata as well would compare the wrong thing.
    """
    reads, _writes = counted
    meta = {"key": "k", "size": 0, "created_at": 0.0, "last_access": 1.0}

    small = str(tmp_path / "small.entry")
    with open(small, "wb") as fh:
        fh.write(pack_entry(meta, b"x" * 100))
    read_entry(small, with_payload=False)
    small_bytes = sum(reads)

    reads.clear()
    big = str(tmp_path / "big2.entry")
    with open(big, "wb") as fh:
        fh.write(pack_entry(meta, b"x" * BIG))
    read_entry(big, with_payload=False)

    assert sum(reads) == small_bytes, (
        f"a {BIG:,}-byte entry cost {sum(reads)} bytes to inspect against "
        f"{small_bytes} for a 100-byte one"
    )


# ---------------------------------------------------------------------------
# Recording an access must not rewrite the value
# ---------------------------------------------------------------------------

def test_updating_metadata_does_not_rewrite_the_payload(tmp_path, counted):
    _reads, writes = counted
    path, meta, payload = _write_big(tmp_path)

    meta["last_access"] = 2.0
    meta["access_count"] = 2
    assert update_metadata_in_place(path, meta) is True

    assert sum(writes) < 4096, (
        f"wrote {sum(writes)} bytes to record an access on a {BIG:,}-byte "
        f"entry; only the header and metadata region should move"
    )
    got, got_payload = read_entry(path, with_payload=True)
    assert got["last_access"] == 2.0 and got["access_count"] == 2
    assert got_payload == payload, "the payload was disturbed"


def test_metadata_that_outgrows_its_slack_is_refused_not_corrupted(tmp_path):
    """The fallback. Reporting failure has to leave the entry readable."""
    path, meta, payload = _write_big(tmp_path, payload=b"z" * 1000)

    bloated = dict(meta, note="x" * 10_000)
    assert update_metadata_in_place(path, bloated) is False

    got, got_payload = read_entry(path, with_payload=True)
    assert got["key"] == "k" and "note" not in got
    assert got_payload == payload


def test_the_reserved_slack_absorbs_what_a_read_adds(tmp_path):
    """64 bytes has to cover real growth, or the fallback becomes the norm.

    What accrues after an entry is written is what ``get`` adds: a wider
    ``access_count``, a new ``last_access``, and the ``source`` key.
    """
    path = str(tmp_path / "e.entry")
    meta = {"key": "m.f:state:0:args", "size": 512, "created_at": 1.0,
            "last_access": 1.0, "access_count": 0, "compressed": False,
            "storage": ["DISK"]}
    with open(path, "wb") as fh:
        fh.write(pack_entry(meta, b"x" * 512))

    grown = dict(meta, access_count=10_000_000, last_access=2.0, source="DISK")
    assert update_metadata_in_place(path, grown) is True, (
        "a normally-aged entry no longer fits its reserved region, so every "
        "flush would fall back to skipping the update"
    )


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path):
    path = str(tmp_path / "e.entry")
    meta = {"key": "k", "nested": {"a": [1, 2, 3]}}
    with open(path, "wb") as fh:
        fh.write(pack_entry(meta, b"payload"))
    assert read_entry(path, with_payload=True) == (meta, b"payload")


def test_packed_size_matches_what_lands_on_disk(tmp_path):
    """The backend keeps its byte total from this instead of a stat."""
    path = str(tmp_path / "e.entry")
    meta = {"key": "k", "size": 7}
    with open(path, "wb") as fh:
        fh.write(pack_entry(meta, b"payload"))
    assert packed_size(meta, 7) == os.path.getsize(path)


@pytest.mark.parametrize("bad", [b"", b"short", b"XXXX" + b"\x00" * 32])
def test_unreadable_bytes_raise_rather_than_return_junk(tmp_path, bad):
    path = str(tmp_path / "bad.entry")
    with open(path, "wb") as fh:
        fh.write(bad)
    with pytest.raises((CorruptEntry, pickle.UnpicklingError, EOFError)):
        read_entry(path, with_payload=False)


# ---------------------------------------------------------------------------
# End to end, through the backend
# ---------------------------------------------------------------------------

def test_a_flush_after_a_read_leaves_a_big_value_intact(tmp_path):
    """The whole point, reached the way a session reaches it."""
    backend = FileBackend(str(tmp_path / "c"), flush_interval=0)
    payload = b"q" * (2 * 1024 * 1024)
    backend.set("k", payload)
    backend._writes.wait_all()

    assert backend.get("k")[1] == payload      # dirties last_access
    backend._flush_metadata()

    metadata, value = backend.get("k")
    assert value == payload
    assert metadata["access_count"] >= 1
