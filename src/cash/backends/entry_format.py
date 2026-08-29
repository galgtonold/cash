"""The on-disk layout of a single file-backend cache entry.

One file per entry, holding both the metadata and the payload::

    offset 0            magic       4 bytes    b"CSH2"
    offset 4            meta_len    uint32 LE  bytes of pickled metadata in use
    offset 8            meta_cap    uint32 LE  bytes RESERVED for metadata
    offset 12           metadata    meta_cap bytes, pickle + zero padding
    offset 12+meta_cap  payload     to end of file

Why one file
------------
It used to be two, a ``.meta`` and a ``.data``. Splitting them was the right
call at the time for one specific reason: reading an entry's metadata must not
cost deserializing its payload, and with a single naive file it would have --
you would read a 200MB frame to answer "when was this last accessed?".

A length-prefixed header removes that reason. ``read_entry(...,
with_payload=False)`` reads twelve bytes plus the metadata and stops, so the
metadata read is O(metadata) whatever the payload weighs. With that settled,
one file is strictly better than two:

* **Half the write cost.** A write is four filesystem metadata operations per
  file -- create a temp file (121us), write it (133us), rename it (156us),
  stat it (21us) -- and about two thirds of that is namespace churn rather
  than data. Doing it once instead of twice is the single largest saving
  available on the write path.
* **Half the files.** 100k entries meant 200k directory entries, and each one
  occupies a filesystem cluster whatever its byte count.
* **Atomicity for free.** With two files a reader could observe the data
  without the metadata, which ``get`` had to detect and report as a miss. One
  file renamed into place is either wholly there or wholly absent.

Why the metadata region is padded
---------------------------------
``last_access`` and ``access_count`` change on every read, and the flusher
writes them back every few seconds. Rewriting a whole entry to update them
would mean rewriting the payload too -- so a session that reads a 100MB frame
would rewrite 100MB every flush interval. Reserving a little slack after the
metadata lets :func:`update_metadata_in_place` seek to offset 0 and rewrite
just the header and metadata, leaving the payload untouched.

The slack has to cover only what metadata gains after it is first written:
``access_count`` widening by a byte or two, and the ``source`` key that
``get`` adds. 64 bytes is several times that, and
:func:`update_metadata_in_place` reports failure rather than corrupting
anything if metadata ever outgrows it.
"""
from __future__ import annotations

import pickle
import struct
from typing import Any

__all__ = [
    "MAGIC",
    "HEADER",
    "HEADER_SIZE",
    "META_SLACK",
    "ENTRY_SUFFIX",
    "CorruptEntry",
    "pack_entry",
    "packed_size",
    "read_entry",
    "unpack_entry",
    "metadata_span",
    "update_metadata_in_place",
]

MAGIC = b"CSH2"
HEADER = struct.Struct("<4sII")
HEADER_SIZE = HEADER.size          # 12
META_SLACK = 64
ENTRY_SUFFIX = ".entry"


class CorruptEntry(ValueError):
    """The bytes at this path are not a readable cache entry.

    Raised rather than returned so it lands in the same handler as a failed
    ``pickle.load``: every caller already treats an unreadable entry as
    absent, because a recompute is always available and is never wrong.
    """


def pack_entry(metadata: dict[str, Any], payload: bytes) -> bytes:
    """Serialize one entry. *payload* is stored verbatim -- compress before."""
    meta_bytes = pickle.dumps(metadata)
    cap = len(meta_bytes) + META_SLACK
    return b"".join((
        HEADER.pack(MAGIC, len(meta_bytes), cap),
        meta_bytes,
        bytes(cap - len(meta_bytes)),
        payload,
    ))


def packed_size(metadata: dict[str, Any], payload_len: int) -> int:
    """On-disk size of the entry ``pack_entry`` would produce.

    Lets the backend keep its byte total without stat-ing what it just wrote.
    Costs one extra ``pickle.dumps`` of the metadata, which measured 0.8us
    against the ~21us the ``stat`` it replaces takes.
    """
    return HEADER_SIZE + len(pickle.dumps(metadata)) + META_SLACK + payload_len


def unpack_entry(blob: bytes, *, with_payload: bool) -> tuple[dict[str, Any], bytes | None]:
    """Parse an entry that is already in memory.

    What a remote backend has: bytes handed back by a GET, not a file it can
    seek in. With ``with_payload=False`` the *blob* may be a PREFIX of the
    entry -- enough to cover the header and metadata region and no more --
    which is how a ranged GET reads metadata without downloading the value.

    Raises :class:`CorruptEntry` if the prefix is too short to hold the
    metadata it declares, so a caller that guessed a prefetch size too small
    can widen it and retry rather than silently returning nothing.
    """
    if len(blob) < HEADER_SIZE:
        raise CorruptEntry(f"truncated header ({len(blob)} bytes)")
    magic, meta_len, meta_cap = HEADER.unpack(blob[:HEADER_SIZE])
    if magic != MAGIC:
        raise CorruptEntry(f"bad magic {magic!r}")
    if meta_len > meta_cap:
        raise CorruptEntry(f"meta_len {meta_len} > cap {meta_cap}")
    end = HEADER_SIZE + meta_len
    if len(blob) < end:
        raise CorruptEntry(
            f"have {len(blob)} bytes, metadata needs {end}")
    metadata = pickle.loads(blob[HEADER_SIZE:end])
    if not with_payload:
        return metadata, None
    return metadata, blob[HEADER_SIZE + meta_cap:]


def metadata_span(blob: bytes) -> int:
    """Bytes needed from the front of an entry to read its metadata.

    Lets a caller check whether a fixed-size prefetch covered the metadata,
    and ask for exactly the right amount if it did not.
    """
    if len(blob) < HEADER_SIZE:
        raise CorruptEntry(f"truncated header ({len(blob)} bytes)")
    _magic, _meta_len, meta_cap = HEADER.unpack(blob[:HEADER_SIZE])
    return HEADER_SIZE + meta_cap


def read_entry(path: str, *, with_payload: bool) -> tuple[dict[str, Any], bytes | None]:
    """Read one entry. With ``with_payload=False`` the payload is never touched.

    That is the whole point of the header: a metadata read seeks past nothing
    and reads only what it needs, so it costs the same for a 200MB entry as
    for a 200-byte one.
    """
    with open(path, "rb") as fh:
        head = fh.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            raise CorruptEntry(f"{path}: truncated header ({len(head)} bytes)")
        magic, meta_len, meta_cap = HEADER.unpack(head)
        if magic != MAGIC:
            raise CorruptEntry(f"{path}: bad magic {magic!r}")
        if meta_len > meta_cap:
            raise CorruptEntry(f"{path}: meta_len {meta_len} > cap {meta_cap}")
        meta_bytes = fh.read(meta_len)
        if len(meta_bytes) < meta_len:
            raise CorruptEntry(f"{path}: metadata truncated")
        metadata = pickle.loads(meta_bytes)
        if not with_payload:
            return metadata, None
        fh.seek(HEADER_SIZE + meta_cap)
        return metadata, fh.read()


def update_metadata_in_place(path: str, metadata: dict[str, Any]) -> bool:
    """Rewrite only the metadata region. False if it no longer fits.

    The header and the metadata go out in ONE ``write`` so a tear cannot leave
    a header pointing past the metadata it describes. This is not atomic
    against a crash -- neither was the separate ``.meta`` file it replaces --
    and the failure mode is identical: the metadata no longer unpickles, the
    entry reads as absent, and the value is recomputed.
    """
    meta_bytes = pickle.dumps(metadata)
    with open(path, "r+b") as fh:
        head = fh.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            return False
        magic, _meta_len, cap = HEADER.unpack(head)
        if magic != MAGIC or len(meta_bytes) > cap:
            return False
        fh.seek(0)
        fh.write(HEADER.pack(MAGIC, len(meta_bytes), cap) + meta_bytes)
    return True
