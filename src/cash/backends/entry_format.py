"""The on-disk layout of a single file-backend cache entry.

One file per entry, holding both the metadata and the payload::

    offset 0            magic       4 bytes    b"CSH2" or b"CSH3"
    offset 4            meta_len    uint32 LE  bytes of pickled metadata in use
    offset 8            meta_cap    uint32 LE  bytes RESERVED for metadata
    offset 12           metadata    meta_cap bytes, pickle + zero padding
    offset 12+meta_cap  payload     to end of file

Under ``CSH2`` the payload is the serialized value in one piece. Under
``CSH3`` (:data:`MAGIC_SPLIT`) it is a pickle stream and the large buffers
it refers to (a numpy array's data, a frame's columns), each stored as it
is in memory -- see "Why large buffers are stored apart" below.

Why one file
------------
Reading an entry's metadata must not cost deserializing its payload -- a 200MB
frame to answer "when was this last accessed?". A length-prefixed header gives
that without a second file: ``read_entry(...,
with_payload=False)`` reads twelve bytes plus the metadata and stops, so the
metadata read is O(metadata) whatever the payload weighs. With that settled,
one file is strictly better than two:

* **Half the write cost.** A write is four filesystem metadata operations per
  file (create a temp file, write, rename, stat), mostly namespace churn
  rather than data; doing them once instead of twice is the largest saving on
  the write path.
* **Half the files.** Each directory entry occupies a filesystem cluster
  whatever its byte count.
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

Why the payload is checksummed
------------------------------
Truncation shows by itself -- a short file cannot satisfy its own header --
but damage that leaves the pickle structurally valid does not: a flipped byte
inside a stored value would come back as data. A recompute is always
available and never wrong, so an entry that does not match its checksum is
treated exactly like a truncated one: absent.

crc32 costs a fraction of writing or reading the payload it covers (several
times faster than sha256). It detects damage, not forgery:
nothing here defends against someone who can write the cache directory, and
nothing needs to.

Why large buffers are stored apart
----------------------------------
Pickled in one piece, a 100 MB array is copied into the pickle stream, and
loading it copies it out again into a new array: two passes over the value
besides the disk I/O, which made a disk hit about 5x an ``np.load``. Pickle
protocol 5 can hand such buffers over "out of band" instead. A ``CSH3``
payload is::

    stream_len  uint64 LE
    n_buffers   uint32 LE
    lengths     n_buffers x uint64 LE
    stream      the pickle stream, at the next multiple of 64
    buffers     each at the next multiple of 64, zero padding between

A read fills one ``bytearray`` per buffer straight from the file and the
unpickled arrays use that memory as it is, with no second copy. The
checksum covers every byte of the payload, padding included, and is checked
before anything is unpickled.
"""

from __future__ import annotations

import os
import pickle
import struct
import zlib
from collections.abc import Sequence
from typing import Any, NamedTuple

__all__ = [
    "MAGIC",
    "MAGIC_SPLIT",
    "MAGICS",
    "HEADER",
    "HEADER_SIZE",
    "META_SLACK",
    "ENTRY_SUFFIX",
    "CHECKSUM_FIELD",
    "CorruptEntry",
    "PackedEntry",
    "SplitPayload",
    "pack_entry",
    "pack_entry_parts",
    "split_chunks",
    "packed_size",
    "read_entry",
    "read_entry_and_checksum",
    "entry_identity",
    "payload_checksum",
    "unpack_entry",
    "metadata_span",
    "update_metadata_in_place",
]

MAGIC = b"CSH2"
#: The payload is a pickle stream plus its out-of-band buffers (module docstring).
MAGIC_SPLIT = b"CSH3"
MAGICS = (MAGIC, MAGIC_SPLIT)
#: Where each part of a split payload starts: a multiple of this, from the
#: payload's start.
SPLIT_ALIGN = 64
_SPLIT_HEAD = struct.Struct("<QI")
HEADER = struct.Struct("<4sII")
HEADER_SIZE = HEADER.size  # 12
META_SLACK = 64
ENTRY_SUFFIX = ".entry"

#: Metadata key holding the crc32 of the payload. Stamped by `pack_entry` and
#: removed again by every read, so it never reaches a backend or a caller. An
#: entry without it is not a readable entry.
#:
#: It is held as four bytes rather than an int so that its WIDTH does not
#: depend on its value: a metadata region whose size varied with the payload's
#: checksum would make `packed_size` a guess and the cost of a metadata read
#: depend on the payload after all.
CHECKSUM_FIELD = "_payload_crc32"


def _checksum(payload: bytes) -> bytes:
    return zlib.crc32(payload).to_bytes(4, "big")


def _checksum_of(chunks: Sequence[Any]) -> bytes:
    crc = 0
    for chunk in chunks:
        crc = zlib.crc32(chunk, crc)
    return crc.to_bytes(4, "big")


def _nbytes(chunk: Any) -> int:
    return chunk.nbytes if isinstance(chunk, memoryview) else len(chunk)


def _pad(offset: int) -> int:
    return -offset % SPLIT_ALIGN


class SplitPayload(NamedTuple):
    """A ``CSH3`` payload: the pickle stream and its out-of-band buffers."""

    stream: Any
    buffers: list


class PackedEntry(NamedTuple):
    """An entry ready to write: *head* (header and metadata region), then
    *chunks* (the payload, in order), *size* bytes in all."""

    head: bytes
    chunks: list
    checksum: bytes
    size: int


def split_chunks(stream: Any, buffers: Sequence[Any]) -> list:
    """The payload of a ``CSH3`` entry, as the pieces to write in order.

    The buffers are written from where they are, not joined into one blob:
    joining a 100 MB payload was a copy of its own."""
    table = _SPLIT_HEAD.pack(len(stream), len(buffers)) + struct.pack(
        f"<{len(buffers)}Q", *(_nbytes(b) for b in buffers)
    )
    chunks: list = [table + bytes(_pad(len(table))), stream]
    offset = _nbytes(chunks[0]) + len(stream)
    for buf in buffers:
        pad = _pad(offset)
        if pad:
            chunks.append(bytes(pad))
            offset += pad
        chunks.append(buf)
        offset += _nbytes(buf)
    return chunks


def payload_checksum(payload: bytes) -> bytes:
    """The checksum `pack_entry` stamps for *payload*."""
    return _checksum(payload)


def entry_identity(metadata: dict[str, Any], checksum: bytes | None) -> tuple:
    """Which write of an entry *metadata* came from.

    Every write stamps a payload checksum and a ``created_at``; a flush of
    access stamps changes neither. Two reads with one identity read the same
    stored result, so metadata a process remembered may stand in for the
    entry's; with another identity, some process has replaced the entry since,
    and what was remembered describes a value that is no longer there.
    """
    return (checksum, metadata.get("created_at"))


class CorruptEntry(ValueError):
    """The bytes at this path are not a readable cache entry.

    Every caller treats an unreadable entry as absent: a recompute is always
    available and is never wrong. Metadata that fails to unpickle is reported
    as this too, so a caller catches ``(OSError, CorruptEntry)`` and nothing
    broader.
    """


def _load_metadata(meta_bytes: bytes, where: str) -> dict[str, Any]:
    """Unpickle an entry's metadata region, or raise :class:`CorruptEntry`."""
    try:
        metadata = pickle.loads(meta_bytes)
    except Exception as exc:  # unpickling runs arbitrary code
        # A module missing in this environment (a numpy scalar written by
        # another one), a user object's __setstate__: any of it means the
        # entry cannot be read here.
        raise CorruptEntry(f"{where}: metadata does not unpickle ({type(exc).__name__}: {exc})") from exc
    if not isinstance(metadata, dict):
        raise CorruptEntry(f"{where}: metadata is a {type(metadata).__name__}, not a dict")
    return metadata


def pack_entry(metadata: dict[str, Any], payload: bytes, checksum: bytes | None = None) -> bytes:
    """Serialize one entry. *payload* is stored verbatim -- compress before.

    *checksum*: `payload_checksum` of *payload*, when the caller has it already.
    For a remote backend, which sends one blob; the file backend writes the
    parts of `pack_entry_parts` one after the other instead."""
    entry = pack_entry_parts(metadata, [payload], checksum=checksum)
    return b"".join((entry.head, payload))


def pack_entry_parts(
    metadata: dict[str, Any], chunks: list, *, split: bool = False, checksum: bytes | None = None
) -> PackedEntry:
    """One entry whose payload is *chunks* in order, without joining them.

    *split*: the chunks are `split_chunks` (a ``CSH3`` entry); otherwise they
    are the serialized value."""
    if checksum is None:
        checksum = _checksum_of(chunks)
    meta_bytes = pickle.dumps({**metadata, CHECKSUM_FIELD: checksum})
    cap = len(meta_bytes) + META_SLACK
    head = b"".join((HEADER.pack(MAGIC_SPLIT if split else MAGIC, len(meta_bytes), cap), meta_bytes, bytes(META_SLACK)))
    return PackedEntry(head, chunks, checksum, len(head) + sum(_nbytes(c) for c in chunks))


def packed_size(metadata: dict[str, Any], payload_len: int) -> int:
    """On-disk size of the entry ``pack_entry`` would produce.

    Lets the backend keep its byte total without stat-ing what it just wrote.
    Costs one extra ``pickle.dumps`` of the metadata, which measured 0.8us
    against the ~21us the ``stat`` it replaces takes.
    """
    stamped = {**metadata, CHECKSUM_FIELD: bytes(4)}
    return HEADER_SIZE + len(pickle.dumps(stamped)) + META_SLACK + payload_len


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
    if magic not in MAGICS or (with_payload and magic != MAGIC):
        # A split entry is written only by the file backend, never sent whole.
        raise CorruptEntry(f"bad magic {magic!r}")
    if meta_len > meta_cap:
        raise CorruptEntry(f"meta_len {meta_len} > cap {meta_cap}")
    end = HEADER_SIZE + meta_len
    if len(blob) < end:
        raise CorruptEntry(f"have {len(blob)} bytes, metadata needs {end}")
    metadata = _load_metadata(blob[HEADER_SIZE:end], "entry")
    expected = metadata.pop(CHECKSUM_FIELD, None)
    if not with_payload:
        return metadata, None
    payload = blob[HEADER_SIZE + meta_cap :]
    _verify(payload, expected, "entry")
    return metadata, payload


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
    metadata, payload, _ = read_entry_and_checksum(path, with_payload=with_payload)
    return metadata, payload


def read_entry_and_checksum(path: str, *, with_payload: bool) -> tuple[dict[str, Any], bytes | None, bytes | None]:
    """`read_entry`, plus the payload checksum the entry was stamped with, for
    `entry_identity`."""
    with open(path, "rb") as fh:
        head = fh.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            raise CorruptEntry(f"{path}: truncated header ({len(head)} bytes)")
        magic, meta_len, meta_cap = HEADER.unpack(head)
        if magic not in MAGICS:
            raise CorruptEntry(f"{path}: bad magic {magic!r}")
        if meta_len > meta_cap:
            raise CorruptEntry(f"{path}: meta_len {meta_len} > cap {meta_cap}")
        meta_bytes = fh.read(meta_len)
        if len(meta_bytes) < meta_len:
            raise CorruptEntry(f"{path}: metadata truncated")
        metadata = _load_metadata(meta_bytes, path)
        expected = metadata.pop(CHECKSUM_FIELD, None)
        if not with_payload:
            return metadata, None, expected
        if magic == MAGIC_SPLIT:
            return metadata, _read_split(fh, HEADER_SIZE + meta_cap, expected, path), expected
        fh.seek(HEADER_SIZE + meta_cap)
        payload = fh.read()
    _verify(payload, expected, path)
    return metadata, payload, expected


def _read_split(fh: Any, start: int, expected: bytes | None, path: str) -> SplitPayload:
    """Read a ``CSH3`` payload into one new ``bytearray`` per part, checked."""
    if expected is None:
        raise CorruptEntry(f"{path}: no payload checksum -- the value will be recomputed")
    end = fh.seek(0, os.SEEK_END)
    fh.seek(start)
    head = _read_exact(fh, _SPLIT_HEAD.size, path)
    stream_len, count = _SPLIT_HEAD.unpack(head)
    if start + _SPLIT_HEAD.size + 8 * count > end:
        raise CorruptEntry(f"{path}: split payload truncated")
    lengths_raw = _read_exact(fh, 8 * count, path)
    lengths = struct.unpack(f"<{count}Q", lengths_raw)
    table = _SPLIT_HEAD.size + 8 * count
    # The whole layout, checked against the file before anything is allocated
    # for it: a damaged length must not ask for terabytes.
    offset = table + _pad(table) + stream_len
    for length in lengths:
        offset += _pad(offset) + length
    if start + offset != end:
        raise CorruptEntry(f"{path}: split payload is {end - start} bytes, its table says {offset}")
    crc = zlib.crc32(lengths_raw, zlib.crc32(head))
    crc = zlib.crc32(_read_exact(fh, _pad(table), path), crc)
    stream = _read_into(fh, stream_len, path)
    crc = zlib.crc32(stream, crc)
    offset = table + _pad(table) + stream_len
    buffers = []
    for length in lengths:
        pad = _pad(offset)
        if pad:
            crc = zlib.crc32(_read_exact(fh, pad, path), crc)
        buf = _read_into(fh, length, path)
        crc = zlib.crc32(buf, crc)
        buffers.append(buf)
        offset += pad + length
    found = crc.to_bytes(4, "big")
    if found != expected:
        raise CorruptEntry(
            f"{path}: payload checksum {found.hex()} != stored {expected.hex()} "
            f"({offset} bytes) -- the value will be recomputed"
        )
    return SplitPayload(stream, buffers)


def _read_exact(fh: Any, n: int, path: str) -> bytes:
    data = fh.read(n) if n else b""
    if len(data) != n:
        raise CorruptEntry(f"{path}: truncated")
    return data


def _read_into(fh: Any, n: int, path: str) -> bytearray:
    """*n* bytes from *fh*, read straight into a new ``bytearray``: no
    intermediate ``bytes`` to copy them out of."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = fh.readinto(view[got:])
        if not k:
            raise CorruptEntry(f"{path}: truncated")
        got += k
    return buf


def _verify(payload: bytes, expected: bytes | None, where: str) -> None:
    """Raise :class:`CorruptEntry` if *payload* is not the bytes that were stored."""
    if expected is None:
        raise CorruptEntry(f"{where}: no payload checksum -- the value will be recomputed")
    found = _checksum(payload)
    if found != expected:
        raise CorruptEntry(
            f"{where}: payload checksum {found.hex()} != stored {expected.hex()} "
            f"({len(payload)} bytes) -- the value will be recomputed"
        )


def update_metadata_in_place(path: str, metadata: dict[str, Any], expected: tuple | None = None) -> bool:
    """Rewrite only the metadata region. False if it no longer fits, or if
    the entry at *path* is not the one *metadata* was read from.

    *expected* is the `entry_identity` of the entry *metadata* came from.
    Another process may have replaced the entry since; its payload must not
    end up under this process's metadata (old file dependencies, an old
    ``created_at``), where it would validate against the wrong inputs.

    The header and the metadata go out in ONE ``write`` so a tear cannot leave
    a header pointing past the metadata it describes. This is not atomic
    against a crash -- neither was the separate ``.meta`` file it replaces --
    and the failure mode is identical: the metadata no longer unpickles, the
    entry reads as absent, and the value is recomputed. Nor is the check
    against *expected* atomic with the write, and need not be: a replacement
    is renamed into place as a new file, so a write through this handle after
    one lands goes to the file it replaced.
    """
    with open(path, "r+b") as fh:
        head = fh.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            return False
        magic, meta_len, cap = HEADER.unpack(head)
        if magic not in MAGICS:
            return False
        # The payload is not being rewritten, so its checksum has to survive --
        # and the caller never saw it, because every read strips it. Reading it
        # back off the entry costs one unpickle of ~280 bytes, measured at
        # 0.6us against the ~150us the write itself takes.
        try:
            previous = _load_metadata(fh.read(meta_len), path)
        except CorruptEntry:
            return False
        checksum = previous.pop(CHECKSUM_FIELD, None)
        if checksum is None:
            return False
        if expected is not None and entry_identity(previous, checksum) != expected:
            return False
        meta_bytes = pickle.dumps({**metadata, CHECKSUM_FIELD: checksum})
        if len(meta_bytes) > cap:
            return False
        fh.seek(0)
        fh.write(HEADER.pack(magic, len(meta_bytes), cap) + meta_bytes)
    return True
