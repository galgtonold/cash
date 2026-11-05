"""Tree digests of big buffers whose bytes are known without reading them.

A loop that stores ``h = matches.assign(p=...)`` on every pass stores the
same columns of ``matches`` every time. Each store read them again: the RAM
tier's digest of a table it keeps uncopied (`frame_sharing.share_checked`)
and the chunk store's digest of each buffer of the disk entry
(`chunk_store.digests`). Fifteen passes over a 60 MB table read 1.3 GB that
had been read before.

What is known, and how:

* A block of a pandas table by its *content key*
  (`shared_blocks.content_keys`): its layout and the bytes each of its
  columns folded into the table's digest, which the frame memo hands out
  only while it vouches that the column is unchanged since it was read.
  Two blocks with one key hold the same bytes, so the tree digest of one is
  the other's (`_BY_CONTENT`). A block of one column needs no earlier read
  at all: its piece is ``b"t"``, its length and the tree digest of its
  bytes (`bulk_digest.fold_buffer`).
* An array the tier copied for a disk write (`note_copy`), by that content
  key: the copy is cash's own and nothing writes it, so it holds the bytes
  the key names for as long as it lives.
* An Arrow buffer in memory Arrow allocated (`frame_sharing.arrow_owned`),
  by its address and size while an array holding it lives (`note_arrow`):
  nothing writes such memory (a hit shares it on the same grounds), and
  while it is held no other buffer can be at that address.

Anything else is read, as before.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import Any

__all__ = ["known", "learn", "lookup", "note_arrow", "note_copy", "remember", "single_column_digest"]

#: ``content key -> tree digest`` of the block bytes it names, least
#: recently used first.
_BY_CONTENT: OrderedDict[tuple, bytes] = OrderedDict()
#: Most keys `_BY_CONTENT` keeps.
_MAX_CONTENT = 4096
#: ``id(array) -> (weak reference, content key)`` of the arrays of copies
#: made for a disk write (`note_copy`).
_COPIES: dict[int, tuple[Any, tuple]] = {}
#: ``(address, size) -> [weak reference to an array holding that buffer,
#: tree digest or None]`` of Arrow buffers in memory Arrow allocated.
_ARROW: dict[tuple[int, int], list[Any]] = {}
_LOCK = threading.RLock()
#: Buffers smaller than this are hashed as they come: what the chunk store
#: keeps apart (`chunk_store.CHUNK_MIN_BYTES`).
MIN_BYTES = 1 << 20
#: Layouts whose block is its numpy memory, a column per row.
_NUMPY_LAYOUTS = frozenset({"ndarray", "DatetimeArray", "TimedeltaArray"})


def single_column_digest(key: tuple, nbytes: int) -> bytes | None:
    """The tree digest of the *nbytes* bytes of a block of one column, read
    from its content *key*: its piece is ``b"t"``, the length and that
    digest (`bulk_digest.fold_buffer`); None for any other key."""
    from ..bulk_digest import DIRECT_MAX_BYTES

    layout, pieces = key
    if len(pieces) != 1 or layout[0] not in _NUMPY_LAYOUTS or nbytes <= DIRECT_MAX_BYTES:
        return None
    piece = pieces[0]
    if len(piece) != 41 or piece[:1] != b"t" or int.from_bytes(piece[1:9], "little") != nbytes:
        return None
    return bytes(piece[9:])


def known(key: tuple, nbytes: int) -> bytes | None:
    """The tree digest of the *nbytes* bytes of a block of content *key*,
    when known."""
    digest = single_column_digest(key, nbytes)
    if digest is not None:
        return digest
    with _LOCK:
        digest = _BY_CONTENT.get(key)
        if digest is not None:
            _BY_CONTENT.move_to_end(key)
    return digest


def remember(key: tuple, digest: bytes) -> None:
    """Record *digest* as the tree digest of the block bytes *key* names."""
    with _LOCK:
        _BY_CONTENT[key] = digest
        _BY_CONTENT.move_to_end(key)
        while len(_BY_CONTENT) > _MAX_CONTENT:
            _BY_CONTENT.popitem(last=False)


def note_copy(copied: Any, keys: dict[tuple, tuple]) -> None:
    """Record the block arrays of *copied*, a table copied for a disk write
    from one whose blocks had *keys* (`shared_blocks.content_keys`), as
    holding the bytes their key names. Only a copy cash made and nothing
    else writes."""
    if not keys:
        return
    from .shared_blocks import _layout

    try:
        for block in copied._mgr.blocks:
            key = keys.get(tuple(int(pos) for pos in block.mgr_locs.as_array))
            values = block.values
            # The copy keeps each block's layout (`copy_keeping_blocks`);
            # checked, not trusted.
            if key is None or _layout(values) != key[0]:
                continue
            array = getattr(values, "_ndarray", values)
            if array.nbytes < MIN_BYTES:
                continue
            at = id(array)
            ref = weakref.ref(array, lambda ref, at=at: _drop(_COPIES, at, ref))
            with _LOCK:
                _COPIES[at] = (ref, key)
    except Exception:  # noqa: BLE001 - only a speedup: read as before
        return


def _drop(table: dict, at: Any, ref: Any) -> None:
    """Forget *table*'s entry at *at* when it is *ref*'s. A weak reference's
    callback: it may run inside a lock this thread holds, so it takes none
    and only pops (one step under the interpreter lock)."""
    found = table.get(at)
    if found is not None and found[0] is ref:
        table.pop(at, None)


def note_arrow(owner: Any, chunks: list[Any]) -> None:
    """Record the big buffers of the Arrow *chunks* that are memory Arrow
    allocated, as held by *owner* (an Arrow array holding every chunk) for
    as long as it lives."""
    from .frame_sharing import arrow_owned

    try:
        held: list[tuple[int, int]] = []
        ref = None
        for chunk in chunks:
            for buffer in chunk.buffers():
                if buffer is None or buffer.size < MIN_BYTES:
                    continue
                at = (buffer.address, buffer.size)
                with _LOCK:
                    found = _ARROW.get(at)
                    if found is not None and found[0]() is not None:
                        continue  # held already: its digest, if any, holds
                if not arrow_owned(buffer):
                    continue
                if ref is None:
                    ref = weakref.ref(owner, lambda ref, held=held: [_drop(_ARROW, at, ref) for at in held])
                with _LOCK:
                    found = _ARROW.get(at)
                    if found is None or found[0]() is None:
                        _ARROW[at] = [ref, None]
                        held.append(at)
    except Exception:  # noqa: BLE001 - only a speedup: read as before
        return


def lookup(view: memoryview) -> tuple[bytes | None, Any]:
    """``(digest, slot)`` for the flat byte *view* of a buffer the chunk
    store is about to hash: its tree digest when known, else None and what
    to hand `learn` with the digest once read (None: nothing to learn)."""
    if view.nbytes < MIN_BYTES:
        return None, None
    try:
        source = view.obj
        if type(source).__name__ == "Buffer" and type(source).__module__ == "pyarrow.lib":
            return _arrow_lookup(source, view.nbytes)
        return _copy_lookup(source, view.nbytes)
    except Exception:  # noqa: BLE001 - only a speedup: read as before
        return None, None


def _arrow_lookup(buffer: Any, nbytes: int) -> tuple[bytes | None, Any]:
    at = (buffer.address, buffer.size)
    if buffer.size != nbytes:
        return None, None
    with _LOCK:
        found = _ARROW.get(at)
    if found is None or found[0]() is None:
        return None, None
    return found[1], ("arrow", at, found)


def _copy_lookup(source: Any, nbytes: int) -> tuple[bytes | None, Any]:
    import numpy as np

    array = source
    for _depth in range(8):
        if not isinstance(array, np.ndarray):
            return None, None
        with _LOCK:
            found = _COPIES.get(id(array))
        if found is not None and found[0]() is array:
            if array.nbytes != nbytes or source.__array_interface__["data"][0] != array.__array_interface__["data"][0]:
                return None, None
            key = found[1]
            return known(key, nbytes), ("content", key)
        array = array.base
    return None, None


def learn(slot: Any, digest: bytes) -> None:
    """Record *digest*, read for the buffer `lookup` gave *slot* for."""
    if slot is None:
        return
    if slot[0] == "content":
        remember(slot[1], digest)
        return
    _kind, at, found = slot
    with _LOCK:
        # The same entry, its array alive since `lookup`: the memory read is
        # the memory it holds.
        if _ARROW.get(at) is found and found[0]() is not None:
            found[1] = digest
