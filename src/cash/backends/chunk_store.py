"""Big buffers stored once per cache folder, by their content.

A split entry (``entry_format.MAGIC_SPLIT``) keeps each buffer of at least
`CHUNK_MIN_BYTES` -- a frame's column, an array's data -- not in its own
file but in ``chunks/<digest>.chunk``, named by the buffer's tree digest
(`cash.bulk_digest.bulk_digest`). The entry's table names the chunk and
carries its checksum. A buffer that is already stored, because another entry
holds the same column (a frame with one column added, the same data stored
by two statements, each iteration of a loop returning its input's columns),
is not written again: the write costs its digest, not its bytes.

The digest carries the hashing scheme (`cash.bulk_digest.TAG`): a chunk
written by an install hashing with BLAKE3 never has the name of one hashed
with SHA-256, so the two never stand in for each other.

A chunk file is the buffer's bytes followed by their crc32 (4 bytes, big
endian). A reader checks the bytes against the crc32 the ENTRY recorded, so a
damaged or missing chunk makes the entry a miss, never other data, and a
damaged one is removed so the next write stores it afresh. A writer that
finds the chunk already there takes its crc32 from that trailer instead of
reading the bytes.

Who holds a chunk
-----------------
Each entry lists its chunks in its metadata (``chunks``), so the holders of
every chunk are known from the metadata alone. This process learns them once,
by reading every entry's metadata (`ensure_index`), the first time it has to
decide whether a chunk can go: when an entry holding chunks is removed or
replaced. From then on it counts its own writes and removals. A chunk goes
when no entry holds it.

Another process may store an entry holding a chunk after this one read the
folder. It touches the chunk (a new mtime) or writes it, so a chunk whose
mtime is newer than this process's reading and not this process's own stamp
is left alone. What a race can still do is remove a chunk an entry another
process just wrote needs: that entry is then a miss, computed again, never
wrong. A chunk nobody holds -- a writer killed between writing it and its
entry, a chunk left alone as above -- goes when a later reading of the folder
finds it unheld and older than `ORPHAN_CHUNK_AGE`.
"""

from __future__ import annotations

import logging
import mmap
import os
import sys
import threading
import time
import zlib
from collections.abc import Callable, Iterable
from typing import Any

from .._paths import replace_with_retry
from ..bulk_digest import LEAF_BYTES, bulk_digest
from .cache_dir import CHUNK_DIRNAME, CHUNK_SUFFIX, create_temp_file, owned_temp_prefix, write_all

logger = logging.getLogger(__name__)

__all__ = [
    "CHUNK_MIN_BYTES",
    "ChunkError",
    "ChunkStore",
    "read_chunk",
]

#: A buffer at least this big is stored as a chunk; a smaller one stays in
#: its entry, where a separate file would cost more than the bytes.
CHUNK_MIN_BYTES = LEAF_BYTES
#: An unheld chunk this old is left over (see the module docstring).
ORPHAN_CHUNK_AGE = 3600.0
#: The crc32 after a chunk's bytes.
TRAILER = 4
#: A chunk at least this big is mapped rather than read, as an entry is.
MAP_MIN_BYTES = 1 << 20
_CAN_MAP = os.name == "posix" and sys.version_info >= (3, 13)


def chunk_name(digest: bytes) -> str:
    return digest.hex() + CHUNK_SUFFIX


class ChunkError(OSError):
    """A chunk an entry names is missing, the wrong size, or damaged."""


def read_chunk(cache_dir: str, digest: bytes, nbytes: int, crc: bytes) -> Any:
    """The *nbytes* bytes of chunk *digest*, checked against *crc*: a view
    of a private copy-on-write mapping of the file, or a new ``bytearray``.

    Raises `ChunkError` when the chunk is missing, has the wrong size or does
    not match *crc*; a damaged chunk is removed on the way, so the next write
    of those bytes stores them again.
    """
    path = os.path.join(cache_dir, CHUNK_DIRNAME, chunk_name(digest))
    try:
        fh = open(path, "rb")
    except OSError as exc:
        raise ChunkError(f"chunk {path}: {exc}") from exc
    with fh:
        size = os.fstat(fh.fileno()).st_size
        if size != nbytes + TRAILER:
            raise ChunkError(f"chunk {path}: {size} bytes, the entry says {nbytes + TRAILER}")
        data: Any = None
        if _CAN_MAP and nbytes >= MAP_MIN_BYTES:
            try:
                mapped = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY, trackfd=False)
            except (OSError, ValueError, TypeError):
                mapped = None
            if mapped is not None:
                data = memoryview(mapped)[:nbytes]
        if data is None:
            data = bytearray(nbytes)
            view = memoryview(data)
            got = 0
            while got < nbytes:
                k = fh.readinto(view[got:])
                if not k:
                    raise ChunkError(f"chunk {path}: truncated")
                got += k
    if zlib.crc32(data).to_bytes(4, "big") != crc:
        _discard(path)
        raise ChunkError(f"chunk {path}: checksum does not match its entry -- the value will be recomputed")
    return data


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        logger.debug("Could not remove damaged chunk %s", path, exc_info=True)


class ChunkStore:
    """The chunks of one cache folder, and which entries hold each.

    *entry_chunks* lists ``(stem, [(chunk name, file bytes), ...])`` for
    every entry in the folder, from their metadata; called once, by
    `ensure_index`.
    """

    def __init__(
        self,
        cache_dir: str,
        entry_chunks: Callable[[], Iterable[tuple[str, Iterable[tuple[str, int]]]]],
        untracked: Callable[[], Any],
    ) -> None:
        self.cache_dir = cache_dir
        self.dir = os.path.join(cache_dir, CHUNK_DIRNAME)
        self._entry_chunks = entry_chunks
        self._untracked = untracked
        self.lock = threading.Lock()
        #: entry stem -> the chunks it holds, for the entries this process knows.
        self._held_by: dict[str, frozenset[str]] = {}
        #: chunk -> the entries holding it.
        self._holders: dict[str, set[str]] = {}
        #: chunk -> writes in flight that will hold it (put, not yet committed).
        self._pending: dict[str, int] = {}
        #: chunk -> its mtime (ns) as this process last wrote or touched it.
        self._own_mtime: dict[str, int] = {}
        #: chunk -> its file's bytes, for the chunks some known entry holds.
        self._size: dict[str, int] = {}
        self.indexed = False
        #: When the folder was read (ns); 0 before.
        self._indexed_at = 0

    def after_fork(self) -> None:
        self.lock = threading.Lock()

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    # -- writing ---------------------------------------------------------------

    def put(self, buffer: Any) -> tuple[bytes, bytes, int]:
        """Store *buffer* as a chunk unless it is stored already.

        Returns ``(digest, crc32, bytes added to the folder)``. The chunk
        counts as held by a write in flight until `commit` or `abandon`.
        """
        view = memoryview(buffer)
        if view.ndim != 1 or view.format != "B":
            view = view.cast("B")
        nbytes = view.nbytes
        digest = bulk_digest(view)
        name = chunk_name(digest)
        with self.lock:
            self._pending[name] = self._pending.get(name, 0) + 1
        path = self._path(name)
        try:
            crc = self._reuse(path, nbytes)
            if crc is not None:
                self._note_own(name, path)
                return digest, crc, 0
            crc = zlib.crc32(view).to_bytes(4, "big")
            added = self._write(path, view, crc)
            self._note_own(name, path)
        except BaseException:
            self._unpend([name])
            raise
        return digest, crc, added

    def _reuse(self, path: str, nbytes: int) -> bytes | None:
        """The crc32 of the chunk at *path*, touched so a remover in another
        process leaves it, if it is there and the right size; else None."""
        with self._untracked():
            try:
                with open(path, "rb") as fh:
                    if os.fstat(fh.fileno()).st_size != nbytes + TRAILER:
                        return None
                    fh.seek(nbytes)
                    crc = fh.read(TRAILER)
                os.utime(path)
            except OSError:
                return None
        return crc if len(crc) == TRAILER else None

    def _write(self, path: str, view: memoryview, crc: bytes) -> int:
        """Write a chunk: a temp file renamed into place, so a reader finds
        all of it or nothing. Returns the bytes it adds to the folder."""
        with self._untracked():
            try:
                before = os.path.getsize(path)
            except OSError:
                before = 0
            try:
                fd, tmp = create_temp_file(self.dir, prefix=owned_temp_prefix())
            except FileNotFoundError:
                os.makedirs(self.dir, exist_ok=True)
                fd, tmp = create_temp_file(self.dir, prefix=owned_temp_prefix())
            try:
                try:
                    write_all(fd, view)
                    write_all(fd, crc)
                finally:
                    os.close(fd)
                replace_with_retry(tmp, path)
            except BaseException:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        return view.nbytes + TRAILER - before

    def _note_own(self, name: str, path: str) -> None:
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            return
        with self.lock:
            self._own_mtime[name] = mtime

    def _unpend(self, names: Iterable[str]) -> None:
        with self.lock:
            self._unpend_locked(names)

    def _unpend_locked(self, names: Iterable[str]) -> None:
        for name in names:
            left = self._pending.get(name, 0) - 1
            if left > 0:
                self._pending[name] = left
            else:
                self._pending.pop(name, None)

    def commit(self, stem: str, held: list[tuple[str, int]], replaced: list[tuple[str, int]] | None) -> int:
        """The entry *stem* is in place, holding *held* (``(name, file
        bytes)``, one item per `put` this write made). *replaced*: the chunks
        of the entry it replaced, as that entry's metadata listed them (None:
        there was none). Returns the bytes freed by chunks nothing holds any
        more."""
        names = frozenset(name for name, _size in held)
        with self.lock:
            for name, size in held:
                self._size[name] = size
            self._unpend_locked(name for name, _size in held)
            before = self._held_by.get(stem)
            if before is None:
                before = frozenset(name for name, _size in replaced or ())
            self._set_holder(stem, names, before)
            gone = before - names
        return self._drop(gone)

    def abandon(self, held: list[tuple[str, int]]) -> int:
        """A write that `put` *held* failed before its entry landed."""
        names = [name for name, _size in held]
        self._unpend(names)
        return self._drop(set(names))

    def release(self, stem: str, held: list[tuple[str, int]] | None) -> int:
        """The entry *stem* was removed. *held*: its chunks as its metadata
        listed them, for when this process does not know the entry. Returns
        the bytes freed."""
        with self.lock:
            before = self._held_by.get(stem)
            if before is None:
                before = frozenset(name for name, _size in held or ())
            self._set_holder(stem, frozenset(), before)
        return self._drop(before)

    def _set_holder(self, stem: str, names: frozenset[str], before: frozenset[str]) -> None:
        """Under the lock."""
        for name in before - names:
            holders = self._holders.get(name)
            if holders is not None:
                holders.discard(stem)
                if not holders:
                    del self._holders[name]
        for name in names:
            self._holders.setdefault(name, set()).add(stem)
        if names:
            self._held_by[stem] = names
        else:
            self._held_by.pop(stem, None)

    # -- removing ----------------------------------------------------------------

    def ensure_index(self) -> int:
        """Learn which entry holds which chunk, once per process. Removes the
        chunks found unheld and older than `ORPHAN_CHUNK_AGE`; returns the
        bytes that freed."""
        if self.indexed:
            return 0
        started = time.time_ns()
        found: dict[str, frozenset[str]] = {}
        sizes: dict[str, int] = {}
        for stem, held in self._entry_chunks():
            names = []
            for name, size in held:
                names.append(name)
                sizes[name] = size
            found[stem] = frozenset(names)
        with self.lock:
            if self.indexed:
                return 0
            for name, size in sizes.items():
                self._size.setdefault(name, size)
            for stem, names in found.items():
                if stem not in self._held_by:
                    self._set_holder(stem, names, frozenset())
            self._indexed_at = started
            self.indexed = True
        return self._sweep_orphans()

    def _sweep_orphans(self) -> int:
        cutoff = time.time() - ORPHAN_CHUNK_AGE
        freed = 0
        try:
            with self._untracked(), os.scandir(self.dir) as found:
                listed = [(e.name, e.path) for e in found if e.name.endswith(CHUNK_SUFFIX)]
        except OSError:
            return 0
        for name, path in listed:
            with self.lock:
                if name in self._holders or name in self._pending:
                    continue
            try:
                st = os.stat(path)
                if st.st_mtime < cutoff:
                    os.remove(path)
                    freed += st.st_size
            except OSError:
                continue
        return freed

    def _drop(self, names: Iterable[str]) -> int:
        """Remove those of *names* no entry holds. Returns the bytes freed."""
        names = [n for n in names]
        if not names:
            return 0
        freed = self.ensure_index()
        for name in names:
            path = self._path(name)
            with self.lock:
                if name in self._holders or name in self._pending:
                    continue
                own = self._own_mtime.get(name)
                indexed_at = self._indexed_at
            try:
                with self._untracked():
                    st = os.stat(path)
                    # Written or touched since this process read the folder,
                    # by someone else: another process's entry may hold it.
                    if st.st_mtime_ns > indexed_at and st.st_mtime_ns != own:
                        continue
                    os.remove(path)
                freed += st.st_size
            except OSError:
                continue
            with self.lock:
                self._own_mtime.pop(name, None)
                if name not in self._holders:
                    self._size.pop(name, None)
        return freed

    def knows(self, stem: str) -> bool:
        """Does this process know which chunks the entry *stem* holds?"""
        with self.lock:
            return stem in self._held_by

    def share(self, stem: str) -> float:
        """The bytes of *stem*'s chunks, each split evenly among the entries
        holding it: what keeping the entry costs the folder, for ranking."""
        with self.lock:
            names = self._held_by.get(stem)
            if not names:
                return 0.0
            total = 0.0
            for name in names:
                holders = len(self._holders.get(name) or ()) or 1
                total += self._size.get(name, 0) / holders
            return total

    def clear(self) -> None:
        """Remove every chunk (`FileBackend.clear`)."""
        with self.lock:
            self._held_by.clear()
            self._holders.clear()
            self._own_mtime.clear()
            self._size.clear()
            self.indexed = True
            self._indexed_at = time.time_ns()
        try:
            with self._untracked(), os.scandir(self.dir) as found:
                paths = [e.path for e in found if e.name.endswith(CHUNK_SUFFIX)]
        except OSError:
            return
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                logger.debug("Could not remove chunk %s during clear", path, exc_info=True)
