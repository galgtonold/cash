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

A chunk is the buffer's bytes followed by their crc32 (4 bytes, big
endian). A reader checks the bytes against the crc32 the ENTRY recorded, so a
damaged or missing chunk makes the entry a miss, never other data, and a
damaged file is removed so the next write stores it afresh.

Packs
-----
The chunks one entry write adds go into ONE file, ``chunks/<id>.pack``, each
at a multiple of 64 bytes, rather than a file each. On Windows the virus
scanner reads every new file the first time it is opened: p08's first hit
opened 415 chunk files at about 5 ms each, twice what the later hits took.
The entry's metadata says where each of its chunks is (``chunk_places``:
digest -> file, offset, size, crc32); the table in its payload still names
each chunk by digest and crc32, and those are what the bytes are checked
against. A pack is written to a temp file and renamed into place, so a reader
finds all of it or nothing.

A chunk already stored, in a pack or a chunk file of an older entry, is not
written again: this process knows where every chunk in a pack is from the
entries' metadata (`ensure_index`, before its first write) and from its own
writes, and while an entry from before packs is left it looks for a
``<digest>.chunk`` file too, which the new entry then names as those did. A
chunk another process stored since is written again, in this write's pack:
room, never a wrong read.

A chunk of a pack that fails its check makes the entries holding it misses,
and they are removed (`ChunkStore.damaged`); the pack stays, with every
other entry's chunks in it. A pack whose holders have let go of more than a
third of it is rewritten without those bytes (`ChunkStore._compact`), by one
process at a time (``<pack>.claim``).

Who holds a chunk
-----------------
Each entry lists the files its chunks are in in its metadata (``chunks``),
so the holders of every file are known from the metadata alone; a file goes
when no entry holds a chunk in it. In the rest of this section a "chunk" is
such a file. This process learns them once,
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

import hashlib
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
from ..bulk_digest import LEAF_BYTES, PARALLEL_MIN_BYTES, bulk_digest, hashing_threads
from .cache_dir import (
    CHUNK_DIRNAME,
    CHUNK_SUFFIX,
    CLAIM_SUFFIX,
    PACK_SUFFIX,
    create_temp_file,
    owned_temp_prefix,
    write_all,
)
from .entry_format import ENTRY_SUFFIX, identity_at, moved_head

logger = logging.getLogger(__name__)

__all__ = [
    "CHUNK_MIN_BYTES",
    "ChunkError",
    "ChunkStore",
    "DamagedChunk",
    "as_bytes",
    "digests",
    "read_chunk",
    "read_chunks",
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
#: Where each chunk in a pack starts: a multiple of this.
PACK_ALIGN = 64

#: Where a chunk is: ``(file name, offset, bytes, crc32)``.
Place = tuple[str, int, int, bytes]


def chunk_name(digest: bytes) -> str:
    return digest.hex() + CHUNK_SUFFIX


def pack_name(members: list[bytes]) -> str:
    """The file a pack of the chunks *members* (their digests, in order) is
    stored in: the same members, the same name and the same bytes."""
    return hashlib.sha256(b"".join(members)).hexdigest() + PACK_SUFFIX


def _aligned(offset: int) -> int:
    return -offset % PACK_ALIGN


def as_bytes(buffer: Any) -> memoryview:
    """*buffer* as a flat view of its bytes (a buffer handed out of band is
    contiguous)."""
    view = memoryview(buffer)
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    return view


def digests(views: list[memoryview]) -> list[bytes]:
    """`bulk_digest` of each view. One under `PARALLEL_MIN_BYTES` is hashed
    on one thread by `bulk_digest`, so an entry of many such columns is
    spread over `hashing_threads` threads here: the writer hashes beside a
    busy caller, and each leaf waits for the interpreter lock it hands back,
    so one thread hashing leaf after leaf mostly waits."""
    out: list[Any] = [None] * len(views)
    small = []
    for i, view in enumerate(views):
        if view.nbytes >= PARALLEL_MIN_BYTES:
            out[i] = bulk_digest(view)  # on several threads already
        else:
            small.append(i)
    threads = min(hashing_threads(), len(small))
    if threads < 2:
        for i in small:
            out[i] = bulk_digest(views[i])
        return out
    errors: list[BaseException] = []

    def run(indices: list[int]) -> None:
        try:
            for i in indices:
                out[i] = bulk_digest(views[i])
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            errors.append(exc)

    workers = []
    for k in range(1, threads):
        worker = threading.Thread(target=run, args=(small[k::threads],), name="cash-digest", daemon=True)
        try:
            worker.start()
        except RuntimeError:  # no new threads (interpreter shutting down): do it here
            run(small[k::threads])
            continue
        workers.append(worker)
    run(small[0::threads])
    for worker in workers:
        worker.join()
    if errors:
        raise errors[0]
    return out


class _Replaced(Exception):
    """Another write replaced an entry while it was being repointed."""


def _swap_if_same(path: str, new: str, identity: tuple, directory: str) -> None:
    """Put the file *new* at *path* if the entry there is still the one
    *identity* names; raise `_Replaced` when another write replaced it.

    A rename cannot compare first, so the entry is taken away (renamed
    aside, which is atomic), checked, and the new one linked in only where
    nothing is: a write landing in between keeps its file, and the rewrite
    is dropped. An entry taken away by mistake is linked back unless a
    newer one is there. A crash leaves the entry, the new one, or no entry
    (a miss), and temp files the folder's sweep removes."""
    fd, aside = create_temp_file(directory, prefix=owned_temp_prefix())
    os.close(fd)
    try:
        try:
            # A folder without hard links (some network shares) is left as
            # it is: found out before the entry is taken away.
            os.remove(aside)
            os.link(new, aside)
            os.remove(aside)
            os.replace(path, aside)
        except OSError:
            # Gone, held open by a reader on Windows, or no hard links.
            raise _Replaced(path) from None
        if identity_at(aside) != identity:
            _link_if_absent(aside, path)
            raise _Replaced(path)
        if not _link_if_absent(new, path):
            raise _Replaced(path)
    finally:
        for left in (aside, new):
            try:
                os.remove(left)
            except OSError:
                pass


def _link_if_absent(src: str, dst: str) -> bool:
    """Hard-link *src* at *dst* unless something is there; False if it was."""
    try:
        os.link(src, dst)
    except FileExistsError:
        return False
    return True


class ChunkError(OSError):
    """A chunk an entry names is missing, the wrong size, or damaged."""


class DamagedChunk(ChunkError):
    """The chunk at *offset* in the pack at *path* does not match its crc32."""

    def __init__(self, path: str, offset: int) -> None:
        super().__init__(f"chunk {path} at {offset}: checksum does not match its entry -- the value will be recomputed")
        self.pack = os.path.basename(path)
        self.offset = offset


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

    *entry_chunks* lists ``(stem, [(file name, file bytes), ...], places)``
    for every entry in the folder, from their metadata (*places*: its
    ``chunk_places``, or None); called once, by `ensure_index`.
    """

    def __init__(
        self,
        cache_dir: str,
        entry_chunks: Callable[[], Iterable[tuple[str, Iterable[tuple[str, int]], Any]]],
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
        #: digest -> where that chunk is in a pack, for the packs this
        #: process knows (`put_all`).
        self._places: dict[bytes, Place] = {}
        #: pack -> the digests `_places` places in it.
        self._in_pack: dict[str, set[bytes]] = {}
        #: entry stem -> where each of its chunks in a pack is, for the
        #: entries this process knows: what tells a pack's live bytes.
        self._placed_by: dict[str, dict[bytes, Place]] = {}
        #: See `entry_lock`.
        self._entry_locks = [threading.RLock() for _ in range(64)]
        #: Does an entry this process knows hold a ``<digest>.chunk`` file,
        #: from before packs? Then `put_all` looks for its buffers there too.
        self._old_chunks = False
        self.indexed = False
        #: When the folder was read (ns); 0 before.
        self._indexed_at = 0

    def after_fork(self) -> None:
        self.lock = threading.Lock()
        self._entry_locks = [threading.RLock() for _ in range(64)]

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    # -- writing ---------------------------------------------------------------

    def put(self, buffer: Any, digest: bytes | None = None) -> tuple[bytes, bytes, int]:
        """Store *buffer* as a chunk unless it is stored already. *digest*:
        its `bulk_digest`, when the caller has it (`digests`).

        Returns ``(digest, crc32, bytes added to the folder)``. The chunk
        counts as held by a write in flight until `commit` or `abandon`.
        """
        view = as_bytes(buffer)
        nbytes = view.nbytes
        if digest is None:
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

    def put_all(
        self, views: list[memoryview], known: list[bytes | None]
    ) -> tuple[list[tuple[bytes, bytes]], dict[str, list], list[tuple[str, int]], int]:
        """Store the buffers of one entry: those not stored already in one new
        pack. *known*: each one's `bulk_digest`, or None.

        Returns ``(refs, places, held, added)``: each buffer's ``(digest,
        crc32)``; the entry's ``chunk_places``; the files it holds, ``(name,
        file bytes)``, each counted as held by a write in flight until
        `commit` or `abandon`; and the bytes added to the folder.
        """
        self.ensure_index()
        refs: list[Any] = [None] * len(views)
        places: dict[str, list] = {}
        files: dict[str, int] = {}
        fresh: dict[bytes, memoryview] = {}
        waiting: list[tuple[int, bytes]] = []
        old_crcs: dict[str, bytes] = {}
        try:
            for i, (view, digest) in enumerate(zip(views, known)):
                if digest is None:
                    digest = bulk_digest(view)
                with self.lock:
                    place = self._places.get(digest)
                if place is not None and place[2] == view.nbytes:
                    name = place[0]
                    if name not in files:
                        size = self._hold_stored(name)
                        if size is None:
                            self._forget(name)
                            place = None
                        else:
                            files[name] = size
                    if place is not None:
                        refs[i] = (digest, place[3])
                        places[digest.hex()] = [name, place[1], place[2], place[3].hex()]
                        continue
                if self._old_chunks:
                    name = chunk_name(digest)
                    crc = old_crcs.get(name)
                    if crc is None:
                        crc = self._hold_chunk(name, view.nbytes)
                        if crc is not None:
                            old_crcs[name] = crc
                            files[name] = view.nbytes + TRAILER
                    if crc is not None:
                        refs[i] = (digest, crc)  # no place: read from its chunk file
                        continue
                fresh.setdefault(digest, view)
                waiting.append((i, digest))
            added = 0
            if fresh:
                name, size, added, written = self._write_pack(fresh)
                files[name] = size
                for i, digest in waiting:
                    place = written[digest]
                    refs[i] = (digest, place[3])
                    places[digest.hex()] = [name, place[1], place[2], place[3].hex()]
        except BaseException:
            self.abandon(list(files.items()))
            raise
        return refs, places, list(files.items()), added

    def _hold_stored(self, name: str) -> int | None:
        """Count the stored file *name* as held by this write, touched so a
        remover in another process leaves it: its bytes, or None (unheld
        again) when it is gone."""
        with self.lock:
            self._pending[name] = self._pending.get(name, 0) + 1
        path = self._path(name)
        with self._untracked():
            try:
                os.utime(path)
                size = os.stat(path).st_size
            except OSError:
                self._unpend([name])
                return None
        self._note_own(name, path)
        return size

    def _hold_chunk(self, name: str, nbytes: int) -> bytes | None:
        """Count the chunk file *name* of an entry from before packs as held
        by this write, if it is there and *nbytes* long: its crc32, or None."""
        with self.lock:
            self._pending[name] = self._pending.get(name, 0) + 1
        path = self._path(name)
        crc = self._reuse(path, nbytes)
        if crc is None:
            self._unpend([name])
            return None
        self._note_own(name, path)
        return crc

    def _forget(self, name: str) -> None:
        """The pack *name* is gone: its chunks are placed nowhere."""
        with self.lock:
            for digest in self._in_pack.pop(name, ()):
                if self._places.get(digest, ("",))[0] == name:
                    del self._places[digest]

    def _learn(self, name: str, placed: dict[bytes, Place]) -> None:
        """Under the lock."""
        self._in_pack.setdefault(name, set()).update(placed)
        self._places.update(placed)

    def _write_pack(self, fresh: dict[bytes, memoryview]) -> tuple[str, int, int, dict[bytes, Place]]:
        """Write *fresh* (digest -> bytes) as one pack: a temp file renamed
        into place. Returns its name, its bytes, the bytes it adds to the
        folder, and where each chunk is in it."""
        members = list(fresh)
        name = pack_name(members)
        with self.lock:
            self._pending[name] = self._pending.get(name, 0) + 1
        path = self._path(name)
        placed: dict[bytes, Place] = {}
        try:
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
                    offset = 0
                    try:
                        for digest in members:
                            view = fresh[digest]
                            pad = _aligned(offset)
                            if pad:
                                write_all(fd, bytes(pad))
                                offset += pad
                            crc = zlib.crc32(view).to_bytes(4, "big")
                            write_all(fd, view)
                            write_all(fd, crc)
                            placed[digest] = (name, offset, view.nbytes, crc)
                            offset += view.nbytes + TRAILER
                    finally:
                        os.close(fd)
                    replace_with_retry(tmp, path)
                except BaseException:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise
            self._note_own(name, path)
        except BaseException:
            self._unpend([name])
            raise
        with self.lock:
            self._learn(name, placed)
        return name, offset, offset - before, placed

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

    def commit(
        self,
        stem: str,
        held: list[tuple[str, int]],
        replaced: list[tuple[str, int]] | None,
        places: dict[str, list] | None = None,
    ) -> int:
        """The entry *stem* is in place, holding *held* (``(name, file
        bytes)``, one item per file `put_all` held). *replaced*: the chunks
        of the entry it replaced, as that entry's metadata listed them (None:
        there was none). *places*: the entry's ``chunk_places``. Returns the
        bytes freed by chunks nothing holds any more, and by packs made
        smaller (`_compact`)."""
        names = frozenset(name for name, _size in held)
        with self.lock:
            for name, size in held:
                self._size[name] = size
            self._unpend_locked(name for name, _size in held)
            before = self._held_by.get(stem)
            if before is None:
                before = frozenset(name for name, _size in replaced or ())
            self._set_holder(stem, names, before)
            self._set_placed(stem, places)
            gone = before - names
        return self._drop(gone) + self._compact(gone | names)

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
            self._placed_by.pop(stem, None)
        return self._drop(before) + self._compact(before)

    def _set_placed(self, stem: str, places: Any) -> None:
        """Under the lock."""
        parsed = dict(_parse_places(places))
        if parsed:
            self._placed_by[stem] = parsed
        else:
            self._placed_by.pop(stem, None)

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
        placed: dict[str, dict[bytes, Place]] = {}
        entry_places: dict[str, Any] = {}
        for stem, held, places in self._entry_chunks():
            entry_places[stem] = places
            names = []
            for name, size in held:
                names.append(name)
                sizes[name] = size
            found[stem] = frozenset(names)
            for digest, place in _parse_places(places):
                placed.setdefault(place[0], {})[digest] = place
        with self.lock:
            if self.indexed:
                return 0
            for name, size in sizes.items():
                self._size.setdefault(name, size)
            for name, chunks in placed.items():
                self._learn(name, chunks)
            for stem, names in found.items():
                if stem not in self._held_by:
                    self._set_holder(stem, names, frozenset())
                    self._set_placed(stem, entry_places.get(stem))
            self._old_chunks = self._old_chunks or any(name.endswith(CHUNK_SUFFIX) for name in sizes)
            self._indexed_at = started
            self.indexed = True
        return self._sweep_orphans()

    def _sweep_orphans(self) -> int:
        cutoff = time.time() - ORPHAN_CHUNK_AGE
        freed = 0
        try:
            with self._untracked(), os.scandir(self.dir) as found:
                listed = [(e.name, e.path) for e in found if e.name.endswith((CHUNK_SUFFIX, PACK_SUFFIX, CLAIM_SUFFIX))]
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
            self._forget(name)
        return freed

    def _compact(self, names: Iterable[str]) -> int:
        """Rewrite each pack of *names* of which more than a third is chunks
        no entry holds any more: its live chunks go into a new pack, and each entry holding
        it is told where they are now (`moved_head`, `_repoint`). A pack lives as long
        as any entry holds a chunk in it, so without this an evicted entry's
        own chunks stayed on disk beside the shared ones its pack also held.
        Returns the bytes freed.

        Only a pack every holder of which this process knows the places of;
        and only when every holder can be told, checked before anything is
        written. A process reading the old pack meanwhile gets a miss, never
        other bytes: each chunk is checked against its entry's crc32.
        """
        freed = 0
        for name in names:
            if not name.endswith(PACK_SUFFIX):
                continue
            try:
                freed += self._compact_one(name)
            except Exception:
                logger.debug("could not compact chunk pack %s", name, exc_info=True)
        return freed

    def _compact_one(self, name: str) -> int:
        with self.lock:
            holders = set(self._holders.get(name) or ())
            if not holders or name in self._pending or any(s not in self._placed_by for s in holders):
                return 0
            live: dict[bytes, Place] = {}
            for stem in holders:
                for digest, place in self._placed_by[stem].items():
                    if place[0] == name:
                        live[digest] = place
            size = self._size.get(name, 0)
        kept = sum(place[2] + TRAILER + PACK_ALIGN for place in live.values())
        if not live or 3 * kept > 2 * size:
            return 0  # a third or less of it dead: not worth copying the rest
        members = sorted(live, key=lambda d: live[d][1])
        new_name = pack_name(members)
        if new_name == name:
            return 0
        claim = self._claim(name)
        if claim is None:
            return 0  # another process is rewriting it
        try:
            return self._rewrite(name, new_name, members, live, holders)
        finally:
            with self._untracked():
                try:
                    os.remove(claim)
                except OSError:
                    logger.debug("could not remove claim %s", claim, exc_info=True)

    def _claim(self, name: str) -> str | None:
        """Claim the pack *name* for rewriting, across processes: create
        ``<name>.claim`` exclusively. None when another process holds it; a
        claim older than `ORPHAN_CHUNK_AGE`, left by a crash, is removed, and
        the pack is left for the next time."""
        path = self._path(name + CLAIM_SUFFIX)
        with self._untracked():
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
                return path
            except FileExistsError:
                try:
                    if os.stat(path).st_mtime < time.time() - ORPHAN_CHUNK_AGE:
                        os.remove(path)
                except OSError:
                    pass
            except OSError:
                pass
        return None

    def _rewrite(
        self, name: str, new_name: str, members: list[bytes], live: dict[bytes, Place], holders: set[str]
    ) -> int:
        """`_compact_one` for the pack *name*, claimed: its *live* chunks
        (*members*, in order) into the pack *new_name*, each of *holders*
        told, then *name* removed."""
        paths = {stem: os.path.join(self.cache_dir, stem + ENTRY_SUFFIX) for stem in holders}
        # Where each live chunk will be, to check every holder can be told.
        planned: dict[str, list] = {}
        offset = 0
        for digest in members:
            offset += _aligned(offset)
            planned[digest.hex()] = [new_name, offset, live[digest][2], live[digest][3].hex()]
            offset += live[digest][2] + TRAILER
        with self._untracked():
            if any(moved_head(p, name, (new_name, offset), planned) is None for p in paths.values()):
                return 0
            data = _read_pack(
                self._path(name), [(i, live[d][1], live[d][2], live[d][3]) for i, d in enumerate(members)]
            )
            # Another process may have written the same pack (the same
            # members): entries of its may name it, so it is never removed here.
            existed = os.path.exists(self._path(new_name))
        # 1. The new pack, beside the old one.
        new_name, new_size, added, placed = self._write_pack({d: as_bytes(data[i]) for i, d in enumerate(members)})
        moved = 0
        try:
            # 2. Each holder, rewritten whole and renamed into place.
            for stem, path in paths.items():
                if self._repoint(stem, path, name, (new_name, new_size), planned, placed):
                    moved += 1
        finally:
            with self.lock:
                self._size[new_name] = new_size
            self._unpend([new_name])
        if not moved:
            if not existed:
                self._drop([new_name])
            return 0
        # 3. The old pack, once no holder this process knows points at it.
        return self._drop([name]) - added

    def entry_lock(self, stem: str) -> Any:
        """The lock a write, a removal or an in-place update of the entry
        *stem* holds in this process, and `_repoint` takes before rewriting
        it: one of a fixed few, by the stem's hash."""
        return self._entry_locks[hash(stem) % len(self._entry_locks)]

    def _repoint(
        self,
        stem: str,
        path: str,
        old: str,
        new: tuple[str, int],
        planned: dict[str, list],
        placed: dict[bytes, Place],
    ) -> bool:
        """Rewrite the entry *stem* at *path* to read its chunks in *old*
        from *new*: a temp file with the new metadata and the payload as it
        was, renamed over the entry. A crash leaves the old entry or the new
        one, each whole.

        Under the entry's lock (`entry_lock`), so no write, removal or access
        update of it in this process lands between reading it and putting
        the new one in place; skipped (False) when another thread holds it.
        Another process's write is never undone either (`_swap_if_same`):
        the rewrite goes in only if the file it takes away is the entry it
        read, else that write stays. Its worst case is a miss -- the entry
        briefly absent, or this rewrite dropped -- and an access-time update
        another process made in that moment lost.
        """
        lock = self.entry_lock(stem)
        if not lock.acquire(blocking=False):
            return False
        try:
            with self._untracked():
                made = moved_head(path, old, new, planned)
                if made is None:
                    return False
                head, start, identity = made
                directory = os.path.dirname(path)
                fd, tmp = create_temp_file(directory, prefix=owned_temp_prefix())
                try:
                    try:
                        write_all(fd, memoryview(head))
                        with open(path, "rb") as src:
                            src.seek(start)
                            while block := src.read(1 << 20):
                                write_all(fd, memoryview(block))
                    finally:
                        os.close(fd)
                    _swap_if_same(path, tmp, identity, directory)
                except BaseException as exc:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    if isinstance(exc, _Replaced):
                        return False
                    raise
            with self.lock:
                held = self._held_by.get(stem, frozenset())
                self._set_holder(stem, (held - {old}) | {new[0]}, held)
                self._placed_by[stem] = {
                    d: (placed[d] if p[0] == old else p) for d, p in self._placed_by.get(stem, {}).items()
                }
            return True
        finally:
            lock.release()

    def damaged(self, pack: str, offset: int) -> list[str]:
        """The chunk at *offset* in *pack* failed its check: this process
        places nothing there any more, so the next write of those bytes
        stores them afresh. Returns the entries known to hold it, to remove:
        the rest of the pack stays, and with it every other entry's chunks."""
        self.ensure_index()
        with self.lock:
            for digest, place in list(self._places.items()):
                if place[0] == pack and place[1] == offset:
                    del self._places[digest]
                    self._in_pack.get(pack, set()).discard(digest)
            return [
                stem
                for stem, placed in self._placed_by.items()
                if any(p[0] == pack and p[1] == offset for p in placed.values())
            ]

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
            self._places.clear()
            self._in_pack.clear()
            self.indexed = True
            self._indexed_at = time.time_ns()
        try:
            with self._untracked(), os.scandir(self.dir) as found:
                paths = [e.path for e in found if e.name.endswith((CHUNK_SUFFIX, PACK_SUFFIX, CLAIM_SUFFIX))]
        except OSError:
            return
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                logger.debug("Could not remove chunk %s during clear", path, exc_info=True)


def _parse_places(places: Any) -> list[tuple[bytes, Place]]:
    """An entry's ``chunk_places`` as ``(digest, place)``; what does not
    parse is left out (that entry's chunks are then written again)."""
    out = []
    if not isinstance(places, dict):
        return out
    for key, value in places.items():
        try:
            name, offset, nbytes, crc = value
            if not (isinstance(name, str) and name.endswith(PACK_SUFFIX)):
                continue
            out.append((bytes.fromhex(key), (name, int(offset), int(nbytes), bytes.fromhex(crc))))
        except (TypeError, ValueError):
            continue
    return out


def read_chunks(
    cache_dir: str, wanted: dict[int, tuple[bytes, bytes]], lengths: Any, places: dict[str, Any] | None
) -> dict[int, Any]:
    """The buffers *wanted* names (``index -> (digest, crc32)``), each of
    ``lengths[index]`` bytes: from where *places* (``digest hex -> [file,
    offset, bytes, crc32 hex]``, the entry's ``chunk_places``) says, each
    file opened once, or from the chunk file named by its digest.

    Raises `ChunkError` as `read_chunk` does."""
    by_file: dict[str, list[tuple[int, int, int, bytes]]] = {}
    out: dict[int, Any] = {}
    for index, (digest, crc) in wanted.items():
        place = (places or {}).get(digest.hex())
        if place is None:
            out[index] = read_chunk(cache_dir, digest, lengths[index], crc)
            continue
        name, offset, nbytes = place[0], int(place[1]), int(place[2])
        if nbytes != lengths[index] or not name.endswith(PACK_SUFFIX) or os.sep in name or "/" in name:
            raise ChunkError(f"chunk {digest.hex()}: the entry places it at {place!r}")
        by_file.setdefault(name, []).append((index, offset, nbytes, crc))
    for name, members in by_file.items():
        out.update(_read_pack(os.path.join(cache_dir, CHUNK_DIRNAME, name), members))
    return out


def _read_pack(path: str, members: list[tuple[int, int, int, bytes]]) -> dict[int, Any]:
    """*members* (``(index, offset, bytes, crc32)``) of the pack at *path*,
    each checked against its crc32; a view of one private copy-on-write
    mapping of the file when it is big, else a new ``bytearray`` each."""
    try:
        fh = open(path, "rb")
    except OSError as exc:
        raise ChunkError(f"chunk {path}: {exc}") from exc
    out: dict[int, Any] = {}
    with fh:
        size = os.fstat(fh.fileno()).st_size
        for _index, offset, nbytes, _crc in members:
            if offset < 0 or offset % PACK_ALIGN or offset + nbytes + TRAILER > size:
                raise ChunkError(f"chunk {path}: {size} bytes, the entry places a chunk at {offset}+{nbytes}")
        mapped = None
        if _CAN_MAP and sum(m[2] for m in members) >= MAP_MIN_BYTES:
            try:
                mapped = memoryview(mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY, trackfd=False))
            except (OSError, ValueError, TypeError):
                mapped = None
        first: dict[int, tuple[int, int, bytes]] = {}
        for index, offset, nbytes, crc in members:
            if offset in first:
                at, size, want = first[offset]
                if (size, want) != (nbytes, crc):
                    raise ChunkError(f"chunk {path}: the entry places two different chunks at {offset}")
                # The same bytes twice in one value: two buffers, as two reads
                # of a chunk file made them, never one shared by both.
                out[index] = bytearray(out[at])
                continue
            first[offset] = (index, nbytes, crc)
            if mapped is not None:
                data = mapped[offset : offset + nbytes]
            else:
                data = bytearray(nbytes)
                view = memoryview(data)
                fh.seek(offset)
                got = 0
                while got < nbytes:
                    k = fh.readinto(view[got:])
                    if not k:
                        raise ChunkError(f"chunk {path}: truncated")
                    got += k
            if zlib.crc32(data).to_bytes(4, "big") != crc:
                # Only this chunk's holders go (`ChunkStore.damaged`): the
                # rest of the pack is other entries' chunks, still good.
                raise DamagedChunk(path, offset)
            out[index] = data
    return out
