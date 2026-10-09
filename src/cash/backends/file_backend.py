"""File-based cache backend: one file per entry, written in the background.

`FileBackend` reads and writes entries. Keeping the directory under its byte
cap is `file_eviction.FileEvictor`'s job, and the format stamp and the other
directory-wide files are `cache_dir`'s.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import pickle
import sys
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any, NamedTuple

from cash._paths import replace_with_retry
from cash.exceptions import CacheBackendError

from ..config.schema import CashConfig
from ..tracking.read_classification import register_cache_dir
from ..tracking.tracker_context import untracked
from ._base import CacheBackend, MetadataDict, entry_expired
from ._writes import PendingWrites
from .cache_dir import (
    CacheDirStamp,
    create_temp_file,
    is_cash_file,
    owned_temp_prefix,
    remove_orphan_temp_files,
    warn_if_unwritable,
    warn_unusable,
    write_all,
)
from .entry_format import (
    ENTRY_SUFFIX,
    CorruptEntry,
    PackedEntry,
    SplitPayload,
    entry_identity,
    pack_entry_parts,
    payload_checksum,
    read_entry,
    read_entry_and_checksum,
    split_chunks,
    update_metadata_in_place,
)
from .file_eviction import FileEvictor
from .serialization import RESTORE_ERRORS, PickleSerializer, Serializer, rebuild, restore_value
from .touched_entries import TouchedEntries, stat_signature
from .versions import VersionIndex, superseded_to_drop

logger = logging.getLogger(__name__)

__all__ = ["FileBackend", "StoredEntry"]


class StoredEntry(NamedTuple):
    """One entry file as `FileBackend.entries` finds it."""

    #: The file name without its suffix: a SHA-256 of the key.
    id: str
    key: str
    #: The whole file's size on disk.
    size: int
    mtime: float
    metadata: dict[str, Any]


# Every live write queue, grouped by the cache directory it writes into. Two
# FileBackends over one directory are two views of one store, but each has its
# own queue; a read waits on all of them (``_wait_for_writes``), or a second
# instance reading a directory the first is still writing reports clean misses.
# Weak, so a backend going out of scope is neither kept alive nor waited on.
_WRITERS_BY_DIR: dict[str, weakref.WeakSet] = {}
_WRITERS_LOCK = threading.Lock()


# Every FileBackend alive in the process, for `_reset_after_fork_in_child`.
_LIVE_BACKENDS: weakref.WeakSet = weakref.WeakSet()


def _reset_after_fork_in_child() -> None:
    """New locks for this module and every live `FileBackend` in a forked child.

    Only the forking thread survives a fork. A lock another thread held at
    that moment -- the background writer recording an entry it had just
    written -- stays held in the child forever. The queues themselves are
    reset by ``_writes`` (`PendingWrites._after_fork_in_child`).
    """
    global _WRITERS_LOCK
    _WRITERS_LOCK = threading.Lock()
    for backend in list(_LIVE_BACKENDS):
        backend._after_fork_in_child()


if hasattr(os, "register_at_fork"):  # not on Windows, which cannot fork
    os.register_at_fork(after_in_child=_reset_after_fork_in_child)


def _writer_scope(cache_dir: str) -> str:
    """Normalized identity of a cache directory (symlinks and relative paths resolved)."""
    try:
        return os.path.realpath(cache_dir)
    except OSError:
        return os.path.abspath(cache_dir)


#: Modules a listing passes through on its way from the code that asked for
#: it: ``glob.glob`` and ``Path.glob`` list with ``os.scandir`` themselves.
_LISTING_HELPERS = ("glob", "_glob", "pathlib", "os", "posixpath", "ntpath", "fnmatch", "shutil")


def _asked_by_storage() -> bool:
    """Did the storage code itself list the directory?

    It waits for the writes it needs to see (``list_entries``, ``clear``),
    and an eviction scan must not wait for the writes behind it. Past the
    stdlib's helpers and cash's own wrappers of them (the file tracker's
    ``glob``): those list for the user's code. Called from the audit
    consumer below, whose caller is the audit hook; the code that listed is
    two frames up.
    """
    frame = sys._getframe(3)
    for _ in range(12):
        if frame is None:
            return False
        name = frame.f_globals.get("__name__", "") or ""
        if name.startswith("cash.backends."):
            return True
        top = name.partition(".")[0]
        if top != "cash" and top not in _LISTING_HELPERS:
            return False
        frame = frame.f_back
    return False


def _finish_before_listing(args: tuple) -> None:
    """A listing of a cache directory sees every write already queued for it.

    The ``os.scandir`` / ``os.listdir`` audit events: code in this process
    looking at the folder itself (``glob('.cash/*.entry')`` in a later cell)
    reads it from disk, so the writes queued for it land first. The storage
    code's own listings wait where they need to (``list_entries``,
    ``clear``), and the write workers never wait. Never raises.
    """
    try:
        if PendingWrites.in_worker_thread():
            return
        with _WRITERS_LOCK:
            busy = {scope: list(b) for scope, b in _WRITERS_BY_DIR.items() if any(w.pending_count() for w in b)}
        if not busy:
            return
        path = args[0] if args else None
        if isinstance(path, int):
            return  # a directory descriptor: no name to compare
        scope = _writer_scope(os.fsdecode(os.fspath(path)) if path is not None else os.curdir)
        writers = busy.get(scope)
        if writers and not _asked_by_storage():
            for writer in writers:
                writer.wait_all()
    except Exception:  # noqa: BLE001 - must never fail the user's listing
        logger.debug("Finishing cache writes before a listing failed", exc_info=True)


_listing_watched = False


def _watch_listings() -> None:
    global _listing_watched
    if _listing_watched:
        return
    from ..tracking import io_watch

    for event in ("os.scandir", "os.listdir"):
        io_watch.subscribe_always(event, _finish_before_listing)
    _listing_watched = True


def _register_writer(cache_dir: str, writes: PendingWrites) -> str:
    """Register *writes* under *cache_dir*'s scope, and return the scope."""
    scope = _writer_scope(cache_dir)
    # Tell the file tracker this directory is cash's own storage, so its entry
    # files never become dependencies of the user's code whatever it is called.
    try:
        register_cache_dir(cache_dir, is_cash_file)
    except Exception:  # tracking is best-effort, storage is not
        logger.debug("Could not register %s with the file tracker", cache_dir, exc_info=True)
    with _WRITERS_LOCK:
        if not _listing_watched:
            _watch_listings()
        bucket = _WRITERS_BY_DIR.get(scope)
        if bucket is None:
            bucket = weakref.WeakSet()
            _WRITERS_BY_DIR[scope] = bucket
        bucket.add(writes)
        # Drop scopes whose backends have all been collected.
        if len(_WRITERS_BY_DIR) > 64:
            for dead in [s for s, b in _WRITERS_BY_DIR.items() if not b]:
                del _WRITERS_BY_DIR[dead]
    return scope


def _sibling_writers(scope: str, own: PendingWrites) -> list[PendingWrites]:
    """Live write queues over the directory *scope* names, other than *own*.

    Takes the scope `_register_writer` returned: resolving the directory again
    is a ``realpath`` on every read."""
    with _WRITERS_LOCK:
        bucket = _WRITERS_BY_DIR.get(scope)
        return [w for w in bucket if w is not own] if bucket else []


def _untracked() -> Any:
    """cash's own I/O on its cache directory, kept out of every dependency.

    A nested cached call's store scans the directory while the OUTER call's
    file tracker is live, and the file filters cannot tell that listing from a
    user's listing of a directory ``cache_dir`` may also be.
    """
    return untracked()


class FileBackend(CacheBackend):
    """One file per entry in a directory; survives restarts.

    Several processes can share the directory. Entries are pickled, and
    loading one runs code, so never point it at a directory from an
    untrusted source (see Security on the Backends page).
    """

    source_label: str = "DISK"
    #: `set` takes ``private=True`` (see there).
    takes_private_values: bool = True

    def __init__(
        self,
        cache_dir: str,
        compress: bool = False,
        max_size_bytes: int | None = None,
        flush_interval: int = CashConfig.flush_interval,
        default_ttl: int | None = None,
        adaptive_cap: bool = False,
    ) -> None:
        """
        Args:
            cache_dir: Directory for cache files.
            compress: Whether to gzip-compress data files.
            max_size_bytes: Byte cap. A write that goes over it evicts the
                entries worth least per byte. ``None``: no cap.
            flush_interval: Seconds between writes of access statistics.
            default_ttl: Seconds an entry stays valid when the caller gives
                no ``ttl``. ``None``: no expiry.
            adaptive_cap: Leave ``False``. Set by cash when the cap was
                sized to the machine, so it may be resized.
        """
        # Absolute, so a later os.chdir() cannot move the cache.
        self.cache_dir = os.path.abspath(cache_dir)
        self.compress = compress
        self._default_ttl = default_ttl
        #: key -> when its access stamp was last written, for the flusher's rate limit.
        self._access_flushed: dict[str, float] = {}
        self._touched = TouchedEntries()
        #: Each statement's versions, pruned as they are written (versions.py).
        #: A key this process has read is in use and never pruned.
        self._versions = VersionIndex(self.cache_dir, _untracked)
        self._read_keys: set[str] = set()
        self._orphans_swept = False
        self._flush_interval = flush_interval
        self._stop_event = threading.Event()

        # The disk I/O runs here; values are serialized on the caller's
        # thread, unless they are `set` ``private``.
        self._writes = PendingWrites()
        self._writer_scope = _register_writer(self.cache_dir, self._writes)
        self.stamp = CacheDirStamp(self.cache_dir, _untracked)
        self.evictor = FileEvictor(
            self.cache_dir,
            max_size_bytes,
            adaptive_cap,
            touched=self._touched,
            writes=self._writes,
            untracked=_untracked,
        )

        # Directory creation, the stamp check and the flusher wait for first use.
        self._initialized = False
        self._init_lock = threading.Lock()
        #: Set when the cache directory turned out to be unusable. Every public
        #: operation then answers as an empty cache would: a miss, a no-op write.
        self._unusable = False
        _LIVE_BACKENDS.add(self)

    def _after_fork_in_child(self) -> None:
        """Replace the locks a thread of the parent may have held at the fork."""
        self._init_lock = threading.Lock()
        self._touched.lock = threading.RLock()
        # The evictor's accounting shares the touched-entries lock.
        self.evictor._lock = self._touched.lock
        self.evictor.rank_index._lock = threading.Lock()
        self.evictor.evictions._lock = threading.Lock()
        self._versions._lock = threading.Lock()

    @property
    def written_stamp(self) -> tuple | None:  # type: ignore[override]
        return self.stamp.written

    @property
    def stamp_writes(self) -> int:  # type: ignore[override]
        return self.stamp.writes

    def _ensure_initialized(self) -> None:
        """Create the directory, check its format stamp and start the flusher,
        once, on the first cache operation.

        O(1) in the number of entries: it runs on the caller's thread, so
        nothing that walks the directory belongs here.
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                created = not os.path.isdir(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                if created:
                    self.stamp.ignore_in_git()
            except OSError as exc:
                # A directory that cannot even be created (read-only volume, a
                # path that is a file) turns this tier off, loudly, instead of
                # failing the caller's program: caching is best effort, and in
                # a tiered stack the RAM tier still works.
                self._disable(exc)
                return
            self.stamp.check()
            warn_if_unwritable(self.cache_dir)
            if self._flush_interval > 0:
                self._flusher_thread = threading.Thread(
                    target=self._flush_periodically,
                    daemon=True,
                )
                self._flusher_thread.start()
            self._initialized = True

    def _disable(self, exc: BaseException) -> None:
        """Turn this tier off for the rest of the process, and say why once."""
        self._unusable = True
        self._initialized = True  # never retried; the answer will not change
        warn_unusable(self.cache_dir, exc)

    def generation_token(self) -> tuple | None:
        """The format stamp's identity: it moves when the directory is cleared
        under this process (see `CacheDirStamp`). None when there is no stamp."""
        return self.stamp.token()

    def _flush_periodically(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._flush_interval):
                break
            self._flush_metadata(periodic=True)
            # Buffered rank records reach the file within one interval, so
            # another process ranking this directory sees them.
            self.evictor.rank_index.flush()

    #: Seconds between persisted access stamps for one entry, while the
    #: process runs; everything outstanding is flushed at shutdown.
    _ACCESS_FLUSH_MIN_INTERVAL = 600.0

    def _flush_metadata(self, periodic: bool = False) -> None:
        """Write the access stamps of recently read entries back, in place.

        ``update_metadata_in_place`` rewrites only the header and metadata
        region, so recording a read never rewrites the payload. Metadata that
        outgrew its reserved slack is skipped: what is lost is ranking
        precision for one entry, never a value.

        *periodic* (the flusher thread) writes an entry's stamp at most once
        per `_ACCESS_FLUSH_MIN_INTERVAL`: each modification of a file in a
        synced folder re-uploads the whole file. Shutdown flushes everything.
        """
        now = time.time()
        due = None
        if periodic:

            def due(k: str) -> bool:
                return now - self._access_flushed.get(k, 0.0) >= self._ACCESS_FLUSH_MIN_INTERVAL

        keys_to_flush = self._touched.take_unflushed(due)
        if not keys_to_flush:
            return

        # A read raises an entry's priority, so a flushed access is also a
        # rank-index record; that is how an old entry still in use gets ranked.
        ranked = self.evictor.capped
        if ranked and keys_to_flush:
            self.evictor.ensure_clock()
        records: list[tuple[str, float]] = []
        for key in keys_to_flush:
            try:
                meta = self._touched.metadata(key)
                path = self._get_path(key)
                if meta and not update_metadata_in_place(path, meta, self._touched.identity(key)):
                    # Replaced by another process since it was read, or grown
                    # past its reserved region. Either way what is held here
                    # is not what the file holds: read it afresh next time.
                    logger.debug("Access stats for %r not flushed: the entry changed or outgrew its region", key)
                    self._touched.drop_metadata(key)
                    continue
                self._access_flushed[key] = now
                if meta and meta.get("version_slot"):
                    self._versions.touch(key, meta.get("last_access", now))
                if meta and ranked:
                    records.append(self.evictor.access_record(key, meta, path))
            except (OSError, pickle.PickleError) as exc:
                logger.debug("Failed to flush metadata for key %r: %s", key, exc)
        self.evictor.rank_index.append(records)

    def _remember(self, key: str, metadata: dict, checksum: bytes | None, stat: tuple | None = None) -> None:
        """Hold one entry's metadata, and the way back from its filename.

        *checksum* is the payload checksum of the write *metadata* belongs to,
        *stat* the file's `stat_signature` when it was read, if known."""
        self._touched.remember(key, self._get_path(key), metadata, entry_identity(metadata, checksum), stat)

    @staticmethod
    def _stem(key: str) -> str:
        """*key*'s entry file name without the suffix: a SHA-256 of the key."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _get_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{self._stem(key)}{ENTRY_SUFFIX}")

    @property
    def local_dir(self) -> str:
        return self.cache_dir

    def get_metadata(self, key: str) -> dict | None:
        """Get only metadata for a cache key without deserializing the value.

        This is useful for lazy deserialization - check if a key exists and
        inspect its metadata before committing to deserialize the data.
        Also returns metadata-only entries (where data was skipped due to
        size-aware caching) — these still carry execution_time, output_lineages,
        etc. for badge display and upstream simulation.

        Reads the header and the metadata region and stops there, so this
        costs the same for a 200MB entry as for a 200-byte one. That property
        is the reason metadata and payload can share a file at all.

        Returns:
            Metadata dict if key exists, None otherwise.
        """
        self._ensure_initialized()
        if self._unusable:
            return None
        # Wait for any in-flight write so the metadata we report reflects
        # the most recent ``set()`` for this key, by any backend over this
        # directory.
        self._wait_for_writes(key)
        cached_meta = self._touched.metadata(key)

        path = self._get_path(key)

        try:
            metadata = None
            if cached_meta is not None:
                # Cheaper than reopening, but the file must still be the one
                # it was read from: a delete by another process must read as
                # absent, and a rewrite as the new entry, never as this
                # process's stale copy.
                st = stat_signature(os.stat(path))
                if self._touched.stat_matches(key, st):
                    metadata = cached_meta
            if metadata is None:
                on_disk, _, checksum = read_entry_and_checksum(path, with_payload=False)
                if cached_meta is not None and entry_identity(on_disk, checksum) == self._touched.identity(key):
                    # The same write, touched since (an access stamp flushed).
                    metadata = cached_meta
                    self._touched.note_stat(key, st)
                else:
                    metadata = on_disk
                    self._remember(key, metadata, checksum, st if cached_meta is not None else None)

            if entry_expired(metadata, self._default_ttl):
                return None

            return metadata
        except FileNotFoundError:
            return None
        except (OSError, CorruptEntry):
            logger.debug("Unreadable metadata for key %s; treating as absent", key, exc_info=True)
            return None

    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        self._ensure_initialized()
        if self._unusable:
            return None, None
        # Wait for any in-flight write for this key so we never return
        # stale-or-missing data when get() races set().
        self._wait_for_writes(key)
        # Check memory cache first for metadata. The cached dict is preferred
        # over the one on disk even though we are about to read the file
        # anyway: callers hold it, `get` mutates it in place, and the flusher
        # writes THAT object back. Replacing it with a fresh dict per read
        # would drop every unflushed access update on the floor. Only while
        # the file still holds the write it was read from, though: another
        # process may have replaced the entry, and its payload paired with
        # the old metadata (the old inputs' file hashes) is a wrong value.
        cached_meta = self._touched.metadata(key)

        path = self._get_path(key)

        try:
            on_disk, payload, checksum = read_entry_and_checksum(path, with_payload=True)
        except FileNotFoundError:
            self._touched.drop_metadata(key)
            return None, None
        except (OSError, CorruptEntry) as exc:
            logger.debug("Cache get failed for key %r: %s", key, exc)
            return None, None

        try:
            if cached_meta is not None and entry_identity(on_disk, checksum) == self._touched.identity(key):
                metadata = cached_meta
            else:
                metadata = on_disk
                self._remember(key, metadata, checksum)

            # A metadata-only entry (the value was too large to persist) has
            # no payload. It is a HIT for `get_metadata`, which wants the
            # execution time and lineages, and a MISS here, because there is
            # nothing to restore. With two files that fell out of requiring
            # both to exist; with one it has to be asked explicitly.
            if metadata.get("metadata_only"):
                return None, None

            if entry_expired(metadata, self._default_ttl):
                self.delete(key)
                return None, None

            # Update Access Time (Async)
            metadata["last_access"] = time.time()
            metadata["access_count"] = metadata.get("access_count", 0) + 1

            self._touched.note_read(key)
            self._read_keys.add(key)
            self.evictor.note_read(key)

            if metadata.get("compressed", False):
                try:
                    payload = gzip.decompress(payload)
                except (OSError, gzip.BadGzipFile, EOFError):
                    # Flag says compressed but the bytes are not: the raw
                    # bytes are the payload.
                    logger.debug("Entry for %r flagged compressed but is not", key)

            if isinstance(payload, SplitPayload):
                # Written only for a PickleSerializer value (`set`).
                value = rebuild(PickleSerializer().deserialize_split, payload.stream, payload.buffers)
            else:
                value = restore_value(metadata, payload)

            metadata.setdefault("source", self.source_label)
            return metadata, value
        except (OSError, *RESTORE_ERRORS) as exc:
            logger.debug("Cache get failed for key %r: %s", key, exc)
            return None, None

    def _wait_for_writes(self, key: str) -> None:
        """Wait for every live write to *key* in this cache directory.

        Own queue first, then any sibling backend's (see ``_WRITERS_BY_DIR``):
        the durability boundary is the directory, not the instance.

        A background write worker never waits on a sibling. If it did, two
        backends over one directory could each have their worker blocked on the
        other's future and deadlock. Workers only ever touch their own queue,
        which the existing per-key re-entrancy guard already handles.

        Sibling failures are swallowed rather than re-raised: a failed write
        cleans up its partial files, so the read below simply finds nothing and
        reports a miss. Surfacing another instance's write error to whoever
        happens to read next would attribute it to the wrong caller.
        """
        self._writes.wait(key)
        if PendingWrites.in_worker_thread():
            return
        for sibling in _sibling_writers(self._writer_scope, self._writes):
            try:
                sibling.wait(key)
            except Exception:
                logger.debug("Sibling write for key %r failed", key, exc_info=True)

    def _atomic_write(self, path: str, payload: bytes | list) -> None:
        """Write *payload* (bytes, or a list of pieces to write in order) to
        *path* so no reader can observe a partial file.

        A temp file in the same directory, renamed into place: a concurrent
        reader sees the previous contents or the complete new ones, never a
        truncated file. Compression, if any, is already in *payload*'s value
        region; the header and metadata stay readable without inflating it.
        """
        directory = os.path.dirname(path) or "."
        # A temp file in the target directory; the leading dot keeps the
        # partial out of the ``*.entry`` glob the backend scans. Its name
        # says which process writes it.
        fd, tmp_path = create_temp_file(directory, prefix=owned_temp_prefix())
        try:
            # Through the descriptor it was created with: reopening the name
            # would raise an ``open`` audit event for the file tracker to
            # sort out, and on Windows opening a file created a moment before
            # is slow.
            try:
                for chunk in [payload] if isinstance(payload, (bytes, bytearray, memoryview)) else payload:
                    write_all(fd, chunk)
            finally:
                os.close(fd)
            replace_with_retry(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Could not remove partial write %s", tmp_path, exc_info=True)
            raise

    def _write_new_in_place(self, path: str, entry: PackedEntry) -> bool:
        """Write a BRAND NEW entry directly, header last. False if one exists.

        Temp-and-rename costs about three times a direct write, and the
        notebook waits for its writes at the end of every cell. What it buys --
        the previous entry untouched until the swap -- means nothing for a key
        that does not exist yet. ``O_EXCL`` makes "does not exist yet" true: two
        processes racing to create one entry cannot both take this path, and
        the loser falls back to the safe one.

        The header goes LAST, so until the payload is there the magic does not
        match and a concurrent reader gets a clean miss. A process killed
        mid-write leaves a headerless entry that reads as a miss, and the next
        write of that key replaces it through the safe path.
        """
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(path, flags)
        except FileExistsError:
            return False  # something is there; it must survive a failure
        except OSError:
            return False  # no directory, no permission -- let the safe path report it

        try:
            # Raw fd writes: ``os.open`` raises the ``open`` audit event with
            # no mode, which neither the file tracker nor the effect observer
            # watches, so the entry never becomes a dependency or an effect of
            # the user's code whatever the directory is called. Unbuffered, the
            # payload and the header reach the page cache in program order.
            os.lseek(fd, len(entry.head), os.SEEK_SET)
            for chunk in entry.chunks:  # payload
                write_all(fd, chunk)
            os.lseek(fd, 0, os.SEEK_SET)
            write_all(fd, entry.head)  # header + metadata, last
            os.close(fd)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            # Ours, and incomplete. Nothing else can be looking at it as a
            # valid entry -- the header was never written.
            try:
                os.remove(path)
            except OSError:
                logger.debug("Could not remove partial write %s", path, exc_info=True)
            raise
        return True

    def _write_cache_files(self, key: str, path: str, metadata: dict, serialized_value: bytes | SplitPayload) -> None:
        """Write one entry -- metadata and payload -- and update size tracking.

        The header, the metadata and each part of the payload are written one
        after the other, never joined into one blob first: for a 100 MB value
        the join and the slice that split it again were two copies of it.

        Raises:
            OSError, pickle.PickleError, ValueError: on write failure (caller handles cleanup).
        """
        if isinstance(serialized_value, SplitPayload):
            chunks = split_chunks(serialized_value.stream, serialized_value.buffers)
            split = True
        else:
            chunks = [gzip.compress(serialized_value) if self.compress else serialized_value]
            split = False
        # The bytes the value occupies on disk, after compression.
        metadata["size"] = sum(len(c) for c in chunks)
        entry = pack_entry_parts(metadata, chunks, split=split)
        checksum = entry.checksum

        # What this entry occupies now, so a rewrite subtracts what was there.
        try:
            old_entry_bytes = os.path.getsize(path)
        except OSError:
            old_entry_bytes = 0

        try:
            # A key that does not exist yet has nothing to preserve, so it
            # skips the temp-and-rename dance. Anything else -- including a
            # concurrent creator that won the O_EXCL race -- takes the safe
            # path, because there a failed write must not destroy what is
            # already cached.
            if not self._write_new_in_place(path, entry):
                self._atomic_write(path, [entry.head, *entry.chunks])
        except FileNotFoundError:
            # The cache directory was deleted under a live process (``cash
            # clear --all``, or by hand): recreate it, stamped first so the next
            # process does not discard these entries as an unknown format, and
            # retry once.
            os.makedirs(self.cache_dir, exist_ok=True)
            self.stamp.ignore_in_git()
            self.stamp.write()
            if not self._write_new_in_place(path, entry):
                self._atomic_write(path, [entry.head, *entry.chunks])

        with self._touched.lock:
            self._remember(key, metadata, checksum)
            self.evictor.note_write(key, old_entry_bytes, entry.size)

        # Only a capped tier ranks, so only a capped tier keeps the rank index:
        # uncapped, it would be a file that only grows.
        if self.evictor.capped:
            self.evictor.record_rank(key, path, metadata, entry.size)

    def set(
        self,
        key: str,
        value: Any,
        metadata: MetadataDict | None = None,
        serializer: Serializer | None = None,
        *,
        private: bool = False,
    ) -> None:
        """Serialize the value on the calling thread, then write to disk
        in the background. ``set()`` returns once the bytes are captured;
        a subsequent ``get(key)`` waits for the write.

        *private*: cash owns *value* and nothing changes it from now on (the
        RAM tier's own copy). Then the background writer serializes it too,
        using its buffers' memory as it is: the caller pays nothing, where a
        caller's value has to be copied before ``set`` may return."""
        self._ensure_initialized()
        if self._unusable:
            return
        path = self._get_path(key)

        metadata = self._init_metadata(metadata, key)

        # Set TTL if not already specified and we have a default
        if "ttl" not in metadata and self._default_ttl is not None:
            metadata["ttl"] = self._default_ttl

        if serializer is None:
            serializer = PickleSerializer()

        if private:
            # Serialized by the writer; the size known now is the caller's.
            nbytes = int(metadata.get("size") or metadata.get("cost_model_size_bytes") or 0)
            job: Callable[..., None] = self._serialize_and_write
            work: Any = (value, serializer)
        else:
            # IMPORTANT: serialize on the calling thread so a post-set()
            # mutation of `value` can't corrupt the cached bytes.
            serialized_value = self._serialize(value, serializer, copy=True)
            nbytes = self._payload_bytes(serialized_value)
            job = self._do_set_sync
            work = serialized_value
            # Pre-compute size from the serialized bytes; the on-disk size
            # may differ under compression but the user-facing metadata
            # needs to be populated synchronously for the badge.
            metadata["size"] = nbytes

        metadata["compressed"] = self.compress
        if "storage" not in metadata:
            metadata["storage"] = [self.source_label]

        # Stored again: a note that the cap evicted it no longer applies.
        self.evictor.evictions.forget(self._stem(key))

        # Freeze a copy of metadata for the background write — the caller
        # can mutate the original after we return without affecting the
        # written entry.
        meta_for_write = dict(metadata)
        # A private value's memory is held by the RAM tier anyway: only the
        # copies made here count against the queue's memory.
        self._writes.submit_sized(key, nbytes, not private, job, key, path, meta_for_write, work)
        slot = metadata.get("version_slot")
        if slot:
            # A version whose values are references to call entries weighs
            # what those hold too (``cash.notebook.call_refs``).
            self._prune_versions(
                slot,
                key,
                nbytes + int(metadata.get("call_ref_bytes") or 0),
                metadata.get("execution_time") or 0.0,
                metadata.get("call_refs") or (),
            )

    def _prune_versions(self, slot: str, key: str, size: int, cost: float, call_refs: Any = ()) -> None:
        """Remove the superseded versions of *key*'s statement that are not
        worth their bytes (``versions.superseded_to_drop``). Best effort: a
        failure here leaves entries for the byte cap, never loses the new one.

        *call_refs* are the new version's own (its metadata's ``call_refs``).
        Reading them back from *key*'s entry instead would wait for the write
        just queued, and make every write of a version synchronous."""
        try:
            versions = self._versions.record(slot, key, size, cost, time.time())
            gone = [k for k in versions if k != key and not os.path.exists(self._get_path(k))]
            for k in gone:
                versions.pop(k)
            drop = superseded_to_drop(versions, key, self._read_keys)
            kept = self._call_refs_of(k for k in versions if k not in drop and k != key)
            freed = self._call_refs_of(drop) - kept - set(call_refs)
            for k in drop:
                self.delete(k)
            # The call entries only the dropped versions referred to go with
            # them. One another statement also refers to makes that one a miss
            # -- it recomputes, never restores something else.
            for k in freed:
                self.delete(k)
            if gone or drop:
                self._versions.forget(gone + drop)
            if drop:
                logger.debug("Pruned %d superseded version(s) of slot %s", len(drop), slot)
        except (OSError, CacheBackendError) as exc:
            logger.debug("Version pruning failed for %r: %s", key, exc)

    def _serialize(self, value: Any, serializer: Serializer, *, copy: bool) -> bytes | SplitPayload:
        """*value* as the bytes an entry stores.

        A `PickleSerializer` value without compression keeps its large
        buffers apart from the stream (``entry_format.MAGIC_SPLIT``), copied
        unless *copy* is False. Compressed, it is one stream: the codec reads
        it once, and the copy into the stream is what keeps a later mutation
        out."""
        if type(serializer) is PickleSerializer and not self.compress:
            stream, buffers = serializer.serialize_split(value, copy=copy)
            return SplitPayload(stream, buffers) if buffers else stream
        return serializer.serialize(value)

    @staticmethod
    def _payload_bytes(serialized_value: bytes | SplitPayload) -> int:
        if isinstance(serialized_value, SplitPayload):
            return len(serialized_value.stream) + sum(memoryview(b).nbytes for b in serialized_value.buffers)
        return len(serialized_value)

    def _serialize_and_write(self, key: str, path: str, metadata: dict, work: tuple) -> None:
        """`_do_set_sync` for a `set` ``private`` value: serialize it here, on
        the writer, without copying its buffers."""
        value, serializer = work
        try:
            serialized_value = self._serialize(value, serializer, copy=False)
        except Exception as exc:  # noqa: BLE001 - whatever the value's pickling raises
            logger.debug("Cache set failed for key %r: %s", key, exc)
            raise CacheBackendError(f"Cache set failed for key {key!r}: {exc}") from exc
        del value, work  # the buffers hold what the write needs
        self._do_set_sync(key, path, metadata, serialized_value)

    def _call_refs_of(self, keys) -> set[str]:
        refs: set[str] = set()
        for k in keys:
            meta = self.get_metadata(k) or {}
            refs.update(meta.get("call_refs") or ())
        return refs

    def _do_set_sync(self, key: str, path: str, metadata: dict, serialized_value: bytes | SplitPayload) -> None:
        """The actual disk write — runs in the PendingWrites worker thread.

        A failure re-raises and touches nothing: the destination is the
        previous entry, untouched because the rename never happened, or absent.
        The exception surfaces on the next ``get(key)`` (or ``shutdown()``).
        """
        if not self._orphans_swept:
            # Once per process, on the write worker: the partial files of
            # writes whose process was killed are in no count and no eviction.
            self._orphans_swept = True
            remove_orphan_temp_files(self.cache_dir)
        try:
            self._write_cache_files(key, path, metadata, serialized_value)
            if self.evictor.capped:
                self.evictor.evict()
        except (OSError, pickle.PickleError, ValueError) as exc:
            logger.debug("Cache set failed for key %r: %s", key, exc)
            raise CacheBackendError(f"Cache set failed for key {key!r}: {exc}") from exc

    def set_metadata_only(self, key: str, metadata: dict) -> None:
        """Store only metadata without data payload.

        Used when the data itself is too large to persist (size-aware caching skip)
        but we still want to retain execution_time, output_lineages, etc.
        for badge display and upstream simulation after kernel restart.

        Will NOT overwrite an existing entry that carries a real payload,
        since that entry is more valuable.
        """
        self._ensure_initialized()
        if self._unusable:
            return
        # Wait for any in-flight async set() for this key to land first —
        # otherwise the check below races a not-yet-flushed full write, sees
        # nothing, and clobbers the (more valuable) full entry with a
        # metadata-only one. get()/get_metadata()/delete() synchronise the
        # same way.
        self._wait_for_writes(key)
        path = self._get_path(key)

        # What this process last wrote there, when it was metadata only, needs
        # no disk read (slow on Windows, and a cheap statement rewrites its
        # entry on every run). If another process has put a full entry there
        # since, this overwrites it -- a later miss, never a wrong value.
        known = self._touched.metadata(key)
        if known is not None and known.get("metadata_only"):
            existing = known
        else:
            try:
                existing, _ = read_entry(path, with_payload=False)
            except (OSError, CorruptEntry):
                existing = None
        if existing is not None and not existing.get("metadata_only"):
            return

        metadata = dict(metadata)  # Don't mutate caller's dict
        metadata["key"] = key
        metadata["metadata_only"] = True
        metadata.setdefault("created_at", time.time())
        metadata.setdefault("last_access", time.time())
        metadata.setdefault("access_count", 0)
        metadata.setdefault("size", 0)

        try:
            entry = pack_entry_parts(metadata, [b""])
            # A new key takes the cheaper in-place write, as a full entry does.
            if existing is not None or not self._write_new_in_place(path, entry):
                self._atomic_write(path, [entry.head])
            self._remember(key, metadata, payload_checksum(b""))
        except OSError as exc:
            logger.debug("Failed to write metadata-only entry for key %r: %s", key, exc)

    def delete(self, key: str) -> None:
        self._ensure_initialized()
        if self._unusable:
            return
        # Drain any pending write for this key — otherwise the write
        # could fire after the delete and leave a ghost entry.
        self._writes.drain(key)
        self.evictor.remove_path(self._get_path(key), key)

    def disk_budget(self) -> Any:
        """This folder's cap and where it comes from (`FileEvictor.budget`)."""
        if self._unusable:
            return None
        return self.evictor.budget()

    def eviction_note(self, key: str) -> Any:
        """Did this tier's cap evict *key*'s entry (`eviction_log`)?"""
        if self._unusable:
            return None
        return self.evictor.evictions.lookup(self._stem(key))

    def take_storage_notices(self) -> list[str]:
        return self.evictor.take_notices()

    def promotion_size_cap(self) -> int | None:
        """Refuse (skip) only an object larger than this tier's WHOLE cap.

        "Keep at most N bytes" reads as: store what fits and evict the rest.
        Any lower threshold refuses values that fit, and a job whose working
        set is a large part of its cap would cache nothing. A write-and-evict
        treadmill is reported when it happens (``CACHE-THRASH``) rather than
        pre-empted. Uncapped, the class-level
        hint applies.
        """
        if self.evictor.max_size_bytes:
            return self.evictor.max_size_bytes
        return type(self).max_size_bytes

    def clear(self) -> None:
        self._ensure_initialized()
        if self._unusable:
            return
        # Drain pending writes so they don't fire after the clear and
        # resurrect entries we just removed from disk.
        self._writes.wait_all()
        for f in self.stamp.entry_files():
            try:
                os.remove(f)
            except OSError:
                # Best-effort removal during cache clear; file may be locked
                logger.debug("Could not remove cache file %s during clear", f, exc_info=True)
        # The priorities and versions describe entries that no longer exist.
        self.evictor.clear()
        self._versions.remove()
        self._touched.clear()

    def shutdown(self) -> None:
        # Drain any in-flight async writes before stopping the flusher,
        # otherwise we could lose the metadata for a not-yet-written entry.
        self._writes.shutdown(wait=True)
        self._stop_event.set()
        if hasattr(self, "_flusher_thread"):
            self._flusher_thread.join(timeout=1.0)
        if self._initialized:
            self._flush_metadata()
            self.evictor.rank_index.flush()

    def list_entries(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self._unusable:
            return []
        # Drain pending writes so the listing reflects everything the
        # caller has already set() — otherwise async writes still in
        # flight would be invisible.
        self._writes.wait_all()
        entries = []
        for path in self.stamp.entry_files():
            try:
                metadata, _ = read_entry(path, with_payload=False)
                entries.append(metadata)
            except (OSError, CorruptEntry):
                logger.debug("Skipping unreadable entry %s in list_entries", path, exc_info=True)
        return entries

    def entry_count(self) -> int:
        """Entry files in the cache directory: one listing, no entry opened.

        Counts an unreadable entry that `list_entries` would skip; the number
        is for display, and reading every header to rule those out takes
        seconds on a large cache.
        """
        self._ensure_initialized()
        if self._unusable:
            return 0
        self._writes.wait_all()
        return len(self.stamp.entry_files())

    def entries(self) -> list[StoredEntry]:
        """Every readable entry, from its metadata region; no payload is read.

        Unlike the other operations it neither creates nor stamps the
        directory, so looking at a cache (``cash inspect``) never changes it.
        """
        self._writes.wait_all()
        found = []
        for path in self.stamp.entry_files():
            try:
                metadata, _ = read_entry(path, with_payload=False)
                st = os.stat(path)
            except (OSError, CorruptEntry):
                logger.debug("Skipping unreadable entry %s", path, exc_info=True)
                continue
            stem = os.path.basename(path)[: -len(ENTRY_SUFFIX)]
            found.append(StoredEntry(stem, metadata.get("key") or "", st.st_size, st.st_mtime, metadata))
        return found

    def bump_generation(self) -> None:
        """Tell processes using this directory that entries were removed under
        them: a running `TieredBackend` notices within a second and drops its
        RAM tier, which may still hold them. `clear` needs no bump when the
        whole directory goes, since the stamp goes with it."""
        self.stamp.bump()

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        self._ensure_initialized()
        if self._unusable:
            return 0
        expired = [e.key for e in self.entries() if e.key and is_expired(e.metadata)]
        for key in expired:
            self.delete(key)
        return len(expired)
