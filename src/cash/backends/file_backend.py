"""File-based cache backend with LRU eviction and optional compression."""

from __future__ import annotations

import glob
import gzip
import logging
import os
import pickle
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

from cash.exceptions import CacheBackendError
from cash.utils import replace_with_retry

from ._base import CacheBackend, MetadataDict, PendingWrites
from .entry_format import (
    ENTRY_SUFFIX,
    pack_entry,
    read_entry,
    update_metadata_in_place,
)
from .serialization import PickleSerializer, Serializer

logger = logging.getLogger(__name__)


# Every live write queue, grouped by the cache directory it writes into.
#
# A cache directory IS the cache; two FileBackend instances pointing at one
# directory are two views of the same store, not two stores. But each carries
# its OWN PendingWrites, and ``get()`` only ever waited on its own queue -- so a
# second instance reading a directory the first was still writing saw a
# half-populated cache and reported clean misses. In the regression that
# exposed this, only 2-3 of 19 entries had reached disk when the second
# instance started reading.
#
# WeakSet: a backend that goes out of scope must not keep its queue (or itself)
# alive, and must stop being waited on.
_WRITERS_BY_DIR: dict[str, weakref.WeakSet] = {}
_WRITERS_LOCK = threading.Lock()


def _writer_scope(cache_dir: str) -> str:
    """Normalized identity of a cache directory.

    ``realpath`` so ``/var/...`` and ``/private/var/...`` (the macOS symlink)
    or a relative path and its absolute form resolve to one scope.
    """
    try:
        return os.path.realpath(cache_dir)
    except OSError:
        return os.path.abspath(cache_dir)


def _register_writer(cache_dir: str, writes: PendingWrites) -> None:
    scope = _writer_scope(cache_dir)
    with _WRITERS_LOCK:
        bucket = _WRITERS_BY_DIR.get(scope)
        if bucket is None:
            bucket = weakref.WeakSet()
            _WRITERS_BY_DIR[scope] = bucket
        bucket.add(writes)
        # Opportunistically drop scopes whose backends have all been collected,
        # so a long test session doesn't accumulate an entry per temp dir.
        if len(_WRITERS_BY_DIR) > 64:
            for dead in [s for s, b in _WRITERS_BY_DIR.items() if not b]:
                del _WRITERS_BY_DIR[dead]


def _sibling_writers(cache_dir: str, own: PendingWrites) -> list[PendingWrites]:
    """Live write queues over *cache_dir* other than *own*."""
    with _WRITERS_LOCK:
        bucket = _WRITERS_BY_DIR.get(_writer_scope(cache_dir))
        return [w for w in bucket if w is not own] if bucket else []

__all__ = ["FileBackend", "CACHE_FORMAT_VERSION"]

# Live entries. One file each; see ``entry_format``.
_ENTRY_GLOB = f"*{ENTRY_SUFFIX}"
# Everything a format migration has to sweep up, including the
# ``.meta``/``.data`` pair that entries were stored as before v2.
_ALL_ENTRY_GLOBS = (_ENTRY_GLOB, "*.meta", "*.data")

# Version of the on-disk cache format (the ``*.entry`` layout described in
# ``entry_format``; v1 was a ``*.meta`` / ``*.data`` pair). Bump this **whenever a change makes caches written by an older
# build undecodable or liable to be misread** by a newer one. On init,
# FileBackend compares this against the stamp it finds in the cache dir and
# auto-invalidates a mismatched cache rather than silently decoding a stale
# layout — automating the "run %cash_repair --full after upgrading" step.
CACHE_FORMAT_VERSION = 2

# Filename of the per-directory format stamp. Has no entry extension so it
# is invisible to entry globs (listing, sizing, clearing).
_VERSION_FILENAME = "CACHE_VERSION"


#: What reading a cache entry's metadata can raise, and why every one of them
#: means the same thing: this entry is not readable HERE, so treat it as absent.
#:
#: ``pickle.load`` fails for far more reasons than a corrupt file. The one that
#: bit in practice was ``ModuleNotFoundError`` -- an entry written by an
#: environment that had numpy, read by one that does not, because a metadata
#: field held a ``numpy.int64`` instead of a plain ``int``. A narrow
#: ``(OSError, pickle.PickleError)`` guard let that escape out of ``%cash_on``
#: and killed the user's cell.
#:
#: ``get()`` (the VALUE read path) already encodes this policy, listing
#: AttributeError/ImportError/EOFError with the comment "the entry is
#: unrestorable here - report it absent so callers recompute". The metadata
#: paths simply never learned it. This is that lesson, applied consistently.
#:
#: Deliberately ``Exception`` rather than a tuple of the known ones. These
#: sites SCAN every entry in the cache, so one poisoned file must never take
#: the process down whatever the cause -- and metadata can now legitimately
#: contain user objects (a callee's captured globals), whose ``__setstate__``
#: can raise anything at all. A skipped entry costs a recompute; an escaped
#: exception costs the session.
UNREADABLE_ENTRY = Exception


class FileBackend(CacheBackend):
    """File-based cache backend.

    .. warning:: Uses `pickle` for serialization.  Cache files are
       assumed to originate from the local machine.  Loading a cache
       deserializes pickled objects, which runs arbitrary code, so never
       point this at a cache directory from an untrusted source.  See the
       Security section of the Backends documentation.
    """
    source_label: str = "DISK"

    def __init__(self, cache_dir: str, compress: bool = False, max_size_bytes: int | None = None, flush_interval: int = 5, default_ttl: int | None = None) -> None:
        """
        Args:
            cache_dir: Directory for cache files.
            compress: Whether to gzip-compress data files.
            max_size_bytes: Maximum total cache size in bytes (triggers LRU eviction).
            flush_interval: Seconds between metadata flush cycles.
            default_ttl: Default time-to-live in seconds for cache entries. None = no expiration.
        """
        # Resolve to absolute path at init time so os.chdir() won't break file paths
        self.cache_dir = os.path.abspath(cache_dir)
        self.compress = compress
        self._max_size_bytes = max_size_bytes
        self._default_ttl = default_ttl
        self._current_size_bytes = 0
        # Evict-after-write detection: a monotonic write counter and
        # the seq at which each key was last written, so eviction can tell when
        # it is discarding something written only a couple of ops ago — the
        # signature of a cache too small to retain the working set. Warned
        # about once per session (per instance = per Cash session).
        self._write_seq = 0
        self._write_seq_by_key: dict[str, int] = {}
        self._warned_evict_after_write = False
        self._dirty_metadata: set[str] = set()
        self._metadata_cache: dict[str, dict] = {}  # partial cache for LRU tracking
        # Whether _metadata_cache holds EVERY entry on disk. False after a
        # plain size scan; set once the eviction path has loaded them all.
        self._metadata_loaded = False
        # Whether _current_size_bytes is an absolute on-disk total (True) or
        # only this process's running delta (False). See _ensure_size_scanned.
        self._size_scanned = False
        self._lock = threading.RLock()
        self._flush_interval = flush_interval
        self._stop_event = threading.Event()

        # Per-backend async writes: serialization happens on the calling
        # thread, the actual disk I/O runs in this executor so a slow
        # write doesn't block cell execution.
        self._writes = PendingWrites()
        _register_writer(self.cache_dir, self._writes)

        # Lazy initialization: defer directory creation, stat scanning,
        # and background thread to first actual use.
        self._initialized = False
        self._init_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Lazily create cache directory, check the format stamp, start the flusher.

        Thread-safe via double-checked locking so this only happens once, on
        the first real cache operation.

        Deliberately O(1) in the number of entries. This runs on the CALLER's
        thread -- the first ``get`` of the session waits for it -- so nothing
        that walks the directory belongs here. The byte total that used to be
        computed at this point is now deferred to ``_ensure_size_scanned``.
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            os.makedirs(self.cache_dir, exist_ok=True)
            self._check_format_version()
            if self._flush_interval > 0:
                self._flusher_thread = threading.Thread(
                    target=self._flush_periodically, daemon=True,
                )
                self._flusher_thread.start()
            self._initialized = True

    def _check_format_version(self) -> None:
        """Enforce on-disk cache-format compatibility.

        Compares the format stamp in the cache directory against
        :data:`CACHE_FORMAT_VERSION`. If they differ — including a cache
        written by a pre-stamp build (no marker) that still holds entries —
        the existing entry files are removed so a stale layout can never be
        decoded as if it were current. The marker is then
        (re)written with the current version.

        A fresh or empty directory is simply stamped: no entries means there
        is nothing incompatible to clear. Runs under ``_init_lock`` (held by
        the caller) and never calls back into ``clear()`` to avoid recursing
        through ``_ensure_initialized``.
        """
        version_path = os.path.join(self.cache_dir, _VERSION_FILENAME)

        stored: int | None = None
        if os.path.exists(version_path):
            try:
                with open(version_path, encoding="utf-8") as fh:
                    stored = int(fh.read().strip())
            except (OSError, ValueError):
                stored = None  # unreadable/corrupt marker → treat as mismatch

        if stored != CACHE_FORMAT_VERSION:
            entry_files = [
                f
                for pattern in _ALL_ENTRY_GLOBS
                for f in glob.glob(os.path.join(self.cache_dir, pattern))
            ]
            if entry_files:
                logger.warning(
                    "Cash cache at %s was written in format v%s but this build "
                    "expects v%s; clearing %d stale file(s). Cache formats are not "
                    "compatible across this change.",
                    self.cache_dir,
                    "<unstamped>" if stored is None else stored,
                    CACHE_FORMAT_VERSION,
                    len(entry_files),
                )
                for f in entry_files:
                    try:
                        os.remove(f)
                    except OSError:
                        logger.debug(
                            "Could not remove stale cache file %s during format "
                            "migration", f, exc_info=True,
                        )
            try:
                with open(version_path, "w", encoding="utf-8") as fh:
                    fh.write(str(CACHE_FORMAT_VERSION))
            except OSError:
                logger.debug(
                    "Could not write cache format marker at %s", version_path,
                    exc_info=True,
                )

    def _scan_size_bytes(self) -> int:
        """Total the directory's bytes with ``scandir`` + ``stat``.

        This is the one operation whose cost still scales with the number of
        entries: about 3.5us each, so ~70ms at 20k, ~0.35s at 100k and ~3.5s
        at 1M. It reads no file *contents* -- an earlier version unpickled
        every entry's metadata here, which was 30x slower again (2037ms vs
        65ms at 20k) and built a metadata cache nothing had asked for.
        """
        total_size = 0
        try:
            with os.scandir(self.cache_dir) as entries:
                for entry in entries:
                    if not entry.name.endswith(ENTRY_SUFFIX):
                        continue
                    try:
                        total_size += entry.stat().st_size
                    except OSError:
                        # Vanished between the listing and the stat, or
                        # unreadable. One entry missing from the total is a
                        # slightly-late eviction, not a wrong answer.
                        continue
        except OSError:
            logger.debug("Could not scan %s for size", self.cache_dir, exc_info=True)
        return total_size

    def _ensure_size_scanned(self) -> None:
        """Establish the on-disk byte total, once, for the size cap.

        ``_check_and_evict`` is the only thing that ever reads that total, so
        only a process that WRITES needs it. It used to be computed at open
        time, by every process, on the calling thread of its first cache
        operation -- so a read-only run (the kernel-restart replay that cash
        exists to make fast) paid a full directory walk for a number it never
        consulted: ~0.35s at 100k entries, ~3.5s at 1M, before the first cell
        could return.

        Deferring it here moves that cost twice over: it is paid only when
        something is actually written, and it is paid on the ``PendingWrites``
        worker rather than the caller, so no cell blocks on it either way.

        The walk runs OUTSIDE ``_lock`` -- holding that across seconds of stat
        calls would stall every concurrent ``get``. The result therefore
        supersedes the running delta rather than adding to it: by the time
        ``_check_and_evict`` calls this the write that triggered it is already
        on disk, so the directory is the truth and whatever this process had
        counted so far is a subset of it. A ``delete`` landing mid-walk may be
        counted as still present, leaving the total high by that entry until
        the next one -- the same slightly-early-eviction direction the
        per-entry ``stat`` failure above already accepts.
        """
        if self._size_scanned:
            return
        scanned = self._scan_size_bytes()
        with self._lock:
            if self._size_scanned:
                return
            self._current_size_bytes = scanned
            self._size_scanned = True

    def _ensure_metadata_loaded(self) -> None:
        """Populate ``_metadata_cache`` from disk, once, for LRU ordering.

        Only the eviction path needs to see entries this process never
        touched: it has to rank ALL of them by last access to pick a victim.
        Everything else reads metadata per key, on demand.
        """
        if self._metadata_loaded:
            return
        for path in glob.glob(os.path.join(self.cache_dir, _ENTRY_GLOB)):
            try:
                meta, _ = read_entry(path, with_payload=False)
            except UNREADABLE_ENTRY:
                logger.debug("Skipping unreadable entry %s", path, exc_info=True)
                continue
            key = meta.get('key')
            if key and key not in self._metadata_cache:
                # Never overwrite a live entry: this process's copy is newer
                # than whatever is on disk for a key it has already written.
                self._metadata_cache[key] = meta
        self._metadata_loaded = True

    def _flush_periodically(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._flush_interval):
                break
            self._flush_metadata()

    def _flush_metadata(self) -> None:
        """Write dirty metadata back into each entry, in place.

        Every ``get`` bumps ``last_access`` and ``access_count`` and marks the
        key dirty, so this runs over recently-read entries every
        ``flush_interval`` seconds. With metadata and payload sharing a file,
        rewriting the whole entry to record an access would mean rewriting the
        payload: a session that reads a 100MB frame would rewrite 100MB every
        few seconds. ``update_metadata_in_place`` writes only the header and
        the metadata region, which is what the reserved slack is for.

        If metadata has outgrown that slack the update is skipped rather than
        forced. What is lost is LRU precision for one entry until it is next
        written -- never a value, and never correctness.
        """
        with self._lock:
            if not self._dirty_metadata:
                return

            keys_to_flush = list(self._dirty_metadata)
            self._dirty_metadata.clear()

        for key in keys_to_flush:
            try:
                meta = self._metadata_cache.get(key)
                if meta and not update_metadata_in_place(self._get_path(key), meta):
                    logger.debug(
                        "Metadata for %r no longer fits its reserved region; "
                        "access stats not flushed", key,
                    )
            except (OSError, pickle.PickleError) as exc:
                logger.debug("Failed to flush metadata for key %r: %s", key, exc)

    def _get_path(self, key: str) -> str:
        import hashlib
        safe_name = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_name}{ENTRY_SUFFIX}")
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
        # Wait for any in-flight write so the metadata we report reflects
        # the most recent ``set()`` for this key.
        self._writes.wait(key)
        cached_meta = self._metadata_cache.get(key)

        path = self._get_path(key)

        try:
            if cached_meta is not None:
                # Cheaper than reopening, but it still has to exist: a delete
                # by another process must read as absent, not as this
                # process's stale copy.
                if not os.path.exists(path):
                    return None
                metadata = cached_meta
            else:
                metadata, _ = read_entry(path, with_payload=False)
                self._metadata_cache[key] = metadata

            ttl = metadata.get('ttl', self._default_ttl)
            if ttl is not None:
                created_at = metadata.get('created_at', 0)
                if time.time() - created_at > ttl:
                    return None

            return metadata
        except FileNotFoundError:
            return None
        except UNREADABLE_ENTRY:
            logger.debug("Unreadable metadata for key %s; treating as absent", key, exc_info=True)
            return None
    def get(self, key: str) -> tuple[MetadataDict | None, Any | None]:
        self._ensure_initialized()
        # Wait for any in-flight write for this key so we never return
        # stale-or-missing data when get() races set().
        self._wait_for_writes(key)
        # Check memory cache first for metadata. The cached dict is preferred
        # over the one on disk even though we are about to read the file
        # anyway: callers hold it, `get` mutates it in place, and the flusher
        # writes THAT object back. Replacing it with a fresh dict per read
        # would drop every unflushed access update on the floor.
        cached_meta = self._metadata_cache.get(key)

        path = self._get_path(key)

        try:
            on_disk, payload = read_entry(path, with_payload=True)
        except FileNotFoundError:
            if key in self._metadata_cache:
                with self._lock:
                    self._metadata_cache.pop(key, None)
            return None, None
        except UNREADABLE_ENTRY as exc:
            logger.debug("Cache get failed for key %r: %s", key, exc)
            return None, None

        try:
            if cached_meta is not None:
                metadata = cached_meta
            else:
                metadata = on_disk
                self._metadata_cache[key] = metadata

            # A metadata-only entry (the value was too large to persist) has
            # no payload. It is a HIT for `get_metadata`, which wants the
            # execution time and lineages, and a MISS here, because there is
            # nothing to restore. With two files that fell out of requiring
            # both to exist; with one it has to be asked explicitly.
            if metadata.get('metadata_only'):
                return None, None

            ttl = metadata.get('ttl', self._default_ttl)
            if ttl is not None:
                created_at = metadata.get('created_at', 0)
                if time.time() - created_at > ttl:
                    # Entry expired - delete it
                    self.delete(key)
                    return None, None

            # Update Access Time (Async)
            metadata['last_access'] = time.time()
            metadata['access_count'] = metadata.get('access_count', 0) + 1

            with self._lock:
                self._dirty_metadata.add(key)

            if metadata.get('compressed', False):
                try:
                    payload = gzip.decompress(payload)
                except (OSError, gzip.BadGzipFile, EOFError):
                    # Flag says compressed but the bytes are not. Fall through
                    # with the raw bytes, as the two-file path did.
                    logger.debug("Entry for %r flagged compressed but is not", key)

            serializer_cls = metadata.get('serializer_cls', PickleSerializer)
            value = serializer_cls().deserialize(payload)

            metadata.setdefault('source', self.source_label)
            return metadata, value
        except (OSError, pickle.PickleError, ValueError, AttributeError,
                ImportError, EOFError) as exc:
            # AttributeError/ImportError: the pickled value references a
            # binding that doesn't exist in this process (e.g. a __main__
            # class from a previous kernel session). The entry is
            # unrestorable here - report it absent so callers recompute
            # instead of crashing the user's cell.
            #
            # EOFError: a truncated file. ``_atomic_write`` now prevents cash
            # from producing one, but a cache directory can still hold a
            # partial file written by an older version, a killed process, or a
            # full disk. EOFError subclasses Exception directly - not OSError,
            # not ValueError - so it used to escape this handler and crash the
            # caller with "Ran out of input" instead of degrading to a miss.
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
        for sibling in _sibling_writers(self.cache_dir, self._writes):
            try:
                sibling.wait(key)
            except Exception:  # noqa: BLE001 — see docstring
                logger.debug("Sibling write for key %r failed", key, exc_info=True)

    @staticmethod
    def _replace_with_retry(tmp_path: str, path: str) -> None:
        """Delegates to :func:`cash.utils.replace_with_retry`.

        Kept as a method because this backend's call site reads better for it,
        and because the shared helper now also serves
        ``notebook.loop_split.LoopSplitStore`` -- which had the same
        tmp-then-replace shape and lost a verdict silently when the
        destination was held open.
        """
        replace_with_retry(tmp_path, path)

    def _atomic_write(self, path: str, payload: bytes) -> None:
        """Write *payload* to *path* so no reader can observe a partial file.

        A plain ``open(path, 'wb')`` makes the file visible the instant it is
        created — while it still holds zero or half its bytes. Any concurrent
        reader (a second Cash instance, or another process sharing the cache
        directory) can pass ``get()``'s existence check and then unpickle a
        truncated file. That is what surfaced on the macOS CI runners as
        ``EOFError: Ran out of input``.

        Writing to a temp file in the SAME directory and renaming makes the
        entry appear all at once: a reader sees either the previous contents or
        the complete new ones, never a torn mixture. ``os.replace`` is atomic on
        POSIX and on Windows, and same-directory keeps it a rename rather than a
        cross-filesystem copy.

        Takes no compression flag. Compression applies to the payload region
        of an entry, not to the file: the header and metadata must stay
        readable without inflating anything, or a metadata read would have to
        decompress the value it exists to avoid touching.
        """
        directory = os.path.dirname(path) or '.'
        # mkstemp in the target directory; the leading dot keeps the partial
        # out of the ``*.entry`` glob the backend scans.
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix='.tmp-', suffix='.part')
        os.close(fd)
        try:
            with open(tmp_path, 'wb') as f:
                f.write(payload)
            self._replace_with_retry(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Could not remove partial write %s", tmp_path, exc_info=True)
            raise

    def _write_cache_files(self, key: str, path: str, metadata: dict,
                           serialized_value: bytes) -> None:
        """Write one entry -- metadata and payload -- and update size tracking.

        One file, one atomic rename. The two-file version had to write the
        data first and the metadata second, and reason about a reader that
        caught the gap; there is no gap to reason about now.

        Raises:
            OSError, pickle.PickleError, ValueError: on write failure (caller handles cleanup).
        """
        payload = gzip.compress(serialized_value) if self.compress else serialized_value
        # ``size`` means the bytes the value occupies, post-compression --
        # what the old code learned by stat-ing the data file it had just
        # written. We already know it, so the stat is gone.
        metadata['size'] = len(payload)
        blob = pack_entry(metadata, payload)

        # Exact, and one syscall: what this entry costs today, so a rewrite
        # subtracts what was really there. The old code subtracted the cached
        # ``size`` plus the NEW metadata size as a proxy, and subtracted
        # nothing at all for a key this process had never written.
        try:
            old_entry_bytes = os.path.getsize(path)
        except OSError:
            old_entry_bytes = 0

        try:
            self._atomic_write(path, blob)
        except FileNotFoundError:
            # The cache dir vanished under a live process. The README suggests
            # deleting ./.cash to wipe the cache, and doing that with the kernel
            # still running used to fail the next write with CacheBackendError
            # instead of simply recreating the directory. Recreate + retry once;
            # costs nothing on the normal path.
            os.makedirs(self.cache_dir, exist_ok=True)
            self._atomic_write(path, blob)

        with self._lock:
            self._current_size_bytes -= old_entry_bytes
            self._metadata_cache[key] = metadata
            self._current_size_bytes += len(blob)

            # Stamp the write order so _check_and_evict can spot a just-written
            # entry being evicted almost immediately (the treadmill signature).
            self._write_seq += 1
            self._write_seq_by_key[key] = self._write_seq

    def set(self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None) -> None:
        """Serialize the value on the calling thread, then write to disk
        in the background. ``set()`` returns once the bytes are captured;
        a subsequent ``get(key)`` waits for the write."""
        self._ensure_initialized()
        path = self._get_path(key)

        metadata = self._init_metadata(metadata, key)

        # Set TTL if not already specified and we have a default
        if 'ttl' not in metadata and self._default_ttl is not None:
            metadata['ttl'] = self._default_ttl

        if serializer is None:
            serializer = PickleSerializer()

        # IMPORTANT: serialize on the calling thread so a post-set()
        # mutation of `value` can't corrupt the cached bytes.
        serialized_value = serializer.serialize(value)

        metadata['compressed'] = self.compress
        # Pre-compute size from the serialized bytes; the on-disk size
        # may differ slightly under compression but the user-facing
        # metadata needs to be populated synchronously for the badge.
        metadata['size'] = len(serialized_value)
        if 'storage' not in metadata:
            metadata['storage'] = [self.source_label]

        # Freeze a copy of metadata for the background write — the caller
        # can mutate the original after we return without affecting the
        # written entry.
        meta_for_write = dict(metadata)
        self._writes.submit(
            key, self._do_set_sync,
            key, path, meta_for_write, serialized_value,
        )

    def _do_set_sync(self, key: str, path: str, metadata: dict,
                     serialized_value: bytes) -> None:
        """The actual disk write — runs in the PendingWrites worker thread.

        Failures clean up any partial file and re-raise. The exception
        is stored on the future and surfaces on the next ``get(key)``
        (or ``shutdown()``).
        """
        try:
            self._write_cache_files(key, path, metadata, serialized_value)
            if self._max_size_bytes:
                self._check_and_evict()
        except (OSError, pickle.PickleError, ValueError) as exc:
            logger.debug("Cache set failed for key %r, cleaning up: %s", key, exc)
            try:
                os.remove(path)
            except OSError:
                logger.debug("Cleanup of partial cache file failed for key %r", key, exc_info=True)
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
        # Wait for any in-flight async set() for this key to land first —
        # otherwise the check below races a not-yet-flushed full write, sees
        # nothing, and clobbers the (more valuable) full entry with a
        # metadata-only one. get()/get_metadata()/delete() synchronise the
        # same way.
        self._writes.wait(key)
        path = self._get_path(key)

        try:
            existing, _ = read_entry(path, with_payload=False)
        except UNREADABLE_ENTRY:
            existing = None
        if existing is not None and not existing.get('metadata_only'):
            return

        metadata = dict(metadata)  # Don't mutate caller's dict
        metadata['key'] = key
        metadata['metadata_only'] = True
        metadata.setdefault('created_at', time.time())
        metadata.setdefault('last_access', time.time())
        metadata.setdefault('access_count', 0)
        metadata.setdefault('size', 0)

        try:
            self._atomic_write(path, pack_entry(metadata, b""))
            with self._lock:
                self._metadata_cache[key] = metadata
        except OSError as exc:
            logger.debug("Failed to write metadata-only entry for key %r: %s", key, exc)

    def delete(self, key: str) -> None:
        self._ensure_initialized()
        # Drain any pending write for this key — otherwise the write
        # could fire after the delete and leave a ghost entry.
        self._writes.drain(key)
        path = self._get_path(key)

        # Measured from the file rather than taken from the metadata cache.
        # The cached figure is only present for keys THIS process has touched,
        # so with metadata loaded lazily an uncached key would have subtracted
        # 0 and drifted the total upward forever.
        try:
            size_to_remove = os.path.getsize(path)
        except OSError:
            size_to_remove = 0

        with self._lock:
            if key in self._metadata_cache:
                del self._metadata_cache[key]
            if key in self._dirty_metadata:
                self._dirty_metadata.remove(key)
            self._write_seq_by_key.pop(key, None)
            self._current_size_bytes -= size_to_remove

        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Failed to remove cache entry %s: %s", path, exc)

    # Evict-after-write is only a treadmill signal if the evicted entry was
    # written within roughly this many writes — something older getting
    # evicted is healthy LRU, not thrash. Only writes bump the counter.
    _EVICT_WARN_RECENT_OPS = 3

    def _check_and_evict(self) -> None:
        """Evict items if over max size."""
        if not self._max_size_bytes:
            return

        # The byte total is established HERE, not at open time: this is the
        # only thing that reads it, so a process that never writes must never
        # pay the directory walk that produces it. Latched, so a write-heavy
        # session pays one walk, not one per write.
        self._ensure_size_scanned()

        if self._current_size_bytes <= self._max_size_bytes:
            return

        # Ranking by last access needs every entry, including the ones this
        # process never touched -- otherwise a fresh process would evict only
        # from the handful of keys it happens to have read, and free almost
        # nothing. Loaded here rather than at startup so only the processes
        # that actually evict pay for it.
        self._ensure_metadata_loaded()

        keys_to_delete = []
        evicted_recent = False
        with self._lock:
            # Oldest access first. Only the ORDER is decided here; how many to
            # take is decided as they go, against the running byte total.
            order = sorted(
                (m.get('last_access', 0), k)
                for k, m in self._metadata_cache.items()
            )

        target = self._max_size_bytes * 0.9
        for _last_access, key in order:
            if self._current_size_bytes <= target:
                break
            # Never evict a key that has a write in flight.
            #
            # `delete` drains that write, and this loop runs ON the single
            # PendingWrites worker -- so draining a write QUEUED BEHIND the
            # one currently executing waits for a task that cannot start until
            # this one returns. The write thread hangs permanently, and the
            # atexit flush behind it hangs with it.
            #
            # Reachable without contriving anything: a key's cached
            # `last_access` is still its PREVIOUS write's, because the queued
            # write has not updated it yet, so re-computing a long-idle entry
            # while the cache sits near its cap makes that entry the obvious
            # LRU victim. Reproduced against a 6KB cap; the process never
            # exited.
            #
            # Skipping is also right on the merits. An entry being written
            # right now is the newest thing in the cache, not the oldest, and
            # the next write re-checks the cap anyway.
            if self._writes.has_pending(key):
                continue
            # Was this entry written only a couple of ops ago?
            written_at = self._write_seq_by_key.get(key)
            if written_at is not None and self._write_seq - written_at <= self._EVICT_WARN_RECENT_OPS:
                evicted_recent = True
            # `delete` subtracts what the file actually weighed, so the loop
            # above stops as soon as enough has really been freed.
            #
            # It used to accumulate each victim's ``size`` instead, which is
            # the PAYLOAD length -- while the total it was working down from
            # counted whole entries, header and metadata included. Every
            # eviction therefore looked smaller than it was, and the loop took
            # more entries than it needed. With two files per entry the gap
            # was small enough to hide; it surfaced as soon as an entry's
            # fixed overhead grew.
            self.delete(key)
            keys_to_delete.append(key)

        if evicted_recent:
            self._warn_evict_after_write(len(keys_to_delete))

    def _promotion_size_cap(self) -> int | None:
        """Refuse (skip) any single object larger than half this tier's cap.

        The file tier's LRU cap is adaptive (a fraction of free disk,
), so its per-object refusal threshold is derived from the
        instance cap rather than a static class attr. Storing something bigger
        than half the cap would leave under half the cap for everything else
        and invite a write-and-evict treadmill; a clean skip (the value stays
        in RAM, and ``TieredBackend`` warns) beats the thrash. Falls back to
        the class-level hint when no cap is configured (a bare, unbounded
        FileBackend accepts anything).
        """
        if self._max_size_bytes:
            return self._max_size_bytes // 2
        return type(self).max_size_bytes

    def _warn_evict_after_write(self, n_evicted: int) -> None:
        """Warn once/session that the cache evicted freshly-written entries.

        The disk cache is too small to retain what is being written to it, so
        entries are evicted within a couple of ops of landing — cash keeps
        re-writing and re-evicting instead of caching anything durably, which
        can make it slower than not caching at all. Deduped to once per
        session (per instance) so a churning workload doesn't spam.
        """
        if self._warned_evict_after_write:
            return
        self._warned_evict_after_write = True
        import warnings

        from cash.exceptions import CashCacheIneffectiveWarning
        from .adaptive_caps import human_bytes
        warnings.warn(
            f"Cash: the disk cache evicted an entry within a couple of writes "
            f"of storing it. The cache cap ({human_bytes(self._max_size_bytes)}) "
            f"is too small to retain this workload, so cash is re-writing and "
            f"re-evicting instead of caching durably -- this can be slower than "
            f"no cache. Raise max_cache_size so the working set fits.",
            CashCacheIneffectiveWarning,
            stacklevel=2,
        )

    def clear(self) -> None:
        self._ensure_initialized()
        # Drain pending writes so they don't fire after the clear and
        # resurrect entries we just removed from disk.
        self._writes.wait_all()
        for pattern in _ALL_ENTRY_GLOBS:
            for f in glob.glob(os.path.join(self.cache_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    # Best-effort removal during cache clear; file may be locked
                    logger.debug("Could not remove cache file %s during clear", f, exc_info=True)
        with self._lock:
            self._metadata_cache.clear()
            self._dirty_metadata.clear()
            self._write_seq_by_key.clear()
            self._current_size_bytes = 0

    def shutdown(self) -> None:
        # Drain any in-flight async writes before stopping the flusher,
        # otherwise we could lose the metadata for a not-yet-written entry.
        self._writes.shutdown(wait=True)
        self._stop_event.set()
        if hasattr(self, '_flusher_thread'):
            self._flusher_thread.join(timeout=1.0)
        if self._initialized:
            self._flush_metadata()

    def list_entries(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        # Drain pending writes so the listing reflects everything the
        # caller has already set() — otherwise async writes still in
        # flight would be invisible.
        self._writes.wait_all()
        entries = []
        for path in glob.glob(os.path.join(self.cache_dir, _ENTRY_GLOB)):
            try:
                metadata, _ = read_entry(path, with_payload=False)
                entries.append(metadata)
            except UNREADABLE_ENTRY:
                logger.debug("Skipping unreadable entry %s in list_entries", path, exc_info=True)
        return entries

    def cleanup_expired(self, is_expired: Callable[[dict[str, Any]], bool]) -> int:
        self._ensure_initialized()
        self._writes.wait_all()
        count = 0
        for path in glob.glob(os.path.join(self.cache_dir, _ENTRY_GLOB)):
            try:
                metadata, _ = read_entry(path, with_payload=False)
                if is_expired(metadata):
                    os.remove(path)
                    count += 1
            except UNREADABLE_ENTRY:
                logger.debug("Skipping unreadable entry %s during cleanup", path, exc_info=True)
        return count
