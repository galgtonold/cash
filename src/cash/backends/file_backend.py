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
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any, NamedTuple

from cash._paths import replace_with_retry
from cash.exceptions import CacheBackendError

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheStoreFailedWarning
from ..tracking.file_tracker import register_cache_dir, untracked
from ._base import CacheBackend, MetadataDict, ttl_expired
from ._writes import PendingWrites
from .cache_dir import CacheDirStamp, create_temp_file, warn_if_unwritable, write_all
from .entry_format import ENTRY_SUFFIX, CorruptEntry, metadata_span, pack_entry, read_entry, update_metadata_in_place
from .file_eviction import FileEvictor
from .serialization import PickleSerializer, Serializer
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


def _writer_scope(cache_dir: str) -> str:
    """Normalized identity of a cache directory (symlinks and relative paths resolved)."""
    try:
        return os.path.realpath(cache_dir)
    except OSError:
        return os.path.abspath(cache_dir)


def _register_writer(cache_dir: str, writes: PendingWrites) -> str:
    """Register *writes* under *cache_dir*'s scope, and return the scope."""
    scope = _writer_scope(cache_dir)
    # Tell the file tracker this directory is cash's own storage, so its entry
    # files never become dependencies of the user's code whatever it is called.
    try:
        register_cache_dir(cache_dir)
    except Exception:  # noqa: BLE001 - tracking is best-effort, storage is not
        logger.debug("Could not register %s with the file tracker", cache_dir, exc_info=True)
    with _WRITERS_LOCK:
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

    def __init__(
        self,
        cache_dir: str,
        compress: bool = False,
        max_size_bytes: int | None = None,
        flush_interval: int = 5,
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
        #: Keys read since their metadata was last written back.
        self._dirty_metadata: set[str] = set()
        #: key -> when its access stamp was last written, for the flusher's rate limit.
        self._access_flushed: dict[str, float] = {}
        #: Metadata for the keys this process has touched, and the way back from
        #: an entry's path to its key (a filename is a SHA-256 of the key).
        self._metadata_cache: dict[str, dict] = {}
        self._paths: dict[str, str] = {}
        #: Each statement's versions, pruned as they are written (versions.py).
        #: A key this process has read is in use and never pruned.
        self._versions = VersionIndex(self.cache_dir, _untracked)
        self._read_keys: set[str] = set()
        self._lock = threading.RLock()
        self._flush_interval = flush_interval
        self._stop_event = threading.Event()

        # Values are serialized on the caller's thread; the disk I/O runs here.
        self._writes = PendingWrites()
        self._writer_scope = _register_writer(self.cache_dir, self._writes)
        self.stamp = CacheDirStamp(self.cache_dir, _untracked)
        self.evictor = FileEvictor(
            self.cache_dir,
            max_size_bytes,
            adaptive_cap,
            metadata=self._metadata_cache,
            paths=self._paths,
            dirty=self._dirty_metadata,
            writes=self._writes,
            lock=self._lock,
            untracked=_untracked,
        )

        # Directory creation, the stamp check and the flusher wait for first use.
        self._initialized = False
        self._init_lock = threading.Lock()
        #: Set when the cache directory turned out to be unusable. Every public
        #: operation then answers as an empty cache would: a miss, a no-op write.
        self._unusable = False

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

        try:
            warn_diagnostic(
                CashCacheStoreFailedWarning,
                "CACHE-DIR-UNWRITABLE",
                f"cash cannot use its cache directory {self.cache_dir} "
                f"({type(exc).__name__}: {exc}). Nothing will be cached to disk "
                f"this run, so every call recomputes -- but the run itself "
                f"continues normally.",
                "point cash somewhere it can write -- cash.configure(cache_dir=...), "
                "CASH_CACHE_DIR, or the cache_dir= argument -- or grant this "
                "user write permission on that path.",
            )
        except Exception:  # noqa: BLE001 - a diagnostic must not become the failure
            logger.warning("Cash disabled its file tier at %s: %s", self.cache_dir, exc)

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
        with self._lock:
            if not self._dirty_metadata:
                return
            if periodic:
                keys_to_flush = [
                    k
                    for k in self._dirty_metadata
                    if now - self._access_flushed.get(k, 0.0) >= self._ACCESS_FLUSH_MIN_INTERVAL
                ]
                self._dirty_metadata.difference_update(keys_to_flush)
            else:
                keys_to_flush = list(self._dirty_metadata)
                self._dirty_metadata.clear()

        # A read raises an entry's priority, so a flushed access is also a
        # rank-index record; that is how an old entry still in use gets ranked.
        ranked = self.evictor.capped
        if ranked and keys_to_flush:
            self.evictor.ensure_clock()
        records: list[tuple[str, float]] = []
        for key in keys_to_flush:
            try:
                meta = self._metadata_cache.get(key)
                path = self._get_path(key)
                if meta and not update_metadata_in_place(path, meta):
                    logger.debug(
                        "Metadata for %r no longer fits its reserved region; access stats not flushed",
                        key,
                    )
                self._access_flushed[key] = now
                if meta and meta.get("version_slot"):
                    self._versions.touch(key, meta.get("last_access", now))
                if meta and ranked:
                    records.append(self.evictor.access_record(key, meta, path))
            except (OSError, pickle.PickleError) as exc:
                logger.debug("Failed to flush metadata for key %r: %s", key, exc)
        self.evictor.rank_index.append(records)

    def _remember(self, key: str, metadata: dict) -> None:
        """Cache one entry's metadata, and the way back from its filename.

        Eviction ranks entries by walking the directory, which yields paths.
        A filename is a SHA-256 of the key, so nothing recovers the key from
        it -- this map is how an evicted path finds the in-process bookkeeping
        that belongs to it. It only ever holds the keys this process has
        touched, which is exactly the set that HAS any bookkeeping.
        """
        with self._lock:
            self._metadata_cache[key] = metadata
            self._paths[self._get_path(key)] = key

    def _get_path(self, key: str) -> str:
        safe_name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_name}{ENTRY_SUFFIX}")

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
                self._remember(key, metadata)

            if ttl_expired(metadata.get("created_at", 0), metadata.get("ttl", self._default_ttl)):
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
        except (OSError, CorruptEntry) as exc:
            logger.debug("Cache get failed for key %r: %s", key, exc)
            return None, None

        try:
            if cached_meta is not None:
                metadata = cached_meta
            else:
                metadata = on_disk
                self._remember(key, metadata)

            # A metadata-only entry (the value was too large to persist) has
            # no payload. It is a HIT for `get_metadata`, which wants the
            # execution time and lineages, and a MISS here, because there is
            # nothing to restore. With two files that fell out of requiring
            # both to exist; with one it has to be asked explicitly.
            if metadata.get("metadata_only"):
                return None, None

            if ttl_expired(metadata.get("created_at", 0), metadata.get("ttl", self._default_ttl)):
                self.delete(key)
                return None, None

            # Update Access Time (Async)
            metadata["last_access"] = time.time()
            metadata["access_count"] = metadata.get("access_count", 0) + 1

            with self._lock:
                self._dirty_metadata.add(key)
                self._read_keys.add(key)
            self.evictor.note_read(key)

            if metadata.get("compressed", False):
                try:
                    payload = gzip.decompress(payload)
                except (OSError, gzip.BadGzipFile, EOFError):
                    # Flag says compressed but the bytes are not. Fall through
                    # with the raw bytes, as the two-file path did.
                    logger.debug("Entry for %r flagged compressed but is not", key)

            serializer_cls = metadata.get("serializer_cls", PickleSerializer)
            value = serializer_cls().deserialize(payload)

            metadata.setdefault("source", self.source_label)
            return metadata, value
        except (OSError, pickle.PickleError, ValueError, AttributeError, ImportError, EOFError) as exc:
            # Unrestorable here, so absent: AttributeError/ImportError for a
            # value naming a binding this process lacks (a __main__ class from
            # an earlier kernel), EOFError for a file truncated by a killed
            # process or a full disk.
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
            except Exception:  # noqa: BLE001 — see docstring
                logger.debug("Sibling write for key %r failed", key, exc_info=True)

    @staticmethod
    def _replace_with_retry(tmp_path: str, path: str) -> None:
        """`cash._paths.replace_with_retry`, as a method so tests can stub it."""
        replace_with_retry(tmp_path, path)

    def _atomic_write(self, path: str, payload: bytes) -> None:
        """Write *payload* to *path* so no reader can observe a partial file.

        A temp file in the same directory, renamed into place: a concurrent
        reader sees the previous contents or the complete new ones, never a
        truncated file. Compression, if any, is already in *payload*'s value
        region; the header and metadata stay readable without inflating it.
        """
        directory = os.path.dirname(path) or "."
        # A temp file in the target directory; the leading dot keeps the
        # partial out of the ``*.entry`` glob the backend scans.
        fd, tmp_path = create_temp_file(directory)
        try:
            # Through the descriptor it was created with: reopening the name
            # goes through the file tracker's patched ``open``, and on Windows
            # opening a file created a moment before is slow.
            try:
                write_all(fd, payload)
            finally:
                os.close(fd)
            self._replace_with_retry(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Could not remove partial write %s", tmp_path, exc_info=True)
            raise

    def _write_new_in_place(self, path: str, blob: bytes) -> bool:
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

        split = metadata_span(blob)
        try:
            # Raw fd writes: ``os.open`` raises the ``open`` audit event with
            # no mode, which neither the file tracker nor the effect observer
            # watches, so the entry never becomes a dependency or an effect of
            # the user's code whatever the directory is called. Unbuffered, the
            # payload and the header reach the page cache in program order.
            os.lseek(fd, split, os.SEEK_SET)
            write_all(fd, blob[split:])  # payload
            os.lseek(fd, 0, os.SEEK_SET)
            write_all(fd, blob[:split])  # header + metadata, last
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

    def _write_cache_files(self, key: str, path: str, metadata: dict, serialized_value: bytes) -> None:
        """Write one entry -- metadata and payload -- and update size tracking.

        Raises:
            OSError, pickle.PickleError, ValueError: on write failure (caller handles cleanup).
        """
        payload = gzip.compress(serialized_value) if self.compress else serialized_value
        # The bytes the value occupies on disk, after compression.
        metadata["size"] = len(payload)
        blob = pack_entry(metadata, payload)

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
            if not self._write_new_in_place(path, blob):
                self._atomic_write(path, blob)
        except FileNotFoundError:
            # The cache directory was deleted under a live process (``cash
            # clear --all``, or by hand): recreate it, stamped first so the next
            # process does not discard these entries as an unknown format, and
            # retry once.
            os.makedirs(self.cache_dir, exist_ok=True)
            self.stamp.ignore_in_git()
            self.stamp.write()
            if not self._write_new_in_place(path, blob):
                self._atomic_write(path, blob)

        with self._lock:
            self._remember(key, metadata)
            self.evictor.note_write(key, old_entry_bytes, len(blob))

        # Only a capped tier ranks, so only a capped tier keeps the rank index:
        # uncapped, it would be a file that only grows.
        if self.evictor.capped:
            self.evictor.record_rank(key, path, metadata, len(blob))

    def set(
        self, key: str, value: Any, metadata: MetadataDict | None = None, serializer: Serializer | None = None
    ) -> None:
        """Serialize the value on the calling thread, then write to disk
        in the background. ``set()`` returns once the bytes are captured;
        a subsequent ``get(key)`` waits for the write."""
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

        # IMPORTANT: serialize on the calling thread so a post-set()
        # mutation of `value` can't corrupt the cached bytes.
        serialized_value = serializer.serialize(value)

        metadata["compressed"] = self.compress
        # Pre-compute size from the serialized bytes; the on-disk size
        # may differ slightly under compression but the user-facing
        # metadata needs to be populated synchronously for the badge.
        metadata["size"] = len(serialized_value)
        if "storage" not in metadata:
            metadata["storage"] = [self.source_label]

        # Freeze a copy of metadata for the background write — the caller
        # can mutate the original after we return without affecting the
        # written entry.
        meta_for_write = dict(metadata)
        self._writes.submit(
            key,
            self._do_set_sync,
            key,
            path,
            meta_for_write,
            serialized_value,
        )
        slot = metadata.get("version_slot")
        if slot:
            # A version whose values are references to call entries weighs
            # what those hold too (``cash.notebook.call_refs``).
            self._prune_versions(
                slot,
                key,
                len(serialized_value) + int(metadata.get("call_ref_bytes") or 0),
                metadata.get("execution_time") or 0.0,
            )

    def _prune_versions(self, slot: str, key: str, size: int, cost: float) -> None:
        """Remove the superseded versions of *key*'s statement that are not
        worth their bytes (``versions.superseded_to_drop``). Best effort: a
        failure here leaves entries for the byte cap, never loses the new one."""
        try:
            versions = self._versions.record(slot, key, size, cost, time.time())
            gone = [k for k in versions if k != key and not os.path.exists(self._get_path(k))]
            for k in gone:
                versions.pop(k)
            drop = superseded_to_drop(versions, key, self._read_keys)
            freed = self._call_refs_of(drop) - self._call_refs_of(k for k in versions if k not in drop)
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

    def _call_refs_of(self, keys) -> set[str]:
        refs: set[str] = set()
        for k in keys:
            meta = self.get_metadata(k) or {}
            refs.update(meta.get("call_refs") or ())
        return refs

    def _do_set_sync(self, key: str, path: str, metadata: dict, serialized_value: bytes) -> None:
        """The actual disk write — runs in the PendingWrites worker thread.

        A failure re-raises and touches nothing: the destination is the
        previous entry, untouched because the rename never happened, or absent.
        The exception surfaces on the next ``get(key)`` (or ``shutdown()``).
        """
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
        self._writes.wait(key)
        path = self._get_path(key)

        # What this process last wrote there, when it was metadata only, needs
        # no disk read (slow on Windows, and a cheap statement rewrites its
        # entry on every run). If another process has put a full entry there
        # since, this overwrites it -- a later miss, never a wrong value.
        known = self._metadata_cache.get(key)
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
            blob = pack_entry(metadata, b"")
            # A new key takes the cheaper in-place write, as a full entry does.
            if existing is not None or not self._write_new_in_place(path, blob):
                self._atomic_write(path, blob)
            with self._lock:
                self._remember(key, metadata)
        except OSError as exc:
            logger.debug("Failed to write metadata-only entry for key %r: %s", key, exc)

    def delete(self, key: str) -> None:
        self._ensure_initialized()
        if self._unusable:
            return
        # Drain any pending write for this key — otherwise the write
        # could fire after the delete and leave a ghost entry.
        self._writes.drain(key)
        path = self._get_path(key)

        # Measured from the file: the metadata cache only knows the keys this
        # process touched.
        try:
            size_to_remove = os.path.getsize(path)
        except OSError:
            size_to_remove = 0

        with self._lock:
            self._metadata_cache.pop(key, None)
            self._dirty_metadata.discard(key)
            self._paths.pop(path, None)
            self.evictor.forget(key, size_to_remove)

        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Failed to remove cache entry %s: %s", path, exc)

    def promotion_size_cap(self) -> int | None:
        """Refuse (skip) only an object larger than this tier's WHOLE cap.

        "Keep at most N bytes" reads as: store what fits and evict the rest. A
        lower threshold (it was half the cap) refused values that fit
        comfortably, and a job whose working set was half its cap cached
        nothing. A write-and-evict treadmill is reported when it happens
        (``CACHE-THRASH``) rather than pre-empted. Uncapped, the class-level
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
        with self._lock:
            self._metadata_cache.clear()
            self._dirty_metadata.clear()
            self._paths.clear()

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
