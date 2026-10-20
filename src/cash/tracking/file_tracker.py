"""File-read tracking for data file dependencies.

`FileAccessTracker` records which files each cached call or notebook statement
reads. Python-level opens and directory listings reach it as audit events
(`cash.tracking.read_events`); readers that open files in C (pyarrow, polars,
sqlite3, some pandas readers), existence probes and ``Path.stat`` are wrapped
while a tracker is open (`cash.tracking.reader_patches`). Reads that are not
the user's data are left out (`cash.tracking.read_classification`), and each
read is also credited to the code that made it, for memos
(`cash.tracking.read_credit`). File hashes are incorporated into cache keys so
that changed data automatically invalidates dependent cached results.
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from collections.abc import Iterator
from typing import Any, Optional

from cash._clock import perf_counter as _perf_counter
from cash._paths import is_remote_url, normalize_path
from cash.effect_observer import reset_in_any_context
from cash.tracking import io_watch, native_reads
from cash.tracking.file_dep_snapshot import (
    digest_later,
    file_content_hash,
    realpath_of_read_this_run,
    remembered_file_digest,
)
from cash.tracking.pending_digest import MOVED, DeferredDigest, PendingDigest
from cash.tracking.read_classification import (
    SCRATCH_MEMMAP,
    folder_stat,
    incidental_read,
    is_cash_internal,
    is_pseudo_fs,
    is_runtime_cache_path,
    own_stat,
    regular_file_stat,
    regular_file_stat_result,
    stat_key,
)
from cash.tracking.read_credit import credit_read_to_stack
from cash.tracking.read_events import subscribe_read_events
from cash.tracking.reader_patches import install_patches, patch_reader_aliases, remove_patches
from cash.tracking.tracker_context import active_tracker, untracked

__all__ = ["FileAccessTracker", "install_read_watch", "tracking_seconds"]

logger = logging.getLogger(__name__)


def install_read_watch() -> None:
    """Watch reads from now on, not only inside cached calls.

    A memo is usually filled before any cached call runs -- ``main()`` logging
    its settings -- and a read nobody was watching is a read no entry can
    depend on. Called when a function is decorated. It turns on the ``open``
    audit event outside tracker scopes and installs no patch: a read through
    a C-level reader outside every cached call is not seen.
    """
    io_watch.watch_outside_scopes()


#: A file at least this big is hashed beside the body (`PendingDigest`)
#: rather than before it reads it: below, the thread costs more than it saves.
BACKGROUND_DIGEST_MIN_BYTES = 4_194_304  # 4 MiB


class FileAccessTracker:
    """Context manager that intercepts file I/O to record which files a
    statement reads.

    **ContextVar dispatch**: Python-level opens and directory listings
    arrive as audit events (:mod:`cash.tracking.io_watch`); the calls that
    raise none get dispatcher wrappers (listed in
    :mod:`cash.tracking.reader_patches`), plus a meta-path import hook for
    libraries loaded later. Both consult a
    ``ContextVar`` (``active_tracker``) at *call* time to decide whether to
    record the access. ``__enter__`` sets that ContextVar to ``self`` and
    stores the token; ``__exit__`` ``reset()``s it.

    Because ``ContextVar`` values are isolated per ``asyncio.Task`` and
    per ``threading.Thread`` by default, concurrent trackers — e.g. two
    coroutines under ``asyncio.gather``, or two worker threads — each
    see only their own block's reads.

    **Installed while in use**: the first tracker to open installs the
    wrappers and the last one to close puts the originals back, so with no
    tracker open ``pd.read_csv`` is pandas' own function again.

    **Writes are not dependencies.** A file opened for writing is never
    recorded as an input: a statement's output is hashed directly, and keying
    on its own output file would invalidate it on every run. ``open()`` writes
    are reported to the active effect observer instead (see
    ``cash.effect_observer``).
    """

    def __init__(self, user_ns=None, propagate_to_parent: bool = False, hash_on_read: bool = False):
        self.accessed_files = set()
        # The top-level package of the code being cached, when *user_ns* is a
        # module's globals (the decorator): an installed tool's own files are
        # its data. A notebook's namespace is `__main__` -- no package.
        name = user_ns.get("__name__") if isinstance(user_ns, dict) else None
        self._own_package = name.split(".")[0] if isinstance(name, str) and name != "__main__" else None
        # The content hash of each regular file WHEN IT WAS FIRST READ, for a
        # caller that stores what the block read (the decorator). Taken at
        # store time instead, a file changed mid-call by a writer that moves no
        # timestamp -- an np.memmap write on Windows -- was fingerprinted as
        # the NEW file next to a result computed from the old one, and served
        # to every later process. Moving the hash here costs
        # nothing extra: the snapshot reuses it while the stat is unchanged.
        self._hash_on_read = hash_on_read
        # Each digest, or the `PendingDigest` still taking it (``read_digests``).
        self._read_digests: dict[str, str | PendingDigest] = {}
        # When each of those digests was taken (``file_dep_is_fresh`` trusts an
        # unchanged file only if it had settled by then).
        self.read_hashed_at: dict[str, float] = {}
        #: Time spent hashing inside the block, which is cash's, not the body's.
        self.read_hash_seconds = 0.0
        # Remote URLs are kept in their own set, never in ``accessed_files``:
        # every consumer of that set stats/hashes its members, and a URL is not
        # a path. They are re-joined downstream as remote dependency entries.
        self.accessed_remote: set[str] = set()
        # Paths this block looked for and did not find. Recorded as their own
        # kind of dependency (``{'absent': True}``) so an entry computed
        # WITHOUT an optional file stops being valid once that file appears --
        # including when the same relative name resolves into a directory that
        # has one. See ``track_absent``.
        self.absent_files: set[str] = set()
        # Paths the block read that cash cannot stat under the spelling it
        # records (see ``_note_if_unstatable``). Each is stored as a dependency
        # that is never fresh: dropped, the entry would have no dependency on
        # the file at all, and every edit to it would be served stale.
        self.unresolved_files: set[str] = set()
        # Paths this block looked for and FOUND without reading them -- a flag
        # file, an output folder -- and what they were (``file``, ``dir`` or
        # ``any``). See ``track_present``.
        self.present_files: dict[str, str] = {}
        # ``(frame, the user code at and above it)`` from the last read's walk
        # up the stack (`credit_read_to_stack`): the next read from below the
        # same frame stops there. Dropped when the block ends, so the frame
        # is not kept alive past it.
        self.credit_anchor: tuple[Any, tuple[Any, ...]] | None = None
        # The same for `incidental_read`'s walk: what it found below the last
        # anchor frame. One slot, replaced as a whole, so pool threads share it.
        self.incidental_anchor: list = [None]
        # Folder -> its real path, for reads outside a cell run, which has
        # its own memo (`realpath_of_read_this_run`); dropped when the block
        # ends, so a link re-pointed between two calls is resolved anew.
        self._folders: dict[tuple[str, str], str] = {}
        self._probes_seen: set[tuple[Any, str | None]] = set()
        # The stat of each regular file WHEN IT WAS FIRST READ. The entry's
        # fingerprint is taken when it is stored, after the body has finished,
        # so a file that changed in between was fingerprinted as if it were
        # what the body read -- and served, stale, forever after.
        # Comparing against this is what lets the store step refuse instead.
        self.read_stats: dict[str, tuple[int, int, int]] = {}
        #: path -> its `regular_file_stat` as the call's snapshot took it, for
        #: `inputs_changed_since_read`; None when no snapshot is pending.
        self.snapshot_stats: dict[str, tuple[int, int, int] | None] | None = None
        #: path -> the stat `inputs_changed_since_read` took of it, when it
        #: found the file as read: a snapshot taken right after takes it
        #: rather than a stat of its own (`snapshot_file_deps`).
        self.checked_stats: dict[str, os.stat_result] = {}
        #: path -> the stat its deferred digest was taken under as this block
        #: ended (`_settle_deferred`): the first check that no input moved
        #: takes it rather than a stat of its own (``reuse_end_stats``).
        self.end_stats: dict[str, os.stat_result] = {}
        # The same for each folder the block listed (`stat_key` of the
        # folder): an entry landing in it while the block runs leaves a
        # listing without the entry, which a snapshot taken at store time
        # would vouch for forever after. Compared only for a folder the
        # block itself changed no entry of (`touched_dirs`).
        self.listed_stats: dict[str, tuple[int, int, int]] = {}
        # Folders the block wrote, made, removed or moved an entry in.
        self.touched_dirs: set[str] = set()
        # Each folder spelling a write named -> its resolved form: a loop
        # writing 2,000 files into one folder resolves it once.
        self._touched_spellings: dict[str, str] = {}
        # Files whose inode change time moves without their content changing
        # (a SQLite ``-wal`` a reader opens as root is chown'ed): compared by
        # size and mtime only in `inputs_changed_since_read`.
        self.ctime_unreliable: set[str] = set()
        self.user_ns = user_ns or {}
        # Stack of ContextVar tokens, one per active __enter__. Supports
        # re-entry of the same instance (an async function that reuses
        # a tracker across awaits, or a sync caller using `with` twice).
        self._token_stack: list[contextvars.Token] = []
        # When True, a file read in this block also registers with the
        # enclosing tracker(s) - so an OUTER cached function records the files
        # its (cold) inner cached calls read, otherwise the outer entry would
        # never invalidate when that file changes (file deps don't propagate
        # through depends_on). The decorator sets this; manual `with
        # FileAccessTracker()` nesting stays isolated by default.
        self._propagate_to_parent = propagate_to_parent
        self._parent_stack: list[Optional["FileAccessTracker"]] = []
        # The native file hook's recording windows (`native_reads`), one per
        # entry; None entries while the hook is off.
        self._native_windows: list[int | None] = []
        # The user code that read a file in THIS block (see
        # `credit_read_to_stack`): its recorded reads are live, not remembered.
        self.reading_codes: set[Any] = set()
        # Cached functions a call in this block served from an entry, and those
        # whose body ran here with no entry of its own (no key, a stream
        # finished from the function). A served entry brings exactly its own
        # files; see `FileDeps.credit_remembered_reads`.
        self.served_functions: set[str] = set()
        self.unentered_functions: set[str] = set()
        # Files a memo handed this block data from that was read from an
        # EARLIER version of the file (see `FileDeps.credit_remembered_reads`).
        self.stale_memo_reads: set[str] = set()
        # Sources a cached call in this block depends on through
        # ``dynamic_depends_on=`` that are not files: (id, token, object id)
        # -> (source, its token when that call keyed on it). An entry of this
        # block's own records them and asks them again on lookup
        # (`pass_dynamic_sources_up`, `dynamic_sources_fresh`).
        self.dynamic_sources: dict[tuple[str, str, int], tuple[Any, str]] = {}
        # The resolver calls behind them, each to be asked again by this
        # process (`dynamic_sources.Resolution`): a resolver may hand out a
        # new source object, which the recorded one never answers for.
        self.dynamic_resolutions: dict[tuple, Any] = {}
        # Cached functions called in this block whose resolver failed: what
        # they depend on is unknown, so an entry of this block's own could
        # never be checked against it and is not stored.
        self.unresolved_dynamic: set[str] = set()
        # Files and directories this block CREATED (opened with "w"/"x",
        # made with mkdir/mkdtemp), resolved. What the block reads back from
        # them is its own output, not an input: unzipping into a temporary
        # directory, reading the files and removing the directory was refused
        # on every call as "changed while the call was running".
        self.created_files: set[str] = set()
        self.created_dirs: list[str] = []
        # Reads of those paths, as the spellings a read records. Not inputs
        # while the block runs (no `read_stats`, so writing, reading back and
        # appending is not "changed while the call was running"), but one
        # still there when the block ends is: a "download if missing, then
        # read" loader reads that file on every later call, and an entry with
        # no dependency on it never saw the user replace it. The scratch file
        # the block removed is gone by then and stays out
        # (`get_accessed_files`).
        self.created_then_read: set[str] = set()
        # Work the block handed to a thread pool or thread (`work_submitted`) that has
        # not finished yet, and whether any was still running when the block
        # ended (`work_outlived_block`).
        self._work_lock = threading.Lock()
        self._work_running = 0
        self.work_outlived_block = False
        #: The functions of that work, for a caller that wants to know what
        #: running work can reach (`thread_reach`).
        self.work_functions: list[Any] = []

    def __enter__(self):
        # The first open tracker installs the wrappers (see
        # `reader_patches.install_patches`).
        io_watch.hold()
        # Capture the enclosing tracker (if any) BEFORE we become active, so a
        # read inside this block also registers with the outer tracker(s).
        self._parent_stack.append(active_tracker.get())
        self._token_stack.append(active_tracker.set(self))
        # Opens made in C libraries, when the native file hook is on.
        self._native_windows.append(native_reads.open_window())
        # A reader the block's own namespace bound by name (``from polars
        # import read_parquet``) after the wrappers went in.
        patch_reader_aliases(self.user_ns)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.credit_anchor = None
        self._folders = {}
        self.incidental_anchor = [None]
        if self._native_windows:
            # Before the parent stack pops: the reads pass up like any other.
            native_reads.close_window(self._native_windows.pop(), self)
        # The deferred digests, while the block's reads are what it read
        # (the native ones included): a file that moved is never fresh from
        # here on, before anything is snapshotted.
        self._settle_deferred()
        if self._token_stack:
            reset_in_any_context(active_tracker, self._token_stack.pop())
            io_watch.release()
            if not self._token_stack and self._work_running:
                self.work_outlived_block = True
        if self._parent_stack:
            self._parent_stack.pop()

    def _propagation_parent(self) -> FileAccessTracker | None:
        """The enclosing tracker a record is passed up to, or None when this
        tracker is isolated (the default for manual nesting)."""
        if not self._propagate_to_parent or not self._parent_stack:
            return None
        parent = self._parent_stack[-1]
        return parent if parent is not self else None

    def _self_and_parents(self) -> Iterator[FileAccessTracker]:
        """This tracker, then each enclosing one a record is passed up to."""
        tracker: FileAccessTracker | None = self
        while tracker is not None:
            yield tracker
            tracker = tracker._propagation_parent()

    def work_submitted(self, fn: Any = None) -> list[FileAccessTracker]:
        """Count one piece of work (running *fn*) handed to a thread pool or
        thread, here and on each tracker a read is passed up to; returns them,
        for `work_finished` (the chain is gone once the block has ended)."""
        trackers = list(self._self_and_parents())
        for tracker in trackers:
            with tracker._work_lock:
                tracker._work_running += 1
                if fn is not None:
                    tracker.work_functions.append(fn)
        return trackers

    @staticmethod
    def work_finished(trackers: list[FileAccessTracker]) -> None:
        """That work is done (or was cancelled before it ran)."""
        for tracker in trackers:
            with tracker._work_lock:
                tracker._work_running -= 1

    def suspend(self):
        """Stop tracking until :meth:`resume`, restoring the enclosing tracker.

        For a streaming cached generator: production is tracked, the caller's
        loop body is not. `__enter__` is too costly to run per item; this is
        one ContextVar swap.
        """
        parent = self._parent_stack[-1] if self._parent_stack else None
        return active_tracker.set(parent)

    def resume(self, token) -> None:
        """Undo :meth:`suspend`, in whatever context the stream is advanced from."""
        reset_in_any_context(active_tracker, token)

    def get_accessed_files(self) -> set[str]:
        """The paths this block read, once it has ended: what it read of its
        own output counts too while that is still there (`created_then_read`),
        snapshotted with its content as it is now."""
        if not self.created_then_read:
            return self.accessed_files
        kept = set()
        for path in self.created_then_read:
            try:
                if os.path.exists(path):
                    kept.add(path)
            except (OSError, ValueError):
                pass
        return self.accessed_files | kept if kept else self.accessed_files

    def inputs_changed_since_read(self, reuse_end_stats: bool = False) -> list[str]:
        """Regular files whose stat moved between their first read and now,
        and folders listed whose stat moved since the listing.

        With *reuse_end_stats*, a file whose deferred digest was taken as the
        block ended is compared by the stat that digest was taken under
        (`end_stats`), not a stat of its own: for the first of a statement's
        two checks, with only cash's code run since. The last check before
        an entry is written always stats again.

        A folder the block changed an entry of itself (`touched_dirs`) is
        left out: a function that lists a folder and writes its output into
        it moves the folder's stat itself, and the stat cannot tell its own
        entry from one that landed from outside. Any other folder that moved
        had an entry added or removed by someone else while the block ran,
        and the listing it made is not the folder a snapshot would describe.
        """
        moved = []
        # The stats the snapshot just took, once (`snapshot_tracked_deps`). A
        # file that moves after the snapshot is not caught here, but the
        # entry holds the stat from before the move, so it is never served
        # for the new content.
        taken, self.snapshot_stats = self.snapshot_stats, None
        if taken is None:
            taken = {}
        checked: dict[str, os.stat_result] = {}
        self.checked_stats = checked
        ended, self.end_stats = self.end_stats, {}
        if not reuse_end_stats:
            ended = {}
        # Over copies: a pool thread the block started can still be reading
        # after it ended, adding to these while they are walked.
        for path, before in list(self.read_stats.items()):
            if path in taken:
                now = taken[path]
            else:
                st = ended.get(path) or regular_file_stat_result(path)
                now = stat_key(st) if st is not None else None
                if now == before:
                    checked[path] = st
            if now != before and not (path in self.ctime_unreliable and now is not None and now[:2] == before[:2]):
                moved.append(path)
        for path, before in list(self.listed_stats.items()):
            if path not in self.touched_dirs and folder_stat(path) != before:
                moved.append(path)
        return moved

    def take_checked_stats(self) -> dict[str, os.stat_result]:
        """`checked_stats`, once: a later snapshot takes stats of its own."""
        checked, self.checked_stats = self.checked_stats, {}
        return checked

    def known_reads(self) -> dict[str, tuple[Any, ...]]:
        """``{path: (stat when read, digest when read, when it was hashed)}``
        for each file hashed as it was read (``hash_on_read``): what a
        snapshot reuses while the stat is unchanged (`snapshot_file_deps`)."""
        return {
            path: (self.read_stats[path], digest, self.read_hashed_at.get(path))
            for path, digest in self.read_digests.items()
            if path in self.read_stats
        }

    def note_ctime_unreliable(self, path) -> None:
        """Compare *path* by size and mtime only in the mid-call check, here
        and in the enclosing trackers (see ``ctime_unreliable``)."""
        try:
            raw = os.fsdecode(path) if isinstance(path, bytes) else str(path)
            resolved = normalize_path(os.path.realpath(raw))
        except (TypeError, ValueError, OSError):
            return
        for tracker in self._self_and_parents():
            tracker.ctime_unreliable.add(resolved)

    def get_accessed_remote_urls(self) -> set[str]:
        """Remote URLs read in this block, tracked by store validator instead."""
        return self.accessed_remote

    def get_absent_files(self) -> set[str]:
        """Paths this block looked for and did not find."""
        return self.absent_files

    def get_present_files(self) -> dict[str, str]:
        """Paths this block probed and found, with what it asked them to be."""
        return self.present_files

    def get_unresolved_files(self) -> set[str]:
        """Paths this block read that cannot be stat'ed as recorded."""
        return self.unresolved_files

    # The three methods below are what the feeders (`read_events`,
    # `reader_patches`, `read_credit`) call when they see a read.

    def track_path(self, path) -> bool:
        """Record a read of *path*, unless it is not a user dependency.
        True when resolving it just found a regular file there (its
        ``lstat``), so the caller need not stat it to know it is there."""
        global _tracking_seconds
        started = _perf_counter()
        try:
            # Untracked: deciding whether this read counts probes paths too
            # (the time zone directories, looked up on the first read in a
            # process), and through the same wrapped functions.
            with untracked():
                return self._track_path_untimed(path)
        finally:
            _tracking_seconds += _perf_counter() - started

    def note_created(self, path, directory: bool = False) -> None:
        """Record *path* as created by this block (and, when propagating, by
        the enclosing ones): not an input from here on (`created_by_block`)."""
        try:
            raw = os.fsdecode(path) if isinstance(path, bytes) else str(path)
            resolved = normalize_path(realpath_of_read_this_run(raw)[0])  # its folder resolved once a run
        except (TypeError, ValueError, OSError):
            return
        for tracker in self._self_and_parents():
            if resolved not in tracker.accessed_files:  # read first: an input
                if directory:
                    tracker.created_dirs.append(resolved.rstrip("/") + "/")
                else:
                    tracker.created_files.add(resolved)

    def note_entry_changed(self, path) -> None:
        """Record that the block writes, makes, removes or moves *path*: the
        folder holding it changes by the block's own hand (`touched_dirs`)."""
        try:
            raw = os.fsdecode(path) if isinstance(path, bytes) else str(path)
            spelled = os.path.dirname(os.path.abspath(raw))
            folder = self._touched_spellings.get(spelled)
            if folder is None:
                folder = normalize_path(os.path.realpath(spelled))
                self._touched_spellings[spelled] = folder
        except (TypeError, ValueError, OSError):
            return
        for tracker in self._self_and_parents():
            tracker.touched_dirs.add(folder)

    def created_by_block(self, abs_path: str) -> bool:
        """Did this block create *abs_path*, or a directory it lies in?"""
        if abs_path in self.created_files:
            return True
        return any(abs_path.startswith(d) or abs_path + "/" == d for d in self.created_dirs)

    def _track_path_untimed(self, path) -> bool:
        if not isinstance(path, (str, bytes, os.PathLike)):
            # ``open(3)`` opens a file DESCRIPTOR: joblib and loky do, and
            # ``str(3)`` was recorded as a read of ``<cwd>/3`` -- a directory
            # every file under the cwd sits in, so every write there read as
            # an input.
            return False
        raw_path = os.fsdecode(path) if isinstance(path, bytes) else str(path)
        if is_pseudo_fs(raw_path):
            # See `is_pseudo_fs`. Checked BEFORE realpath, which on
            # Windows rewrites /proc/... to C:/proc/... and would slip past.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", raw_path)
            return False
        if is_remote_url(raw_path):
            # A remote URL is a real dependency, just not a stat-able one:
            # ``realpath`` would mangle it into a nonexistent local path and the
            # dependency would vanish. Record it on the remote channel, where it
            # is tracked by the store's own validator.
            self.add_tracked_remote(raw_path)
            return False
        try:
            # Normalize path using realpath to get canonical path
            # This resolves symlinks and normalizes the path, making it
            # stable across os.chdir() calls. Resolved once per cell run
            # (``realpath_this_run``): a loop reads the same files again.
            resolved, read_lstat = realpath_of_read_this_run(raw_path, self._folders)
            abs_path = normalize_path(resolved)
        except (TypeError, ValueError, OSError) as e:
            logger.debug("[TRACKER] Could not track file path %r: %s", path, e)
            return False
        # Returned, not kept on the tracker: a thread pool's workers share it,
        # and a flag one of them set would answer another's missing file.
        found = read_lstat is not None
        if is_pseudo_fs(abs_path):
            # See `is_pseudo_fs`: recording one of these makes the entry
            # permanently unfreshenable. Return before the relative-path arm
            # too — these paths are always absolute.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", abs_path)
            return found
        if SCRATCH_MEMMAP in abs_path:
            # joblib's memmaps of a parallel call's arrays: deleted when the
            # call returns, so an entry depending on one could never be fresh.
            logger.debug("[TRACKER] Ignoring joblib scratch read %r", abs_path)
            return found
        if is_runtime_cache_path(abs_path):
            logger.debug("[TRACKER] Ignoring runtime-cache read %r", abs_path)
            return found
        if is_cash_internal(abs_path):
            # See `is_cash_internal`. Checked after realpath so a relative
            # or symlinked cache path is caught too.
            logger.debug("[TRACKER] Ignoring cash-internal read %r", abs_path)
            return found
        if (self.created_files or self.created_dirs) and self.created_by_block(abs_path):
            logger.debug("[TRACKER] A read of %r, which this block created: an input if it stays", abs_path)
            self._add_created_read(abs_path, self._unresolved_spelling(path, abs_path))
            return found
        why = incidental_read(abs_path, self._own_package, self.incidental_anchor)
        if why is not None:
            logger.debug("[TRACKER] Ignoring %s read %r", why, abs_path)
            return found
        self.add_tracked(abs_path, lstat=read_lstat)
        if abs_path not in self.read_stats and os.path.isabs(abs_path):
            self._note_if_unstatable(raw_path, abs_path)
        try:
            credit_read_to_stack(abs_path, self)
        except Exception:  # attribution is an aid; the read counts regardless
            logger.debug("[TRACKER] Could not credit %r to the stack", abs_path, exc_info=True)
        twin = self._unresolved_spelling(path, abs_path)
        if twin is not None:
            self.add_tracked(twin)
        return found

    @staticmethod
    def _unresolved_spelling(path, abs_path: str) -> str | None:
        """The read path as written, when a read records it beside *abs_path*.

        A RELATIVE read path also records the UN-resolved relative string as
        its own dependency. The realpath is frozen to the cwd at track time,
        so after an ``os.chdir`` edit it still looks fresh even though a
        re-run would read a different file. The relative dep is re-resolved
        against the CURRENT cwd at each freshness check
        (``resolve_file_dep_path``), so a collision - a different file with
        the same relative name in the new directory - is caught via its
        mtime/size.

        An ABSOLUTE path through a junction or symlink gets the same
        treatment: its unresolved form is recorded too. The realpath resolved
        the link at WRITE time, so after the link is re-pointed every check
        stats the old target -- which still exists and has not changed; a
        rollback would return the NEWER release's answer. Checked through the
        link as it points NOW, the switch is seen; the realpath entry keeps
        catching an edit to the target itself.
        """
        try:
            raw = str(path)
            if raw and not os.path.isabs(raw):
                rel = normalize_path(raw)
                return rel if rel != abs_path else None
            if raw:
                link = normalize_path(os.path.abspath(raw))
                return link if os.path.normcase(link) != os.path.normcase(abs_path) else None
        except (TypeError, ValueError, OSError):
            logger.debug("[TRACKER] Could not record unresolved path for %r", path)
        return None

    def _add_created_read(self, *paths: str | None) -> None:
        """Record a read of the block's own output (`created_then_read`), here
        and on the parents."""
        for tracker in self._self_and_parents():
            tracker.created_then_read.update(p for p in paths if p is not None)

    def _note_if_unstatable(self, raw_path: str, abs_path: str) -> None:
        """Fail closed when the recorded spelling of a read cannot be stat'ed.

        Reached for a path that gave no regular-file stat: a directory, a file
        that is not there (the read failed, and the absence is recorded on its
        own), or a spelling cash cannot stat although the reader opened it --
        a long Windows path whose ``\\\\?\\`` prefix was lost, say. The last
        is the one that matters: the snapshot drops a path it cannot stat, so
        the entry would be stored with no dependency on the file.
        """
        try:
            own_stat(abs_path)
            return
        except (OSError, ValueError):
            pass
        try:
            own_stat(raw_path)
        except (OSError, ValueError):
            return  # not there under either spelling: nothing was read
        logger.debug("[TRACKER] %r was read but %r cannot be stat'ed; never fresh", raw_path, abs_path)
        self.add_tracked_unresolved(abs_path)

    def add_tracked_unresolved(self, path: str) -> None:
        """Record a read that cannot be checked, here and on the parents."""
        for tracker in self._self_and_parents():
            tracker.unresolved_files.add(path)

    def add_tracked(self, abs_path: str, digest: str | None = None, lstat: Any = None) -> None:
        """Record *abs_path* on this tracker and, when propagation is enabled,
        on the enclosing tracker(s) too - so nested cached reads count as the
        outer cached function's deps. Manual tracker nesting stays isolated.

        *digest* is the content hash an inner tracker already took of the same
        read, handed up so the outer one does not hash the file again.

        *lstat* is the stat taken while resolving the path, of a regular file
        that is not a link, so it is the stat the file would have given."""
        st = stat_key(lstat) if lstat is not None else None
        for tracker in self._self_and_parents():
            # One read, one stat: an enclosing tracker takes the stat this
            # one took of the same read (the file is stat'ed once per
            # tracker otherwise, at each nested loop level).
            digest, st = tracker._add_tracked_here(abs_path, digest, st, lstat)

    @property
    def read_digests(self) -> dict[str, str]:
        """Each file's content hash as the body first read it (see
        ``__init__``), waiting for any still being taken (`PendingDigest`)."""
        self._settle_deferred()
        digests: dict[str, str] = {}
        for path, digest in list(self._read_digests.items()):
            if isinstance(digest, PendingDigest):
                digest = digest.result()
                if digest is None:
                    del self._read_digests[path]
                    continue
                self._read_digests[path] = digest
            digests[path] = digest
        return digests

    def _settle_deferred(self) -> None:
        """Take this block's deferred digests (`DeferredDigest`). A file that
        moved since it was read is recorded as never fresh: what the block
        read of it is gone, and the digest of what is there now would vouch
        for a value computed from something else."""
        for path, digest in list(self._read_digests.items()):
            if isinstance(digest, DeferredDigest):
                digest, st = digest.taken()
                if st is not None:
                    self.end_stats[path] = st
                if digest is MOVED:
                    self.add_tracked_unresolved(path)
                    digest = None
                if digest is None:
                    del self._read_digests[path]
                else:
                    self._read_digests[path] = digest

    def _add_tracked_here(
        self,
        abs_path: str,
        digest: str | PendingDigest | None,
        st: tuple[int, int, int] | None,
        lstat: Any = None,
    ) -> tuple[str | PendingDigest | None, tuple[int, int, int] | None]:
        """`add_tracked` on this tracker alone, *st* the `stat_key` of this
        read when one was taken (of *lstat*, when given); returns the digest
        and the stat to hand up."""
        self.accessed_files.add(abs_path)
        if abs_path not in self.read_stats and os.path.isabs(abs_path):
            # Absolute paths only: a relative twin is re-resolved against the
            # cwd at check time, and a chdir during the call would make its
            # stat look like a change that never happened.
            if st is None:
                st = regular_file_stat(abs_path)
            if st is None and abs_path not in self.listed_stats:
                folder = folder_stat(abs_path)
                if folder is not None:
                    self.listed_stats[abs_path] = folder
            if st is not None:
                self.read_stats[abs_path] = st
                if self._hash_on_read:
                    if digest is None:
                        self.read_hashed_at[abs_path] = time.time()
                        digest = self._digest_now(abs_path, st[0], lstat)
                    if digest is not None:
                        self._read_digests[abs_path] = digest
        elif digest is None:
            digest = self._read_digests.get(abs_path)
        return digest, st

    def _digest_now(self, abs_path: str, size: int, st: Any = None) -> str | PendingDigest | DeferredDigest | None:
        """The file's content hash as the body is about to read it; *st* the
        stat this read took of it, when there is one (`file_content_hash`).

        A big file the process does not know yet is hashed on a thread of its
        own (`PendingDigest`) while the body reads and computes; the digest
        is waited for when the entry is stored (``read_digests``), and before
        anything in this process opens the file to write (`settle_before_write`).
        """
        t0 = _perf_counter()
        try:
            if size >= BACKGROUND_DIGEST_MIN_BYTES:
                known = remembered_file_digest(abs_path)
                return known if known is not None else PendingDigest(abs_path, size, file_content_hash)
            if st is not None:
                later = digest_later(abs_path, size, st)
                if later is not None:
                    return later
            return file_content_hash(abs_path, size, st)
        finally:
            self.read_hash_seconds += _perf_counter() - t0

    def note_reading_code(self, code: Any) -> None:
        """Record *code* as user code that read a file in this block, here and
        in every tracker this one propagates to."""
        for tracker in self._self_and_parents():
            tracker.reading_codes.add(code)

    def note_served(self, func_name: str) -> None:
        """Record that a call of the cached *func_name* was served from an
        entry in this block, here and in every tracker this one propagates to."""
        for tracker in self._self_and_parents():
            tracker.served_functions.add(func_name)

    def note_unentered(self, func_name: str) -> None:
        """Record that the body of the cached *func_name* ran in this block
        without an entry of its own, here and in every tracker this one
        propagates to."""
        for tracker in self._self_and_parents():
            tracker.unentered_functions.add(func_name)

    def track_absent(self, path) -> None:
        """Record *path* as looked-for-and-missing."""
        if self._probed_before(path, None):
            return
        # Untracked: the filters below probe paths too, through the same
        # wrapped functions.
        with untracked():
            normalized = self._probed_path(path)
        if normalized is not None:
            self.add_tracked_absent(normalized)

    def track_present(self, path, kind: str) -> None:
        """Record *path* as looked-for-and-found: ``kind`` is ``file``,
        ``dir`` or ``any`` (what the probe asked). A flag file checked with
        ``os.path.exists`` and never opened decides the result by being there,
        and a result computed with it stays right only while it is."""
        if self._probed_before(path, kind):
            return
        # Untracked: the filters below probe paths too, through the same
        # wrapped functions.
        with untracked():
            normalized = self._probed_path(path)
        if normalized is not None:
            self.add_tracked_present(normalized, kind)

    def _probed_before(self, path, kind: str | None) -> bool:
        """Was this probe, with this answer, recorded in this block already?

        A loop that checks the same file on every iteration asked once, and
        recording it again would cost time for nothing. The recorded
        form does not depend on the working directory (see `_probed_path`),
        so the path as given is the key.
        """
        try:
            key = (os.fspath(path), kind)
        except TypeError:
            return False
        if key in self._probes_seen:
            return True
        if len(self._probes_seen) < 65536:
            self._probes_seen.add(key)
        return False

    def _probed_path(self, path) -> str | None:
        """How an existence probe of *path* is recorded, or None when it is
        not the user's question.

        Kept as WRITTEN, not resolved: a relative probe is about "a file with
        this name, here", and that is exactly the thing that must be
        re-evaluated against the live cwd on the next run. Resolving it would
        freeze the directory the probe happened to run in -- which is the bug
        this exists to close, in mirror image.

        The same filters as ``track_path``: kernel pseudo-filesystems and
        cash's own storage are not user dependencies.
        """
        try:
            raw = os.fsdecode(path) if isinstance(path, bytes) else str(path)
        except (TypeError, ValueError):
            return None
        if not raw or is_pseudo_fs(raw):
            return None
        try:
            normalized = normalize_path(raw)
        except (TypeError, ValueError):
            return None
        if is_cash_internal(normalized):
            return None
        if os.path.isabs(raw):
            # An absolute probe is about one fixed file, so record it resolved
            # the way a read of it would be -- otherwise the two spellings of
            # the same path would not match.
            try:
                normalized = normalize_path(os.path.realpath(raw))
            except (TypeError, ValueError, OSError):
                pass
            if is_pseudo_fs(normalized) or is_cash_internal(normalized):
                return None
        # A library probing for an optional file while it is imported
        # (matplotlib looks for a `matplotlibrc` in the working directory) or
        # inside its own package is not the user's question either.
        if incidental_read(os.path.abspath(raw), self._own_package, self.incidental_anchor) is not None:
            return None
        if (self.created_files or self.created_dirs) and self.created_by_block(normalize_path(os.path.realpath(raw))):
            # The block's own output: whether it is there is its doing, while
            # it runs. Kept, it is checked as a read would be.
            self._add_created_read(normalized)
            return None
        return normalized

    def add_tracked_absent(self, path: str) -> None:
        """Record an absent path here and, when propagating, on the parents."""
        for tracker in self._self_and_parents():
            tracker.absent_files.add(path)

    def add_tracked_present(self, path: str, kind: str) -> None:
        """Record a path probed and found here and, when propagating, on the
        parents. Probed as two kinds, it is recorded as ``any``."""
        for tracker in self._self_and_parents():
            known = tracker.present_files.get(path)
            tracker.present_files[path] = kind if known in (None, kind) else "any"

    def add_tracked_remote(self, url: str) -> None:
        """Record a remote *url* read, propagating to the enclosing tracker."""
        for tracker in self._self_and_parents():
            tracker.accessed_remote.add(url)

    def add_dynamic_source(self, source_id: str, source: Any, token: str) -> None:
        """Record a data source this block depends on, with its token now,
        here and on the parents. Every source is kept, two with one id too:
        an id may name a catalog rather than the table, and keeping the
        first source met under it lost a change to the second. A token that
        moved within the block makes the entry miss on its next lookup."""
        for tracker in self._self_and_parents():
            tracker.dynamic_sources.setdefault((source_id, token, id(source)), (source, token))

    def add_unresolved_dynamic(self, func_name: str) -> None:
        """Record that *func_name*, called in this block, could not resolve
        its ``dynamic_depends_on=`` sources, here and on the parents."""
        for tracker in self._self_and_parents():
            tracker.unresolved_dynamic.add(func_name)

    def add_dynamic_resolution(self, key: tuple, resolution: Any) -> None:
        """Record a ``dynamic_depends_on=`` resolver call this block depends
        on, here and on the parents (`dynamic_sources.Resolution`)."""
        for tracker in self._self_and_parents():
            tracker.dynamic_resolutions.setdefault(key, resolution)


#: Seconds spent recording reads; `tracking_seconds`.
_tracking_seconds = 0.0


def tracking_seconds() -> float:
    """Seconds cash has spent recording file reads in this process.

    Read before and after a statement, the difference is cash's own time inside
    it, which is not the statement's cost and must not be credited as time a
    later hit saves.
    """
    return _tracking_seconds


# What feeds a tracker: the audit events Python raises for its own opens and
# listings, and wrappers on the readers that raise none, installed while one
# is open.
subscribe_read_events()
io_watch.add_patcher(install_patches, remove_patches)
