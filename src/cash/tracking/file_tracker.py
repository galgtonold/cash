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
import time
from typing import Any, Optional

from cash._clock import perf_counter as _perf_counter
from cash._paths import is_remote_url, normalize_path
from cash.tracking import io_watch
from cash.tracking.file_dep_snapshot import file_content_hash, realpath_of_read_this_run
from cash.tracking.read_classification import (
    RUNTIME_CACHE_SEGMENT,
    RUNTIME_CACHE_SUFFIXES,
    SCRATCH_MEMMAP,
    incidental_read,
    is_cash_internal,
    is_pseudo_fs,
    regular_file_stat,
)
from cash.tracking.read_credit import credit_read_to_stack
from cash.tracking.read_events import subscribe_read_events
from cash.tracking.reader_patches import install_patches, remove_patches
from cash.tracking.tracker_context import active_tracker

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


class FileAccessTracker:
    """Context manager that intercepts file I/O to record which files a
    statement reads.

    **ContextVar dispatch**: Python-level opens and directory listings
    arrive as audit events (:mod:`cash.tracking.io_watch`); readers that open
    in C, existence probes and ``Path.stat`` get dispatcher wrappers, plus a
    meta-path import hook for libraries loaded later. Both consult a
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
        self.read_digests: dict[str, str] = {}
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
        # has one. See ``_track_absent``.
        self.absent_files: set[str] = set()
        # The stat of each regular file WHEN IT WAS FIRST READ. The entry's
        # fingerprint is taken when it is stored, after the body has finished,
        # so a file that changed in between was fingerprinted as if it were
        # what the body read -- and served, stale, forever after.
        # Comparing against this is what lets the store step refuse instead.
        self.read_stats: dict[str, tuple[int, int, int]] = {}
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
        # The user code that read a file in THIS block (see
        # `credit_read_to_stack`): its recorded reads are live, not remembered.
        self.reading_codes: set[Any] = set()
        # Files a memo handed this block data from that was read from an
        # EARLIER version of the file (see `Cash._credit_remembered_reads`).
        self.stale_memo_reads: set[str] = set()

    def __enter__(self):
        # The first open tracker installs the wrappers (see
        # `reader_patches.install_patches`).
        io_watch.hold()
        # Capture the enclosing tracker (if any) BEFORE we become active, so a
        # read inside this block also registers with the outer tracker(s).
        self._parent_stack.append(active_tracker.get())
        self._token_stack.append(active_tracker.set(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token_stack:
            active_tracker.reset(self._token_stack.pop())
            io_watch.release()
        if self._parent_stack:
            self._parent_stack.pop()

    def _propagation_parent(self) -> FileAccessTracker | None:
        """The enclosing tracker a record is passed up to, or None when this
        tracker is isolated (the default for manual nesting)."""
        if not self._propagate_to_parent or not self._parent_stack:
            return None
        parent = self._parent_stack[-1]
        return parent if parent is not self else None

    def suspend(self):
        """Stop tracking until :meth:`resume`, restoring the enclosing tracker.

        For a streaming cached generator: production is tracked, the caller's
        loop body is not. `__enter__` cannot be used per item -- it took 5.1us
        against 0.15us for the ContextVar swap alone, which on a 200k-item
        iterator is over a second of pure bookkeeping.
        """
        parent = self._parent_stack[-1] if self._parent_stack else None
        return active_tracker.set(parent)

    def resume(self, token) -> None:
        """Undo :meth:`suspend`."""
        active_tracker.reset(token)

    def get_accessed_files(self) -> set[str]:
        return self.accessed_files

    def inputs_changed_since_read(self) -> list[str]:
        """Regular files whose stat moved between their first read and now.

        Directories are left out on purpose: a function that lists a directory
        and writes its output into it moves the directory's mtime itself, and
        refusing to cache every such function would be a regression for no
        gain -- a new entry appearing mid-call is a smaller hole than a file
        rewritten under the reader.
        """
        moved = []
        for path, before in self.read_stats.items():
            if regular_file_stat(path) != before:
                moved.append(path)
        return moved

    def get_accessed_remote_urls(self) -> set[str]:
        """Remote URLs read in this block, tracked by store validator instead."""
        return self.accessed_remote

    def get_absent_files(self) -> set[str]:
        """Paths this block looked for and did not find."""
        return self.absent_files

    def _track_path(self, path):
        global _tracking_seconds
        started = _perf_counter()
        try:
            self._track_path_untimed(path)
        finally:
            _tracking_seconds += _perf_counter() - started

    def _track_path_untimed(self, path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            # ``open(3)`` opens a file DESCRIPTOR: joblib and loky do, and
            # ``str(3)`` was recorded as a read of ``<cwd>/3`` -- a directory
            # every file under the cwd sits in, so every write there read as
            # an input.
            return
        raw_path = os.fsdecode(path) if isinstance(path, bytes) else str(path)
        if is_pseudo_fs(raw_path):
            # See `is_pseudo_fs`. Checked BEFORE realpath, which on
            # Windows rewrites /proc/... to C:/proc/... and would slip past.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", raw_path)
            return
        if is_remote_url(raw_path):
            # A remote URL is a real dependency, just not a stat-able one:
            # ``realpath`` would mangle it into a nonexistent local path and the
            # dependency would vanish. Record it on the remote channel, where it
            # is tracked by the store's own validator.
            self.add_tracked_remote(raw_path)
            return
        try:
            # Normalize path using realpath to get canonical path
            # This resolves symlinks and normalizes the path, making it
            # stable across os.chdir() calls. Resolved once per cell run
            # (``realpath_this_run``): a loop reads the same files again.
            resolved, read_lstat = realpath_of_read_this_run(raw_path)
            abs_path = normalize_path(resolved)
        except (TypeError, ValueError, OSError) as e:
            logger.debug("[TRACKER] Could not track file path %r: %s", path, e)
            return
        if is_pseudo_fs(abs_path):
            # See `is_pseudo_fs`: recording one of these makes the entry
            # permanently unfreshenable. Return before the relative-path arm
            # too — these paths are always absolute.
            logger.debug("[TRACKER] Ignoring pseudo-fs read %r", abs_path)
            return
        if SCRATCH_MEMMAP in abs_path:
            # joblib's memmaps of a parallel call's arrays: deleted when the
            # call returns, so recorded, every entry that read them was stale
            # for ever -- a ``cross_val_predict(n_jobs=4)`` loop re-ran on
            # every run of the report cell.
            logger.debug("[TRACKER] Ignoring joblib scratch read %r", abs_path)
            return
        if RUNTIME_CACHE_SEGMENT in abs_path or abs_path.endswith(RUNTIME_CACHE_SUFFIXES):
            logger.debug("[TRACKER] Ignoring runtime-cache read %r", abs_path)
            return
        if is_cash_internal(abs_path):
            # See `is_cash_internal`. Checked after realpath so a relative
            # or symlinked cache path is caught too.
            logger.debug("[TRACKER] Ignoring cash-internal read %r", abs_path)
            return
        why = incidental_read(abs_path, self._own_package)
        if why is not None:
            logger.debug("[TRACKER] Ignoring %s read %r", why, abs_path)
            return
        self.add_tracked(abs_path, lstat=read_lstat)
        try:
            credit_read_to_stack(abs_path, self)
        except Exception:  # noqa: BLE001 - attribution is an aid; the read counts regardless
            logger.debug("[TRACKER] Could not credit %r to the stack", abs_path, exc_info=True)
        # a RELATIVE read path also records the UN-resolved relative
        # string as its own dependency. The realpath above is frozen to the cwd
        # at track time, so after an ``os.chdir`` edit it still looks fresh even
        # though a re-run would read a different file. The relative dep is
        # re-resolved against the CURRENT cwd at each freshness check
        # (``resolve_file_dep_path``), so a collision - a different file with the
        # same relative name in the new directory - is caught via its mtime/size.
        try:
            raw = str(path)
            if raw and not os.path.isabs(raw):
                rel = normalize_path(raw)
                if rel != abs_path:
                    self.add_tracked(rel)
            elif raw:
                # An ABSOLUTE path through a junction or symlink gets the same
                # treatment: its unresolved form is recorded too. The realpath
                # above resolved the link at WRITE time, so after the link is
                # re-pointed every check stats the old target -- which still
                # exists and has not changed; a rollback would return the
                # NEWER release's answer. Checked through the link as it
                # points NOW, the switch is seen; the realpath entry keeps
                # catching an edit to the target itself.
                link = normalize_path(os.path.abspath(raw))
                if os.path.normcase(link) != os.path.normcase(abs_path):
                    self.add_tracked(link)
        except (TypeError, ValueError, OSError):
            logger.debug("[TRACKER] Could not record unresolved path for %r", path)

    def add_tracked(self, abs_path: str, digest: str | None = None, lstat: Any = None) -> None:
        """Record *abs_path* on this tracker and, when propagation is enabled,
        on the enclosing tracker(s) too - so nested cached reads count as the
        outer cached function's deps. Manual tracker nesting stays isolated.

        *digest* is the content hash an inner tracker already took of the same
        read, handed up so the outer one does not hash the file again.

        *lstat* is the stat taken while resolving the path, of a regular file
        that is not a link, so it is the stat the file would have given."""
        self.accessed_files.add(abs_path)
        if abs_path not in self.read_stats and os.path.isabs(abs_path):
            # Absolute paths only: a relative twin is re-resolved against the
            # cwd at check time, and a chdir during the call would make its
            # stat look like a change that never happened.
            st = (
                regular_file_stat(abs_path)
                if lstat is None
                else (lstat.st_size, lstat.st_mtime_ns, getattr(lstat, "st_ctime_ns", 0))
            )
            if st is not None:
                self.read_stats[abs_path] = st
                if self._hash_on_read:
                    if digest is None:
                        self.read_hashed_at[abs_path] = time.time()
                        digest = self._digest_now(abs_path, st[0])
                    if digest is not None:
                        self.read_digests[abs_path] = digest
        elif digest is None:
            digest = self.read_digests.get(abs_path)
        parent = self._propagation_parent()
        if parent is not None:
            parent.add_tracked(abs_path, digest)

    def _digest_now(self, abs_path: str, size: int) -> str | None:
        """The file's content hash as the body is about to read it."""
        t0 = _perf_counter()
        try:
            return file_content_hash(abs_path, size)
        finally:
            self.read_hash_seconds += _perf_counter() - t0

    def _note_reading_code(self, code: Any) -> None:
        self.reading_codes.add(code)
        parent = self._propagation_parent()
        if parent is not None:
            parent._note_reading_code(code)

    def _track_absent(self, path) -> None:
        """Record *path* as looked-for-and-missing.

        Kept as WRITTEN, not resolved: a relative probe is about "a file with
        this name, here", and that is exactly the thing that must be
        re-evaluated against the live cwd on the next run. Resolving it would
        freeze the directory the probe happened to run in -- which is the bug
        this exists to close, in mirror image.

        The same filters as ``_track_path``: kernel pseudo-filesystems and
        cash's own storage are not user dependencies.
        """
        try:
            raw = str(path)
        except (TypeError, ValueError):
            return
        if not raw or is_pseudo_fs(raw):
            return
        try:
            normalized = normalize_path(raw)
        except (TypeError, ValueError):
            return
        if is_cash_internal(normalized):
            return
        if os.path.isabs(raw):
            # An absolute probe is about one fixed file, so record it resolved
            # the way a read of it would be -- otherwise the two spellings of
            # the same path would not match.
            try:
                normalized = normalize_path(os.path.realpath(raw))
            except (TypeError, ValueError, OSError):
                pass
            if is_pseudo_fs(normalized) or is_cash_internal(normalized):
                return
        # A library probing for an optional file while it is imported
        # (matplotlib looks for a `matplotlibrc` in the working directory) or
        # inside its own package is not the user's question either.
        if incidental_read(os.path.abspath(raw), self._own_package) is not None:
            return
        self.add_tracked_absent(normalized)

    def add_tracked_absent(self, path: str) -> None:
        """Record an absent path here and, when propagating, on the parents."""
        self.absent_files.add(path)
        parent = self._propagation_parent()
        if parent is not None:
            parent.add_tracked_absent(path)

    def add_tracked_remote(self, url: str) -> None:
        """Record a remote *url* read, propagating to the enclosing tracker."""
        self.accessed_remote.add(url)
        parent = self._propagation_parent()
        if parent is not None:
            parent.add_tracked_remote(url)


#: Seconds spent recording reads; `tracking_seconds`.
_tracking_seconds = 0.0


def tracking_seconds() -> float:
    """Seconds cash has spent recording file reads in this process.

    Read before and after a statement, the difference is cash's own time inside
    it, which is not the statement's cost: a folder read recorded 16.5 s for a
    load that takes 1.8 s without cash, and a later hit credited all of it as
    saved.
    """
    return _tracking_seconds


# What feeds a tracker: the audit events Python raises for its own opens and
# listings, and wrappers on the readers that raise none, installed while one
# is open.
subscribe_read_events()
io_watch.add_patcher(install_patches, remove_patches)
