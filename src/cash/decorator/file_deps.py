"""Files as inputs: the ones ``file_depends_on=`` declares, the ones a body
read, and whether an entry's files are still as recorded."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._clock import perf_counter as _perf_counter
from .._paths import normalize_path
from ..code_digest import own_source_digest
from ..exceptions import CashCacheIneffectiveWarning, CashCacheStoreFailedWarning
from ..file_source import FileDataSource
from ..remote_source import (
    RemoteFileDataSource,
    measured_validation,
    remember_read_options,
    validation_is_expensive,
    warn_validation_cost_once,
)
from ..tracking.file_dep_snapshot import (
    attach_code_relative,
    dep_path_for_this_process,
    snapshot_dependencies,
    snapshot_is_fresh,
)
from ..tracking.read_credit import credited_reads
from ..tracking.tracker_context import active_tracker
from .cache_metadata import CacheMetadata
from .function_identity import CODE_KEYED_STATS

if TYPE_CHECKING:
    from .cached_function import CachedFunction
    from .registry import FunctionRegistry
    from .reporting import Notices

logger = logging.getLogger(__name__)


def snapshot_tracked_deps(tracker: Any, code_module: str | None = None) -> dict[str, dict[str, Any]] | None:
    """Snapshot everything *tracker* saw this call read - local and remote.

    Both land in one dict: they answer the same question ("did what this
    call read change since?"), and every consumer already routes that
    question through ``file_dep_is_fresh``, which branches on the entry.
    Remote entries cost one metadata request each to snapshot; that is the
    price of the read being tracked at all, and it is small against the
    download the entry exists to avoid.
    """

    read_stats = getattr(tracker, "read_stats", {})
    hashed_at = getattr(tracker, "read_hashed_at", {})
    known = {
        path: (read_stats[path], digest, hashed_at.get(path))
        for path, digest in getattr(tracker, "read_digests", {}).items()
        if path in read_stats
    }
    deps = snapshot_dependencies(
        tracker.get_accessed_files(),
        tracker.get_accessed_remote_urls(),
        tracker.get_absent_files(),
        known=known,
        unresolved=getattr(tracker, "unresolved_files", None),
        present=getattr(tracker, "present_files", None),
    )
    # A file beside the function's own code is part of this INSTALL, not a
    # fixed location: record where it sits relative to the code, so another
    # install or release checks its own copy.
    return attach_code_relative(deps, code_module) or None


def propagate_file_deps_to_active_tracker(metadata: CacheMetadata, func_name: str) -> None:
    """Register this entry's recorded deps with the enclosing
    ``FileAccessTracker`` (if any), so a cached function that calls this
    one on a *hit* still inherits its dependencies. Best-effort: any
    failure (no tracker active, import issue) is silently ignored.

    The enclosing tracker also learns that *func_name* was served from an
    entry: those files are all this call brings it, so what *func_name*'s code
    read for OTHER entries is not credited to it as a memo's reads
    (`FileDeps.credit_remembered_reads`)."""
    try:
        tracker = active_tracker.get()
    except Exception:  # noqa: BLE001 - tracking is best-effort
        return
    if tracker is None:
        return
    note_served = getattr(tracker, "note_served", None)
    if note_served is not None:
        note_served(func_name)
    snap = getattr(metadata, "auto_file_deps", None)
    if not snap:
        return
    for path, recorded in snap.items():
        # A remote entry must go back onto the remote channel: routed to
        # ``add_tracked`` it would enter the file set, be stat'ed, and be
        # dropped - so the outer entry would silently lose the dependency.
        if isinstance(recorded, dict) and recorded.get("remote"):
            if recorded.get("options"):
                # The store this entry's read went to, for the outer entry's check.
                remember_read_options(path, recorded["options"])
            tracker.add_tracked_remote(path)
        elif isinstance(recorded, dict) and recorded.get("absent"):
            # Looked for and missing: still an absence, not a file to stat.
            tracker.add_tracked_absent(dep_path_for_this_process(path, recorded))
        elif isinstance(recorded, dict) and "present" in recorded:
            tracker.add_tracked_present(dep_path_for_this_process(path, recorded), recorded["present"])
        else:
            # The file THIS process would read -- another install's copy
            # would give the enclosing entry the writer's path.
            # The hit just checked this file against the recorded hash, so
            # that hash is the file as it is: no second read to take it.
            digest = recorded.get("hash") if isinstance(recorded, dict) else None
            tracker.add_tracked(dep_path_for_this_process(path, recorded), digest)


def note_unentered_body(func_name: str) -> None:
    """Tell the enclosing tracker (if any) that the body of the cached
    *func_name* is running with no entry of its own, so its code's remembered
    reads are credited to the enclosing call as any helper's are."""
    try:
        tracker = active_tracker.get()
    except Exception:  # noqa: BLE001 - tracking is best-effort
        return
    note = getattr(tracker, "note_unentered", None)
    if note is not None:
        note(func_name)


def warn_if_validation_is_expensive(validation: Any, metadata: CacheMetadata) -> None:
    """Say so when checking freshness costs a serious share of the saving.

    A freshness check that has to ask the network is the one overhead a user
    cannot see: it happens on the HIT path, where the badge shows a saving
    and nothing shows what the saving cost to establish.
    """
    if not validation.count:
        return

    saved = metadata.execution_time
    if validation_is_expensive(validation.seconds, saved):
        warn_validation_cost_once(
            metadata.func_name or "a cached call",
            validation.count,
            validation.seconds,
            saved,
        )


def argument_paths(args: tuple, kwargs: dict) -> set[str]:
    """The resolved paths among a call's arguments, one container deep."""

    values: list[Any] = [*args, *kwargs.values()]
    for value in list(values):
        if isinstance(value, (list, tuple)) and len(value) <= 64:
            values.extend(value)
    found: set[str] = set()
    for value in values:
        if isinstance(value, os.PathLike) or (isinstance(value, str) and 0 < len(value) < 1024 and "\n" not in value):
            try:
                found.add(normalize_path(os.path.realpath(os.fspath(value))))
            except (TypeError, ValueError, OSError):
                continue
    return found


def _track_one(tracker: Any, path: str) -> None:
    """Record the existing *path* by its real path and, when it is relative,
    also as written, as a body's ``open`` of it is: that spelling is resolved
    against the working directory of each lookup, so a lookup from another
    directory checks the file there, not the one this call found."""
    tracker.add_tracked(normalize_path(os.path.realpath(path)))
    if not os.path.isabs(path):
        tracker.add_tracked(normalize_path(os.path.normpath(path)))


def _track_directory(tracker: Any, path: str) -> None:
    """A declared directory: every file under it by content, and every
    directory in it by its listing, so an edit, a new file and a removed one
    all count. The directory's own timestamp does not move when a file in
    it is edited, so it alone would not do."""
    for root, dirs, files in os.walk(path):
        dirs.sort()
        _track_one(tracker, root)
        for name in sorted(files):
            _track_one(tracker, os.path.join(root, name))


def _track_pattern(tracker: Any, pattern: str) -> None:
    """A declared glob (``data/*.csv``): each match by content, and the
    directories the matches were listed from, so a new match counts. It was
    recorded as a file of that literal name, which never exists, so nothing
    ever invalidated the entry."""
    listed = {_glob_base(pattern)}
    for match in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isdir(match):
            _track_directory(tracker, match)
            continue
        listed.add(os.path.dirname(match) or ".")
        _track_one(tracker, match)
    for directory in sorted(listed):
        if os.path.isdir(directory):
            _track_one(tracker, directory)


def _track_declared(tracker: Any, path: str) -> None:
    """Record a declared path on *tracker*: a pattern's matches, a
    directory's files, a file, or its absence. A relative one is resolved in
    the working directory of this call and recorded as written too
    (`_track_one`)."""
    if glob.has_magic(path):
        _track_pattern(tracker, path)
    elif os.path.isdir(path):
        _track_directory(tracker, path)
    elif os.path.exists(path):
        _track_one(tracker, path)
    else:
        tracker.add_tracked_absent(normalize_path(os.path.abspath(path)))
        if not os.path.isabs(path):
            tracker.add_tracked_absent(normalize_path(os.path.normpath(path)))


def pass_dynamic_sources_up(sources: list[tuple[Any, str]], resolutions: dict[tuple, Any] | None = None) -> None:
    """Make the cached calls around this one depend on the sources its
    ``dynamic_depends_on=`` resolved to, each with the token the key took.

    They are folded into this call's own key, which an enclosing cached
    caller never sees: ``report()`` calling ``load("data.txt")`` kept serving
    its old result after ``data.txt`` changed, while ``load`` recomputed. A
    file -- local or remote -- is recorded on the enclosing tracker as a read,
    so the caller's entry checks it on every lookup, as for a file
    ``file_depends_on=`` names. Any other source is recorded with its token:
    the caller's entry keeps both and asks the source again on every lookup
    (`dynamic_sources_fresh`). Called while the key is built, on a hit as on
    a miss, when the active tracker is the caller's, and when a caller's own
    entry is served inside another cached call (`CallRunner._try_get_cached`).
    *resolutions* are the resolver calls behind them, by key, which this
    process asks again on a lookup of the caller (`dynamic_sources.Resolution`).
    """
    tracker = active_tracker.get()
    if tracker is None:
        return
    if resolutions:
        add_resolution = getattr(tracker, "add_dynamic_resolution", None)
        if add_resolution is not None:
            for key, resolution in resolutions.items():
                add_resolution(key, resolution)
    for source, token in sources:
        if isinstance(source, FileDataSource):
            _track_declared(tracker, source.filepath)
        elif isinstance(source, RemoteFileDataSource):
            if source.storage_options:
                remember_read_options(source.url, source.storage_options)
            tracker.add_tracked_remote(source.url)
        else:
            add = getattr(tracker, "add_dynamic_source", None)
            if add is not None:
                add(source.get_id(), source, token)


def note_unresolved_dynamic(func_name: str) -> None:
    """Tell the cached calls around this one that *func_name*'s
    ``dynamic_depends_on=`` resolver failed: what it depends on is unknown,
    so they are not stored (`ResultStore.refusal`). Recorded only when it
    failed, it left the caller's entry with no record of the dependency,
    served for good once the resolver worked again."""
    tracker = active_tracker.get()
    add = getattr(tracker, "add_unresolved_dynamic", None)
    if add is not None:
        add(func_name)


def _glob_base(pattern: str) -> str:
    """The deepest directory of *pattern* with no wildcard in it."""
    parts = pattern.replace("\\", "/").split("/")
    base: list[str] = []
    for part in parts[:-1]:
        if glob.has_magic(part):
            break
        base.append(part)
    return "/".join(base) or ("/" if pattern.startswith("/") else ".")


def _code_of(fn: Any) -> types.CodeType | None:
    """*fn*'s code object, or None. A class whose metaclass answers every
    attribute (a catch-all ``__getattr__``) hands back a value for
    ``__code__`` too; that is not code."""
    code = getattr(fn, "__code__", None)
    return code if isinstance(code, types.CodeType) else None


class FileDeps:
    """The files a cached call depends on: declared with ``file_depends_on=``,
    and read by the body or its helpers."""

    def __init__(self, registry: FunctionRegistry, notices: Notices) -> None:
        self._registry = registry
        self._notices = notices

    def fold_declared_files(self, cf: CachedFunction, state_hash: str) -> str:
        """Fold which files ``file_depends_on=`` names, as written, into the key.

        Their content is checked against the entry on lookup
        (`FileDeps.track_declared_files`); this is what makes adding, removing or
        re-pointing one a different key, since an entry that recorded file A
        would otherwise keep hitting after the declaration moved to file B.
        As written rather than absolute, so a relative path keys the same on
        every machine.
        """
        declared = cf.declared_files
        if not declared:
            return state_hash
        names = json.dumps(sorted(declared))
        return hashlib.sha256(f"{state_hash}:files:{names}".encode()).hexdigest()

    def track_declared_files(self, tracker: Any, cf: CachedFunction) -> None:
        """Record *cf*'s ``file_depends_on=`` paths on *tracker*.

        As reads, so the entry snapshots their content and every lookup checks
        it with ``file_dep_is_fresh``, exactly like a file the body opened: a
        ``touch`` does not recompute, and an edit that keeps the mtime does.
        Called inside the timed body, so the content hash is taken off the body
        time with the tracker's other read hashes.

        A relative path is resolved against the working directory of this
        call, as the body's own ``open`` of it would be, and is recorded as
        written too (`_track_one`): resolved once at decoration, a run from
        another directory checked, and was served, the first directory's file.
        """

        for path in cf.declared_files:
            _track_declared(tracker, path)

    def auto_file_deps_fresh(self, metadata: CacheMetadata, *, quiet: bool = False) -> bool:
        """Return True if every file recorded in ``metadata.auto_file_deps``
        still matches on disk.

        Auto-tracked deps are captured during the first compute via
        `cash.tracking.file_tracker.FileAccessTracker` and stored as
        ``{path: {'mtime': float, 'size': int, 'hash': str}}``. If a recorded
        path is gone or its content changed, we invalidate the cache so the next
        compute re-reads the file. A path that disappears is also a change.

        Freshness is decided by the shared
        :func:`cash.tracking.file_dep_snapshot.snapshot_is_fresh` - the same
        content-authoritative check the notebook path uses, so
        the two subsystems can't drift. ``(mtime, size)`` alone is ambiguous in
        both directions: a touch (identical content, bumped mtime) would
        recompute needlessly, and a same-size edit under an indistinguishable
        mtime would be missed. The helper checks the cheap size first and
        only hashes when the size matches.

        *quiet* checks without saying that checking was expensive.
        """
        snap = metadata.auto_file_deps or {}
        if not snap:
            return True  # nothing to check
        if quiet:
            return snapshot_is_fresh(snap)[0]

        # Remote entries cost a network round trip each to check, so the check
        # itself is worth measuring - see warn_if_validation_is_expensive.
        #
        # Local ones are measured too, as what is left of the pass once the
        # remote resolutions are taken out. Hashing is not free either, and
        # file deps PROPAGATE: an aggregate that calls ten cached functions
        # inherits their inputs, so a fifty-file pipeline pays for fifty
        # checks on every one of those hits, and the user is told when that
        # costs a real share of the saving.
        started = _perf_counter()
        with measured_validation() as validation:
            fresh, stale = snapshot_is_fresh(snap)
        local_seconds = max(0.0, _perf_counter() - started - validation.seconds)
        local_count = sum(
            1 for recorded in snap.values() if not (isinstance(recorded, dict) and recorded.get("remote"))
        )
        if stale is not None:
            logger.debug("[FILE_DEP] stale (%s): %s", stale.reason, stale.path)
        warn_if_validation_is_expensive(validation, metadata)
        self._warn_if_local_validation_is_expensive(local_seconds, local_count, metadata)
        return fresh

    def _warn_if_local_validation_is_expensive(
        self,
        seconds: float,
        count: int,
        metadata: CacheMetadata,
    ) -> None:
        """Say so when hashing this entry's own files costs a real share of the
        saving.

        The same rule the remote channel uses (``validation_is_expensive``):
        more than half the compute it avoids past a 0.25 s floor, or more than
        2 s outright. Shared deliberately -- "proving it fresh cost more than
        recomputing would" is one judgement, and it should not depend on whether
        the input was a file or a URL.

        After the first check of a file this is microseconds (the digest is
        memoized per process), so reaching the threshold means many
        dependencies, very large ones, or a slow filesystem. Each of those is
        something the user can act on, and none of them shows up anywhere else.
        """
        if not count or not seconds:
            return

        saved = metadata.execution_time
        if not validation_is_expensive(seconds, saved):
            return
        label = metadata.func_name or "a cached call"
        against = f", against {saved:.2f}s of compute it avoids" if saved and saved > 0 else ""
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            label,
            "local-freshness-cost",
            f"cash spent {seconds:.2f}s checking {count} tracked "
            f"{'file' if count == 1 else 'files'} for freshness on {label}"
            f"{against}, so proving the result fresh costs a serious share of "
            f"what it saves.",
            code="CACHE-FRESHNESS-COST",
            fix="depend on fewer or smaller files -- cache a summary rather than "
            "every input -- or split the function so the expensive inputs are "
            "read by a callee whose deps the aggregates do not inherit.",
        )

    def credit_remembered_reads(self, func_name: str, tracker: Any, args: tuple, kwargs: dict) -> None:
        """Add the files a helper read in an EARLIER call to this call's inputs.

        A parse memoised with ``functools.lru_cache`` or a module dict: the
        first cached consumer read the file and recorded it; the second got
        the memoised rows, read nothing, and stored ``file_deps: None`` -- so
        after the file changed it kept serving the old total.

        For each function this call's code reaches that did NOT read a file in
        this call, its remembered files are added (`credited_reads`). A cached
        function reached only as entries served from the cache is not followed:
        each served entry already brought its own files, and what its code read
        for OTHER entries is not this call's input. Once its body ran here with
        no entry of its own (no key, a stream finished from the function), it
        is followed again. A memo
        keyed by a path the call was given (``parse(path)``) adds only that
        path when it is among them; a memo of a fixed file adds what it read.
        The cached function's own history is left out -- it is per argument.

        A remembered read also says which version of the file it was. When the
        file has changed since, the memo handed this call the OLD version's
        data -- right for this process until it refills, but not an answer
        for the file as it is now, which is what the entry would be stored
        against. Such a path goes into ``stale_memo_reads``, and the
        store is refused.
        """

        func = self._registry.functions.get(func_name)
        if func is None or tracker is None:
            return
        live = getattr(tracker, "reading_codes", set())
        own = _code_of(func)
        have = tracker.get_accessed_files()
        arg_paths: set[str] | None = None
        served = getattr(tracker, "served_functions", set()) - getattr(tracker, "unentered_functions", set())
        for fn in self._registry.code_functions(func, func_name, served):
            code = _code_of(fn)
            if code is None or code is own or code in live:
                continue
            remembered = credited_reads(code)
            if not remembered or remembered.keys() <= have:
                continue
            if arg_paths is None:
                arg_paths = argument_paths(args, kwargs)
            # Chosen BEFORE what is already tracked is taken away: `have` grows
            # as files are added, and a remainder that misses the arguments
            # would read as a memo of a fixed file.
            chosen = (remembered.keys() & arg_paths) or set(remembered)
            for path in sorted(chosen - have):
                tracker.add_tracked(path)
                then = remembered[path]
                if then is not None and tracker.read_stats.get(path, then) != then:
                    tracker.stale_memo_reads.add(path)

    def code_moved_since_keyed(self, func: Callable, func_name: str) -> bool:
        """Did a file this call's code came from change after its key was read?

        A function is keyed by the text of its file, read once per process;
        the code that runs is what the process loaded -- or, for a worker a
        pool starts during the call, whatever the file holds THEN. Edited in
        between (a helper edited while a pooled call runs; a deploy that
        replaces a helper under a running job), the result of one version
        would be stored under the other's key. Nothing can say which version
        the result came from, so it is returned and not stored.

        Only THIS code's text counts: a file edited elsewhere -- another
        function, a comment -- runs the same code in a new worker, and the
        entry is still right.

        One ``os.stat`` per code file, on a miss only; the text is re-read only
        for a file that moved.
        """
        stats: dict[str, tuple[int, int] | None] = {}
        moved: list[str] = []
        for fn in self._registry.code_functions(func, func_name):
            code = _code_of(fn)
            rec = CODE_KEYED_STATS.get(id(code)) if code is not None else None
            if rec is None or rec[0] is not code:
                continue
            path = rec[1]
            if path not in stats:
                try:
                    st = os.stat(path)
                    stats[path] = (st.st_size, st.st_mtime_ns)
                except OSError:
                    stats[path] = None
            now = stats[path]
            if now is None or now == (rec[2], rec[3]) or path in moved:
                continue
            # None -- the function is gone from the file -- is not the same text.
            same_text = own_source_digest(fn) == rec[4]
            if same_text:
                CODE_KEYED_STATS[id(code)] = (code, path, *now, rec[4])
            else:
                moved.append(path)
        if not moved:
            return False
        shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
        self._notices.warn_once(
            CashCacheStoreFailedWarning,
            func_name,
            "code_changed",
            f"@cash.cache on {func_name}: {shown} changed on disk after this "
            f"process read the code it keys {func_name} by. The result was "
            f"returned but not cached: a worker process started now runs the "
            f"file's new code, this process runs the old, and nothing can say "
            f"which one produced it.",
            code="STORE-CODE-CHANGED",
            fix="restart the process to run -- and cache -- the new code. A "
            "deploy that replaces files under a running job opens this window.",
        )
        return True

    def inputs_moved_during_call(self, func_name: str, tracker: Any) -> bool:
        """Did a file this call read change before the call returned?

        The entry's file fingerprints are taken when it is STORED. A file
        rewritten after the body read it but before it returned (a sync job
        overlapping a long pipeline; an outer aggregate re-fingerprinting a
        file its inner call had already read) would be fingerprinted in its
        new state, and the entry would match a file it was not computed
        from. Writing to a temp file and renaming does not avoid this: the
        rename lands before the store.

        The result is still returned: it is what the body computed. It is
        only not cached, because nothing can say which content it came from.
        """
        moved_fn = getattr(tracker, "inputs_changed_since_read", None)
        if moved_fn is None:
            return False
        try:
            moved = moved_fn()
        except Exception:  # noqa: BLE001 - never let the check break a call
            return False
        if not moved:
            return False
        shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
        self._notices.warn_once(
            CashCacheStoreFailedWarning,
            func_name,
            "input_changed",
            f"@cash.cache on {func_name}: {shown} changed while the call was "
            f"running, after it had been read. The result was returned but not "
            f"cached, because it cannot be told which version of the file it "
            f"was computed from.",
            code="STORE-INPUT-CHANGED",
            fix="nothing, if something else writes these files while this runs "
            "-- the next call reads the settled file and caches normally. If "
            "the function reads a file and then writes to it, that is why: "
            "split the read and the write.",
        )
        return True
