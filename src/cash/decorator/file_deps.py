"""Files as inputs: the ones ``file_depends_on=`` declares, the ones a body
read, and whether an entry's files are still as recorded."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._clock import perf_counter as _perf_counter
from .._paths import normalize_path
from ..exceptions import CashCacheIneffectiveWarning, CashCacheStoreFailedWarning
from ..remote_source import measured_validation, validation_is_expensive, warn_validation_cost_once
from ..source_norm import own_source_digest
from ..tracking.file_dep_snapshot import (
    attach_code_relative,
    dep_path_for_this_process,
    snapshot_dependencies,
    snapshot_is_fresh,
)
from ..tracking.read_credit import credited_reads
from ..tracking.tracker_context import active_tracker
from .cache_metadata import CacheMetadata
from .code_identity import CODE_KEYED_STATS

if TYPE_CHECKING:
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
    )
    # A file beside the function's own code is part of this INSTALL, not a
    # fixed location: record where it sits relative to the code, so another
    # install or release checks its own copy.
    return attach_code_relative(deps, code_module) or None


def propagate_file_deps_to_active_tracker(metadata: CacheMetadata) -> None:
    """Register this entry's recorded deps with the enclosing
    ``FileAccessTracker`` (if any), so a cached function that calls this
    one on a *hit* still inherits its dependencies. Best-effort: any
    failure (no tracker active, import issue) is silently ignored."""
    snap = getattr(metadata, "auto_file_deps", None)
    if not snap:
        return
    try:
        tracker = active_tracker.get()
    except Exception:  # noqa: BLE001 - tracking is best-effort
        return
    if tracker is None:
        return
    for path, recorded in snap.items():
        # A remote entry must go back onto the remote channel: routed to
        # ``add_tracked`` it would enter the file set, be stat'ed, and be
        # dropped - so the outer entry would silently lose the dependency.
        if isinstance(recorded, dict) and recorded.get("remote"):
            tracker.add_tracked_remote(path)
        else:
            # The file THIS process would read -- another install's copy
            # would give the enclosing entry the writer's path.
            # The hit just checked this file against the recorded hash, so
            # that hash is the file as it is: no second read to take it.
            digest = recorded.get("hash") if isinstance(recorded, dict) else None
            tracker.add_tracked(dep_path_for_this_process(path, recorded), digest)


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


class FileDeps:
    """The files a cached call depends on: declared with ``file_depends_on=``,
    and read by the body or its helpers."""

    def __init__(self, registry: FunctionRegistry, notices: Notices) -> None:
        self._registry = registry
        self._notices = notices

    def fold_declared_files(self, func_name: str, state_hash: str) -> str:
        """Fold which files ``file_depends_on=`` names, as written, into the key.

        Their content is checked against the entry on lookup
        (`FileDeps.track_declared_files`); this is what makes adding, removing or
        re-pointing one a different key, since an entry that recorded file A
        would otherwise keep hitting after the declaration moved to file B.
        As written rather than absolute, so a relative path keys the same on
        every machine.
        """
        cf = self._registry.cached.get(func_name)
        declared = cf.declared_files if cf is not None else ()
        if not declared:
            return state_hash
        names = json.dumps(sorted(raw for raw, _ in declared))
        return hashlib.sha256(f"{state_hash}:files:{names}".encode()).hexdigest()

    def track_declared_files(self, tracker: Any, func_name: str) -> None:
        """Record *func_name*'s ``file_depends_on=`` paths on *tracker*.

        As reads, so the entry snapshots their content and every lookup checks
        it with ``file_dep_is_fresh``, exactly like a file the body opened: a
        ``touch`` does not recompute, and an edit that keeps the mtime does.
        Called inside the timed body, so the content hash is taken off the body
        time with the tracker's other read hashes.
        """

        cf = self._registry.cached.get(func_name)
        for _, path in cf.declared_files if cf is not None else ():
            if os.path.exists(path):
                tracker.add_tracked(normalize_path(os.path.realpath(path)))
            else:
                tracker.add_tracked_absent(normalize_path(path))

    def auto_file_deps_fresh(self, metadata: CacheMetadata) -> bool:
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
        the two subsystems can't drift. ``(mtime, size)`` alone was ambiguous in
        both directions: a touch (identical content, bumped mtime)
        recomputed needlessly, and a same-size edit under an indistinguishable
        mtime was missed and served stale. The helper checks the cheap size
        first and only hashes when the size matches.
        """
        snap = metadata.auto_file_deps or {}
        if not snap:
            return True  # nothing to check

        # Remote entries cost a network round trip each to check, so the check
        # itself is worth measuring - see _warn_if_validation_is_expensive.
        #
        # Local ones are measured too, as what is left of the pass once the
        # remote resolutions are taken out. Hashing is not free either, and
        # file deps PROPAGATE: an aggregate that calls ten cached functions
        # inherits their inputs, so a fifty-file pipeline paid for fifty checks
        # on every one of those hits. Measured at 168 ms a hit before the
        # digest memo landed, with nothing anywhere to say so -- the remote
        # channel had a cost warning and the local one, which every user has,
        # did not.
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
            "read by a callee whose deps the aggregates do not inherit. Note "
            "that files above file_hash_full_max_bytes are sampled rather "
            "than hashed in full, which is cheaper per file but not per file "
            "COUNT.",
        )

    def credit_remembered_reads(self, func_name: str, tracker: Any, args: tuple, kwargs: dict) -> None:
        """Add the files a helper read in an EARLIER call to this call's inputs.

        A parse memoised with ``functools.lru_cache`` or a module dict: the
        first cached consumer read the file and recorded it; the second got
        the memoised rows, read nothing, and stored ``file_deps: None`` -- so
        after the file changed it kept serving the old total.

        For each function this call's code reaches that did NOT read a file in
        this call, its remembered files are added (`credited_reads`). A memo
        keyed by a path the call was given (``parse(path)``) adds only that
        path when it is among them; a memo of a fixed file adds what it read.
        The cached function's own history is left out -- it is per argument --
        and so is a function that read too many files to attribute.

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
        own = getattr(func, "__code__", None)
        have = tracker.get_accessed_files()
        arg_paths: set[str] | None = None
        for fn in self._registry.code_functions(func, func_name):
            code = getattr(fn, "__code__", None)
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
        between, the result of one version was stored under the other's key,
        and a later process running the first version was served the second
        one's numbers (a helper edited while a pooled call ran; a deploy that
        replaced a helper under a running job). Nothing can say
        which version the result came from, so it is returned and not stored.

        Only THIS code's text counts: a file edited elsewhere -- another
        function, a comment -- runs the same code in a new worker, and the
        entry is still right.

        One ``os.stat`` per code file, on a miss only; the text is re-read only
        for a file that moved.
        """
        stats: dict[str, tuple[int, int] | None] = {}
        moved: list[str] = []
        for fn in self._registry.code_functions(func, func_name):
            code = getattr(fn, "__code__", None)
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
        rewritten after the body read it but before it returned was
        fingerprinted in its new state, so the entry matched the new file
        and served the old answer on every later call (a sync job overlapping
        a long pipeline; and, one level up, an outer
        aggregate re-fingerprinting a file its inner call had already read).
        The documented mitigation -- write to a temp file and rename -- did
        not help, because the rename lands before the store.

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
            "the function writes a file it also reads, that is why: split the "
            "read and the write.",
        )
        return True
