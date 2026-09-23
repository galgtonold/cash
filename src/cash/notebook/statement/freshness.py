"""Cache freshness checks for statement-level caching.

Owns the operation "is this cache entry still valid?" — TTL expiry,
direct file-dependency changes, and input-file change propagation.

Single public method: :meth:`CacheFreshnessChecker.check_cache`.

**Anti-god-class rule (load-bearing):** this module is freshness
checks only.  It does **not** mutate :class:`TrackingState` and does
**not** write to the cache.  Its single output is "cached_data, or
None if stale, plus the metadata and a check-time measurement."  Side
effect: a ``last_miss_reason`` string is updated for the badge debug
display.

Distinct from [[Cacheability decision]] (which is the *pre-execution*
"should we cache this statement at all?" question).  This is the
*post-execution* "is the entry we already have still good?" question.

The file-snapshot helper (`snapshot_file_deps`) lives in
``cash.tracking.file_dep_snapshot``, not here: it has cross-cluster callers
(ADR-011).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ...backends._base import ttl_expired
from ...tracking.file_dep_snapshot import FreshnessMemo, snapshot_is_fresh
from ..call_refs import resolve_call_refs
from ._metadata import StatementCacheMetadata

if TYPE_CHECKING:
    from .._protocols import TrackingState

logger = logging.getLogger(__name__)

#: "Look the producer's file deps up yourself" (``None`` means it has none).
_UNSET = object()


#: How long a cell's statements may share the answer for a file (``forget_file_answers``).
_ANSWERS_LAST_S = 2.0

#: Below this many dependencies the per-file answers are cheap enough.
_SET_MEMO_MIN = 64


class CacheFreshnessChecker:
    """Decide whether a cache entry is still fresh.

    Holds the cache backend by reference and a transient
    ``last_miss_reason`` for badge attribution.  All :class:`TrackingState`
    access happens through the ``tracking_state`` method parameter — no
    aliased dict references on this instance.
    """

    def __init__(
        self,
        backend: Any,
    ) -> None:
        self._backend = backend
        self.last_miss_reason: str | None = None
        self._memo = FreshnessMemo()
        #: Dependency sets verified fresh whole, while the answers above last.
        self._fresh_sets: list[dict] = []
        self._epoch: Any = None
        self._answered_at = 0.0

    def forget_file_answers(self, epoch: Any = None) -> None:
        """Check files afresh from here on.

        One answer per (path, snapshot) holds for as long as nothing could
        have changed a file: until a statement executes, or the next cell. A
        cell's statements reading the same 10,000 documents re-checked every
        one of them per statement: 120,000 checks, 1.1 s,
        for a cell served entirely from the cache.

        Only inside a cell run (an ``int`` *epoch*: ``file_state_epoch``,
        which every cell run moves) and for at most ``_ANSWERS_LAST_S``: a
        file another process or a background thread rewrites mid-cell is seen
        within that, and a caller outside any cell gets one check per lookup.
        Not the shell's ``execution_count``: a cell run without history
        (``shell.run_cell(code)``) leaves it unchanged, and the next cell
        reused this one's answers -- a same-size edit made between the two,
        within the window, was served stale.
        """
        self._memo = FreshnessMemo()
        self._fresh_sets = []
        self._epoch = epoch
        self._answered_at = time.monotonic()

    def check_cache(
        self,
        tracking_state: "TrackingState",
        cache_key: str,
        ttl: int | None,
        inputs: set[str] | None = None,
        epoch: Any = None,
    ) -> tuple["StatementCacheMetadata | None", Any | None, float]:
        """Look up *cache_key* and run freshness checks.

        Returns ``(metadata, cached_data, check_time)``.  When the entry
        is stale, ``cached_data`` is None and :attr:`last_miss_reason`
        carries a short human-readable explanation.
        """
        # Reset per-lookup miss attribution. The invalidator helpers may
        # overwrite this with a more specific reason on the way out.
        self.last_miss_reason = None
        # A statement's own recorded deps include the ones it inherited from
        # its inputs, so the two passes below checked every file twice. One
        # answer per (path, snapshot) until a statement executes or the cell
        # changes (``forget_file_answers``); stats from directory listings too.
        if not isinstance(epoch, int) or epoch != self._epoch or time.monotonic() - self._answered_at > _ANSWERS_LAST_S:
            self.forget_file_answers(epoch)

        t3 = time.time()
        raw_metadata, cached_data = self._backend.get(cache_key)
        metadata = StatementCacheMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
        cache_check_time = time.time() - t3

        if cached_data and metadata:
            # `is not None`, not truthiness: ttl=0 is a REQUEST ("expire
            # immediately"), not an absence. Reading it as falsy skipped the
            # check and made the entry immortal — the inverse of the ask.
            if ttl is not None:
                cached_data = self._invalidate_if_ttl_expired(metadata, cached_data, ttl)
            if cached_data:
                cached_data = self._invalidate_if_direct_file_changed(metadata, cached_data)
            # CRITICAL: also check file dependencies inherited from INPUT variables.
            # Fixes the bug where `df` cell was cached even when the source CSV changed.
            if cached_data and inputs:
                cached_data = self._invalidate_if_input_file_changed(tracking_state, inputs, cached_data)
            if cached_data:
                # Call results the entry refers to rather than copies (call_refs).
                cached_data = resolve_call_refs(cached_data, self._backend)

        return metadata, cached_data, cache_check_time

    # ------------------------------------------------------------------
    # Private freshness checks
    # ------------------------------------------------------------------

    def _known_fresh(self, deps: dict) -> bool:
        """Was a set equal to *deps* verified fresh while the answers last?

        A loop body over a frame read from 5,000 files: every statement of
        every iteration carries the same 5,000 dependencies, and building a
        memo key per file per lookup was a million calls, 2.4 s of a 1.1 s
        cell. Comparing the whole set runs in C.
        """
        return len(deps) >= _SET_MEMO_MIN and any(
            len(known) == len(deps) and known == deps for known in self._fresh_sets
        )

    def _remember_fresh(self, deps: dict) -> None:
        if len(deps) >= _SET_MEMO_MIN and len(self._fresh_sets) < 16:
            self._fresh_sets.append(deps)

    def _invalidate_if_ttl_expired(
        self,
        metadata: "StatementCacheMetadata",
        cached_data: Any,
        ttl: int,
    ) -> Any:
        """Return None if the cache entry has exceeded *ttl* seconds, else return *cached_data*."""
        if ttl_expired(metadata.timestamp, ttl):
            age = time.time() - (metadata.timestamp or 0)
            self.last_miss_reason = f"cache TTL expired ({age:.0f}s old, limit {ttl}s)"
            logger.debug("[CACHE DEBUG] Cache expired (TTL)")
            return None
        return cached_data

    def _invalidate_if_direct_file_changed(
        self,
        metadata: "StatementCacheMetadata",
        cached_data: Any,
    ) -> Any:
        """Return None if any direct file dep in *metadata* is missing or modified."""
        file_deps = metadata.file_dependencies or {}
        if self._known_fresh(file_deps):
            return cached_data
        # Content is authoritative when the size matches; a bare size/mtime
        # check both over-invalidates on a touch and misses a same-size
        # sub-resolution edit. See file_dep_is_fresh.
        fresh, stale = snapshot_is_fresh(file_deps, self._memo)
        if not fresh:
            if stale.reason == "missing":
                self.last_miss_reason = f"file dependency missing: {stale.path}"
            elif stale.reason == "unreadable":
                self.last_miss_reason = f"file dependency unreadable: {stale.resolved}"
            elif stale.reason == "size":
                self.last_miss_reason = f"file changed (size): {stale.resolved}"
            else:
                self.last_miss_reason = f"file changed: {stale.resolved or stale.path}"
            logger.debug("[CACHE DEBUG] File dependency stale: %s", stale)
            return None
        self._remember_fresh(file_deps)
        return cached_data

    def _source_file_deps(self, tracking_state: "TrackingState", input_var: str) -> dict | None:
        """The file dependencies *input_var*'s producer recorded, or None.

        Read ONCE per input, and from metadata alone. Fetched through
        ``get()`` once per file, it deep-copied the producer's cached VALUE --
        for a frame read from 1,200 files, 1,200 copies of the whole frame on
        every lookup of every statement that read it: a 0.4 s cell took 42 s
        when served from the cache.
        """
        source_cache_key = tracking_state.variable_sources.get(input_var)
        if not source_cache_key:
            return None
        raw_source_meta = self._backend.peek_metadata(source_cache_key)
        if not raw_source_meta:
            return None
        return StatementCacheMetadata.from_dict(raw_source_meta).file_dependencies or {}

    def _input_file_changed(
        self,
        tracking_state: "TrackingState",
        input_var: str,
        fpath: str,
        source_file_deps: dict | None | object = _UNSET,
    ) -> bool:
        """Return True if *fpath* (a dep of *input_var*) has been modified since it was cached.

        The producer's PERSISTED snapshot is the authority on what actually
        counts as a dependency, and it must be consulted before *fpath* is
        judged missing. ``tracking_state.executed_file_deps`` is a
        strict superset of that snapshot: it records every path the tracker saw
        an access *attempt* for, including reads that raised (``tracked_open``
        records before it calls through), while ``snapshot_file_deps`` silently
        drops paths that could not be stat'd. Importing sklearn makes
        ``importlib.metadata`` probe for optional metadata that legitimately
        does not exist (``direct_url.json``, ``entry_points.txt``,
        ``pythonXY.zip`` on ``sys.path``); those land in the in-memory set and
        never in the snapshot. Judging them missing first invalidated the
        consumer of every such variable on every run, forever — a path that
        never existed cannot have *changed*.

        A dep that WAS snapshotted and has since been deleted is still caught:
        it is present in ``source_file_deps``, so it reaches the check below.
        """
        if source_file_deps is _UNSET:
            source_file_deps = self._source_file_deps(tracking_state, input_var)
        if not source_file_deps or fpath not in source_file_deps:
            return False
        return self._inherited_deps_changed(input_var, {fpath: source_file_deps[fpath]})

    def _inherited_deps_changed(self, input_var: str, deps: dict) -> bool:
        """True, with the miss reason set, when one of *deps* (inherited through
        *input_var*) is no longer as its producer recorded it. Same
        content-authoritative check as the direct dependencies."""
        fresh, stale = snapshot_is_fresh(deps, self._memo)
        if fresh:
            return False
        if stale.resolved is None:
            self.last_miss_reason = f"input file missing (via {input_var}): {stale.path}"
        else:
            size_note = " (size)" if stale.reason == "size" else ""
            self.last_miss_reason = f"file changed{size_note} via input '{input_var}': {stale.resolved}"
        logger.debug("[CACHE DEBUG] Input '%s' source file stale: %s", input_var, stale)
        return True

    def _invalidate_if_input_file_changed(
        self, tracking_state: "TrackingState", inputs: set[str], cached_data: Any
    ) -> Any:
        """Return None if any file dep of an input variable has changed since it was computed."""
        for input_var in inputs:
            paths = tracking_state.executed_file_deps.get(input_var, ())
            if not paths:
                continue
            source_file_deps = self._source_file_deps(tracking_state, input_var)
            if not source_file_deps or self._known_fresh(source_file_deps):
                continue
            # Only what the producer's snapshot recorded counts: see _input_file_changed.
            deps = {p: source_file_deps[p] for p in paths if p in source_file_deps}
            if self._inherited_deps_changed(input_var, deps):
                return None
            if paths and len(paths) >= len(source_file_deps) and all(p in paths for p in source_file_deps):
                self._remember_fresh(source_file_deps)
        return cached_data
