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

The two file-snapshot helpers (`snapshot_file_deps`, `split_file_dep_value`)
that used to live alongside `CacheFreshnessChecker` were extracted to
``cash.notebook.file_dep_snapshot`` before this module moved into the
``statement/`` package (ADR-011): they have cross-cluster callers and don't
belong inside ``statement/``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ...utils import resolve_file_dep_path
from ..file_dep_snapshot import (
    _LISTING_MIN_FILES,
    _full_hash_max_bytes,
    file_dep_is_fresh,
    stats_from_listings,
)
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
        debug: bool = False,
    ) -> None:
        self._backend = backend
        self.debug = debug
        self.last_miss_reason: str | None = None
        self._checked: dict = {}
        self._listed: dict = {}
        #: Dependency sets verified fresh whole, while the answers above last.
        self._fresh_sets: list[dict] = []
        self._epoch: Any = None
        self._answered_at = 0.0

    def forget_file_answers(self, epoch: Any = None) -> None:
        """Check files afresh from here on.

        One answer per (path, snapshot) holds for as long as nothing could
        have changed a file: until a statement executes, or the next cell. A
        cell's statements reading the same 10,000 documents re-checked every
        one of them per statement; on r24s4 that was 120,000 checks, 1.1 s,
        for a cell served entirely from the cache.

        Only inside a real cell (an ``int`` execution count) and for at most
        ``_ANSWERS_LAST_S``: a file another process or a background thread
        rewrites mid-cell is seen within that, and a caller outside any cell
        gets one check per lookup, as before.
        """
        self._checked = {}
        self._listed = {}
        self._fresh_sets = []
        self._epoch = epoch
        self._answered_at = time.monotonic()

    def check_cache(
        self,
        tracking_state: 'TrackingState',
        cache_key: str,
        ttl: int | None,
        inputs: set[str] | None = None,
        epoch: Any = None,
    ) -> tuple['StatementCacheMetadata | None', Any | None, float]:
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
        if (not isinstance(epoch, int) or epoch != self._epoch
                or time.monotonic() - self._answered_at > _ANSWERS_LAST_S):
            self.forget_file_answers(epoch)

        t3 = time.time()
        raw_metadata, cached_data = self._backend.get(cache_key)
        metadata = (
            StatementCacheMetadata.from_dict(raw_metadata)
            if raw_metadata is not None
            else None
        )
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

        return metadata, cached_data, cache_check_time

    # ------------------------------------------------------------------
    # Private freshness checks
    # ------------------------------------------------------------------

    def _resolve_and_check(self, fpath: str, stored: Any, full_hash_max: int | None):
        """``(resolved, is_fresh, reason)`` for one dependency, once per lookup."""
        # A tuple of the snapshot's items, not its repr: the key is built on
        # every call, answered or not, and a sorted repr was 1 s of r24s4's
        # 120,000 lookups against 5,000 real checks.
        try:
            memo_key = (fpath, tuple(stored.items()) if isinstance(stored, dict) else stored)
            hash(memo_key)
        except TypeError:
            try:
                memo_key = (fpath, repr(sorted(stored.items())) if isinstance(stored, dict) else repr(stored))
            except TypeError:
                memo_key = None
        checked = getattr(self, '_checked', None)
        if memo_key is not None and checked is not None and memo_key in checked:
            return checked[memo_key]
        answer = None
        if isinstance(stored, dict) and 'size' in stored:
            # A local snapshot checked where it was recorded first: the stat
            # that decides freshness also says the file is there, which is all
            # ``resolve_file_dep_path``'s ``exists`` was asking -- one syscall
            # per dependency per lookup instead of two (a re-run of statements
            # derived from 3,000 files made 72,000; round 23). A path that is
            # not there any more goes through the relocation fallbacks as before.
            listed = getattr(self, '_listed', None)
            is_fresh, reason = file_dep_is_fresh(
                fpath, stored, full_hash_max, listed.get(fpath) if listed else None)
            if reason != 'unreadable':
                answer = (fpath, is_fresh, reason)
        if answer is None:
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                answer = (None, False, 'missing')
            else:
                answer = (resolved, *file_dep_is_fresh(resolved, stored, full_hash_max))
        if memo_key is not None and checked is not None:
            checked[memo_key] = answer
        return answer

    def _known_fresh(self, deps: dict) -> bool:
        """Was a set equal to *deps* verified fresh while the answers last?

        A loop body over a frame read from 5,000 files: every statement of
        every iteration carries the same 5,000 dependencies, and building a
        memo key per file per lookup was a million calls, 2.4 s of a 1.1 s
        cell (round 25, r25s4). Comparing the whole set runs in C.
        """
        return len(deps) >= _SET_MEMO_MIN and any(
            len(known) == len(deps) and known == deps for known in self._fresh_sets)

    def _remember_fresh(self, deps: dict) -> None:
        if len(deps) >= _SET_MEMO_MIN and len(self._fresh_sets) < 16:
            self._fresh_sets.append(deps)

    def _invalidate_if_ttl_expired(
        self, metadata: 'StatementCacheMetadata', cached_data: Any, ttl: int,
    ) -> Any:
        """Return None if the cache entry has exceeded *ttl* seconds, else return *cached_data*."""
        timestamp = metadata.timestamp or 0
        age = time.time() - timestamp
        # ttl<=0 means "never fresh", decided without consulting the clock: a
        # same-tick re-read can measure age == 0.0 on a coarse timer, and
        # `0.0 > 0` would hand back the entry ttl=0 exists to reject.
        if ttl <= 0 or age > ttl:
            self.last_miss_reason = f"cache TTL expired ({age:.0f}s old, limit {ttl}s)"
            if self.debug:
                logger.debug("[CACHE DEBUG] Cache expired (TTL)")
            return None
        return cached_data

    def _invalidate_if_direct_file_changed(
        self, metadata: 'StatementCacheMetadata', cached_data: Any,
    ) -> Any:
        """Return None if any direct file dep in *metadata* is missing or modified."""
        file_deps = metadata.file_dependencies or {}
        if self._known_fresh(file_deps):
            return cached_data
        full_hash_max = _full_hash_max_bytes() if file_deps else None
        if len(file_deps) >= _LISTING_MIN_FILES:
            # Many files: read their directories once rather than stat each
            # (see ``stats_from_listings``). Taken at the first lookup that
            # needs them, as current as the stats they replace.
            unlisted = [p for p, s in file_deps.items()
                        if isinstance(s, dict) and 'size' in s and p not in self._listed]
            if len(unlisted) >= _LISTING_MIN_FILES:
                self._listed.update(stats_from_listings(unlisted))
        for fpath, stored in file_deps.items():
            # Content is authoritative when the size matches; a bare size/mtime
            # check both over-invalidates on a touch and misses a
            # same-size sub-resolution edit. See file_dep_is_fresh.
            resolved, is_fresh, reason = self._resolve_and_check(fpath, stored, full_hash_max)
            if resolved is None:
                self.last_miss_reason = f"file dependency missing: {fpath}"
                if self.debug:
                    logger.debug("[CACHE DEBUG] File dependency missing: %s", fpath)
                return None
            if not is_fresh:
                if reason == "unreadable":
                    self.last_miss_reason = f"file dependency unreadable: {resolved}"
                elif reason == "size":
                    self.last_miss_reason = f"file changed (size): {resolved}"
                else:
                    self.last_miss_reason = f"file changed: {resolved}"
                if self.debug:
                    logger.debug("[CACHE DEBUG] File dependency stale (%s): %s", reason, resolved)
                return None
        self._remember_fresh(file_deps)
        return cached_data

    def _source_file_deps(self, tracking_state: 'TrackingState', input_var: str) -> dict | None:
        """The file dependencies *input_var*'s producer recorded, or None.

        Read ONCE per input, and from metadata alone. Fetched through
        ``get()`` once per file, it deep-copied the producer's cached VALUE --
        for a frame read from 1,200 files, 1,200 copies of the whole frame on
        every lookup of every statement that read it: a 0.4 s cell took 42 s
        when served from the cache (round 23, r23s2).
        """
        source_cache_key = tracking_state.variable_sources.get(input_var)
        if not source_cache_key:
            return None
        peek = getattr(self._backend, 'peek_metadata', None)
        raw_source_meta = peek(source_cache_key) if peek is not None else self._backend.get(source_cache_key)[0]
        if not raw_source_meta:
            return None
        return StatementCacheMetadata.from_dict(raw_source_meta).file_dependencies or {}

    def _input_file_changed(
        self, tracking_state: 'TrackingState', input_var: str, fpath: str,
        source_file_deps: dict | None | object = _UNSET, full_hash_max: int | None = None,
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
        # Same content-authoritative freshness as the direct-dep check.
        resolved, is_fresh, reason = self._resolve_and_check(fpath, source_file_deps[fpath], full_hash_max)
        if resolved is None:
            self.last_miss_reason = f"input file missing (via {input_var}): {fpath}"
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' file dependency missing: %s", input_var, fpath)
            return True
        if not is_fresh:
            size_note = " (size)" if reason == "size" else ""
            self.last_miss_reason = f"file changed{size_note} via input '{input_var}': {resolved}"
            if self.debug:
                logger.debug(
                    "[CACHE DEBUG] Input '%s' source file stale (%s): %s",
                    input_var, reason, resolved,
                )
            return True
        return False

    def _invalidate_if_input_file_changed(self, tracking_state: 'TrackingState', inputs: set[str], cached_data: Any) -> Any:
        """Return None if any file dep of an input variable has changed since it was computed."""
        full_hash_max = None
        for input_var in inputs:
            paths = tracking_state.executed_file_deps.get(input_var, ())
            if not paths:
                continue
            source_file_deps = self._source_file_deps(tracking_state, input_var)
            if not source_file_deps or self._known_fresh(source_file_deps):
                continue
            if full_hash_max is None:
                full_hash_max = _full_hash_max_bytes()
            for fpath in paths:
                if self._input_file_changed(tracking_state, input_var, fpath,
                                            source_file_deps, full_hash_max):
                    return None
            if paths and len(paths) >= len(source_file_deps) and all(p in paths for p in source_file_deps):
                self._remember_fresh(source_file_deps)
        return cached_data
