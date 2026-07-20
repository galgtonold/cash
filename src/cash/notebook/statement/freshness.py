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
from ..file_dep_snapshot import file_dep_is_fresh
from ._metadata import StatementCacheMetadata

if TYPE_CHECKING:
    from .._protocols import TrackingState

logger = logging.getLogger(__name__)


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

    def check_cache(
        self,
        tracking_state: 'TrackingState',
        cache_key: str,
        ttl: int | None,
        inputs: set[str] | None = None,
    ) -> tuple['StatementCacheMetadata | None', Any | None, float]:
        """Look up *cache_key* and run freshness checks.

        Returns ``(metadata, cached_data, check_time)``.  When the entry
        is stale, ``cached_data`` is None and :attr:`last_miss_reason`
        carries a short human-readable explanation.
        """
        # Reset per-lookup miss attribution. The invalidator helpers may
        # overwrite this with a more specific reason on the way out.
        self.last_miss_reason = None

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
        for fpath, stored in file_deps.items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                self.last_miss_reason = f"file dependency missing: {fpath}"
                if self.debug:
                    logger.debug("[CACHE DEBUG] File dependency missing: %s", fpath)
                return None
            # Content is authoritative when the size matches; a bare size/mtime
            # check both over-invalidates on a touch (CAS-98) and misses a
            # same-size sub-resolution edit (CAS-10). See file_dep_is_fresh.
            is_fresh, reason = file_dep_is_fresh(resolved, stored)
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
        return cached_data

    def _input_file_changed(self, tracking_state: 'TrackingState', input_var: str, fpath: str) -> bool:
        """Return True if *fpath* (a dep of *input_var*) has been modified since it was cached.

        The producer's PERSISTED snapshot is the authority on what actually
        counts as a dependency, and it must be consulted before *fpath* is
        judged missing (CAS-185). ``tracking_state.executed_file_deps`` is a
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
        source_cache_key = tracking_state.variable_sources.get(input_var)
        if not source_cache_key:
            return False
        raw_source_meta, _ = self._backend.get(source_cache_key)
        if not raw_source_meta:
            return False
        source_meta = StatementCacheMetadata.from_dict(raw_source_meta)
        source_file_deps = source_meta.file_dependencies or {}
        if fpath not in source_file_deps:
            return False
        resolved = resolve_file_dep_path(fpath)
        if resolved is None:
            self.last_miss_reason = f"input file missing (via {input_var}): {fpath}"
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' file dependency missing: %s", input_var, fpath)
            return True
        # Same content-authoritative freshness as the direct-dep check.
        is_fresh, reason = file_dep_is_fresh(resolved, source_file_deps[fpath])
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
        for input_var in inputs:
            for fpath in tracking_state.executed_file_deps.get(input_var, ()):
                if self._input_file_changed(tracking_state, input_var, fpath):
                    return None
        return cached_data
