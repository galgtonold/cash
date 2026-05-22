"""Cache freshness checks for statement-level caching.

Owns the operation "is this cache entry still valid?" — TTL expiry,
direct file-dependency changes, and input-file change propagation.

Single public method: :meth:`CacheFreshnessChecker.check_cache`.  Five
private helpers (TTL, direct file, input file, etc.) plus two
module-level file-snapshot utilities used by sibling sub-systems.

**Anti-god-class rule (load-bearing):** this module is freshness
checks only.  It does **not** mutate :class:`TrackingState` and does
**not** write to the cache.  Its single output is "cached_data, or
None if stale, plus the metadata and a check-time measurement."  Side
effect: a ``last_miss_reason`` string is updated for the badge debug
display.

Distinct from [[Cacheability decision]] (which is the *pre-execution*
"should we cache this statement at all?" question).  This is the
*post-execution* "is the entry we already have still good?" question.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from ..utils import resolve_file_dep_path

if TYPE_CHECKING:
    from ._protocols import TrackingState
    from .statement_processor import StatementCacheMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-snapshot helpers — used by this module and (later) by StatementFileDeps
# ---------------------------------------------------------------------------

def snapshot_file_deps(paths: set[str]) -> dict[str, dict[str, float]]:
    """Return ``{path: {'mtime': mtime, 'size': size}}`` for paths that exist."""
    snapshot: dict[str, dict[str, float]] = {}
    for f in paths:
        try:
            st = os.stat(f)
        except OSError:
            continue
        snapshot[f] = {'mtime': st.st_mtime, 'size': st.st_size}
    return snapshot


def split_file_dep_value(value: Any) -> tuple[float, int | None]:
    """Return ``(mtime, size_or_None)`` for either the new or legacy form.

    New form: ``{'mtime': ..., 'size': ...}``.
    Legacy:   bare float (mtime only).  Cache entries from ≤0.5.0 use
    the legacy form; we still honour them but skip the size check.
    """
    if isinstance(value, dict):
        return float(value.get('mtime', 0.0)), value.get('size')
    return float(value), None


# ---------------------------------------------------------------------------
# CacheFreshnessChecker
# ---------------------------------------------------------------------------

class CacheFreshnessChecker:
    """Decide whether a cache entry is still fresh.

    Holds :class:`TrackingState` and the cache backend by reference;
    never mutates them.  ``last_miss_reason`` is updated as a
    side-effect whenever a check rejects an entry, so the badge can
    surface "why did this cell re-run?" in expanded row detail.
    """

    def __init__(
        self,
        tracking_state: 'TrackingState',
        backend: Any,
        debug: bool = False,
    ) -> None:
        self._backend = backend
        self.debug = debug
        self.last_miss_reason: str | None = None
        self.set_tracking_state(tracking_state)

    def set_tracking_state(self, state: 'TrackingState') -> None:
        """Re-wire individual dict references from *state*.

        Matches the alias-not-store pattern used throughout the codebase
        (``StatementProcessor``, ``NotebookSimulator``).  Tests can rebind
        these attributes on the orchestrator (e.g. ``processor.variable_sources
        = {...}``) and the freshness checker stays consistent via its own
        rebind below.
        """
        self.variable_sources = state.variable_sources
        self.executed_file_deps = state.executed_file_deps

    def check_cache(
        self,
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
        metadata, cached_data = self._backend.get(cache_key)
        cache_check_time = time.time() - t3

        if cached_data and metadata:
            if ttl:
                cached_data = self._invalidate_if_ttl_expired(metadata, cached_data, ttl)
            if cached_data:
                cached_data = self._invalidate_if_direct_file_changed(metadata, cached_data)
            # CRITICAL: also check file dependencies inherited from INPUT variables.
            # Fixes the bug where `df` cell was cached even when the source CSV changed.
            if cached_data and inputs:
                cached_data = self._invalidate_if_input_file_changed(inputs, cached_data)
            if cached_data is not None and 'output_lineages' not in metadata:
                cached_data = None
                self.last_miss_reason = "cache entry uses an old format (missing lineage metadata)"
                if self.debug:
                    logger.debug("[CACHE DEBUG] Cache entry missing lineage metadata (stale format), invalidating.")

        return metadata, cached_data, cache_check_time

    # ------------------------------------------------------------------
    # Private freshness checks
    # ------------------------------------------------------------------

    def _invalidate_if_ttl_expired(
        self, metadata: 'StatementCacheMetadata', cached_data: Any, ttl: int,
    ) -> Any:
        """Return None if the cache entry has exceeded *ttl* seconds, else return *cached_data*."""
        timestamp = metadata.get('timestamp', 0)
        age = time.time() - timestamp
        if age > ttl:
            self.last_miss_reason = f"cache TTL expired ({age:.0f}s old, limit {ttl}s)"
            if self.debug:
                logger.debug("[CACHE DEBUG] Cache expired (TTL)")
            return None
        return cached_data

    def _invalidate_if_direct_file_changed(
        self, metadata: 'StatementCacheMetadata', cached_data: Any,
    ) -> Any:
        """Return None if any direct file dep in *metadata* is missing or modified."""
        file_deps = metadata.get('file_dependencies', {})
        for fpath, stored in file_deps.items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                self.last_miss_reason = f"file dependency missing: {fpath}"
                if self.debug:
                    logger.debug("[CACHE DEBUG] File dependency missing: %s", fpath)
                return None
            stored_mtime, stored_size = split_file_dep_value(stored)
            try:
                cur_stat = os.stat(resolved)
            except OSError:
                self.last_miss_reason = f"file dependency unreadable: {resolved}"
                return None
            mtime_delta = abs(cur_stat.st_mtime - stored_mtime)
            if mtime_delta > 0.01:
                self.last_miss_reason = f"file changed: {resolved}"
                if self.debug:
                    logger.debug("[CACHE DEBUG] File dependency mtime changed: %s (delta=%.4fs)", resolved, mtime_delta)
                return None
            # Filesystems with coarse mtime granularity (HFS+/APFS, some
            # ext4 configs) can produce identical mtimes for back-to-back
            # writes.  Falling back to size catches that case for the
            # common "rewrote the CSV" scenario.
            if stored_size is not None and cur_stat.st_size != stored_size:
                self.last_miss_reason = f"file changed (size): {resolved}"
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] File dependency size changed: %s (was %d, now %d)",
                        resolved, stored_size, cur_stat.st_size,
                    )
                return None
        return cached_data

    def _input_file_changed(self, input_var: str, fpath: str) -> bool:
        """Return True if *fpath* (a dep of *input_var*) has been modified since it was cached."""
        resolved = resolve_file_dep_path(fpath)
        if resolved is None:
            self.last_miss_reason = f"input file missing (via {input_var}): {fpath}"
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' file dependency missing: %s", input_var, fpath)
            return True
        source_cache_key = self.variable_sources.get(input_var)
        if not source_cache_key:
            return False
        source_meta, _ = self._backend.get(source_cache_key)
        if not source_meta:
            return False
        source_file_deps = source_meta.get('file_dependencies', {})
        if fpath not in source_file_deps:
            return False
        stored_mtime, stored_size = split_file_dep_value(source_file_deps[fpath])
        try:
            cur_stat = os.stat(resolved)
        except OSError:
            return True
        mtime_delta = abs(cur_stat.st_mtime - stored_mtime)
        if mtime_delta > 0.01:
            self.last_miss_reason = f"file changed via input '{input_var}': {resolved}"
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' source file mtime changed: %s (delta=%.4fs)", input_var, resolved, mtime_delta)
            return True
        if stored_size is not None and cur_stat.st_size != stored_size:
            self.last_miss_reason = f"file changed (size) via input '{input_var}': {resolved}"
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' source file size changed: %s (was %d, now %d)", input_var, resolved, stored_size, cur_stat.st_size)
            return True
        return False

    def _invalidate_if_input_file_changed(self, inputs: set[str], cached_data: Any) -> Any:
        """Return None if any file dep of an input variable has changed since it was computed."""
        for input_var in inputs:
            for fpath in self.executed_file_deps.get(input_var, ()):
                if self._input_file_changed(input_var, fpath):
                    return None
        return cached_data
