"""The look-up phase of a statement: is there a fresh entry under its key?

:class:`StatementLookup` reads the entry of a statement whose key is built
(through :class:`CacheFreshnessChecker`, which also runs the freshness
checks), tells the perpetual-miss guard how the look-up went, and logs it.
It decides nothing about the key and writes nothing to the cache: what it
is given is the run with its key, and what it hands back is the entry's
metadata and payload, or two Nones.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from cash.control_markers import has_marker
from cash.notebook.statement.directives import ttl_floor_from_called_functions
from cash.tracking.file_dep_snapshot import file_state_epoch

if TYPE_CHECKING:
    from cash.notebook.statement._metadata import StatementCacheMetadata
    from cash.notebook.statement.freshness import CacheFreshnessChecker
    from cash.notebook.statement.miss_guard import MissGuard
    from cash.notebook.statement.records import StatementRecords
    from cash.notebook.statement.run import StatementRun
    from cash.notebook.tracking_state import TrackingState

# The processor's logger: these lines were its own before the look-up had
# an object to itself, and ``%cash_debug`` and the tests read them there.
logger = logging.getLogger("cash.notebook.statement.processor")

_LOG_ANNOTATION = "[ANNOTATION]"

__all__ = ["StatementLookup"]


class StatementLookup:
    """Looks a statement up under its key.

    *shell* is read for its namespace at each look-up (a kernel can swap
    it); *tracking_state* is the session's lineage store; *freshness* reads
    the entry and checks it; *miss_guard* is told each outcome; *records*
    says which lineages the inputs had, for the guard's "what kept changing".
    """

    def __init__(
        self,
        shell: Any,
        tracking_state: TrackingState,
        freshness: CacheFreshnessChecker,
        miss_guard: MissGuard,
        records: StatementRecords,
    ) -> None:
        self._shell = shell
        self._tracking_state = tracking_state
        self._freshness = freshness
        self._miss_guard = miss_guard
        self._records = records

    def lookup(
        self, run: StatementRun, analysis_time: float, hash_time: float
    ) -> tuple[StatementCacheMetadata | None, Any | None]:
        """Look *run* up in the cache: ``(metadata, cached_data)``, both None
        on a miss or a skipped lookup."""
        run.effective_ttl = ttl_floor_from_called_functions(run.inputs, run.effective_ttl, self._shell.user_ns)
        metadata, cached_data, cache_check_time = self.read_entry(
            run.skip_cache, run.cache_key, run.effective_ttl, run.inputs, run.metrics["uncacheable_reasons"]
        )
        if cached_data is None and not run.skip_cache:
            run.stale_file = self._freshness.last_stale_file
        self.observe(run.skip_cache, run.code, run.source_hash, run.cache_key, cached_data, run.inputs)

        if logger.isEnabledFor(logging.DEBUG):
            self._log(run.code, run.cache_key, run.inputs, cached_data, analysis_time, hash_time, cache_check_time)
        return metadata, cached_data

    def read_entry(
        self,
        skip_cache: bool,
        cache_key: str,
        ttl: int | None,
        inputs: set[str],
        reasons: Sequence[str] = (),
    ) -> tuple[StatementCacheMetadata | None, Any | None, float]:
        """Run cache lookup unless *skip_cache* is set; a skipped lookup is
        logged with the *reasons* the statement runs every time."""
        if not skip_cache:
            return self._freshness.check_cache(
                self._tracking_state, cache_key, ttl, inputs, epoch=file_state_epoch(), bound=self._shell.user_ns
            )
        logger.debug(
            "%s Skipping cache lookup: %s",
            _LOG_ANNOTATION,
            "; ".join(reasons) or "the statement runs every time (@cash:no-cache, or a statement cash never stores)",
        )
        return None, None, 0.0

    def observe(
        self,
        skip_cache: bool,
        code: str,
        source_hash: str,
        cache_key: str,
        cached_data: Any,
        inputs: set[str] | None = None,
    ) -> None:
        """Feed one lookup outcome to the perpetual-miss guard.

        Called on every run that actually performed a lookup — a skipped lookup
        never serialises either, so it carries no evidence about whether
        serialising pays back.

        A *hit* is "the key matched an entry", i.e. ``cached_data`` is not None.
        A key that matched but whose entry was invalidated (TTL / changed file
        dep) reads as a miss here, and correctly so: the key was STABLE, so it
        registers no churn and cannot move the counter. That workflow —
        recompute because a file changed, cache for the next unchanged run — is
        exactly the one that must never be guarded.

        Control-structure BODY statements are excluded, following the same
        precedent as mutation classification: they arrive with a per-iteration
        marker comment, so every iteration is a different ``source_hash``. The
        "identical source, run repeatedly" signature is meaningless for them, and
        recording one per iteration would grow the store by the loop's trip
        count.
        """
        if skip_cache:
            return
        if has_marker(code):
            return
        self._miss_guard.observe(
            source_hash,
            cache_key,
            hit=cached_data is not None,
            components=self._records.lineages_read(inputs) if inputs else {},
        )

    @staticmethod
    def _log(
        code: str,
        cache_key: str,
        inputs: set[str],
        cached_data: Any,
        analysis_time: float,
        hash_time: float,
        cache_check_time: float,
    ) -> None:
        logger.debug("[CACHE DEBUG] Statement: %s...", code[:50])
        logger.debug("[CACHE DEBUG] Key: %s...", cache_key[:40])
        logger.debug("[CACHE DEBUG] Inputs: %s", inputs)
        logger.debug("[CACHE DEBUG] Cache hit: %s", cached_data is not None)
        logger.debug(
            "[TIMING] Analysis: %.1fms | Hash: %.1fms | Lookup: %.1fms",
            analysis_time * 1000,
            hash_time * 1000,
            cache_check_time * 1000,
        )
