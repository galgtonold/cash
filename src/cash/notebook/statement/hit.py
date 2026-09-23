"""Serving a statement from its cache entry.

:class:`CacheHitServer` restores a hit's outputs and replays its output
(through :class:`StatementRestorer`), then fills the statement's metrics from
the entry the way a computed statement's metrics are filled from its run.
"""

from __future__ import annotations

import logging
import pickle
import time
from typing import TYPE_CHECKING, Any

from cash.exceptions import CacheBackendError, CacheSerializationError
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement.results import COST_MODEL_KEYS

if TYPE_CHECKING:
    from cash.notebook._protocols import TrackingState
    from cash.notebook.statement._metadata import StatementCacheMetadata
    from cash.notebook.statement.rebuild_cost import RebuildCostLedger
    from cash.notebook.statement.restore import StatementRestorer
    from cash.notebook.statement.results import ProcessResult
    from cash.notebook.statement.run import StatementRun

logger = logging.getLogger(__name__)

__all__ = ["CacheHitServer"]


class CacheHitServer:
    """Restores a cache hit and reports it in the statement's metrics."""

    def __init__(
        self, tracking_state: TrackingState, restorer: StatementRestorer, rebuild_cost: RebuildCostLedger
    ) -> None:
        self.tracking_state = tracking_state
        self._restorer = restorer
        self._rebuild_cost = rebuild_cost

    def serve(
        self,
        run: StatementRun,
        cached_data: Any,
        metadata: StatementCacheMetadata | None,
    ) -> ProcessResult | None:
        """Restore from cache and populate *metrics* for a cache-hit path.

        Returns the completed *metrics* dict on success, or ``None`` if
        restoration fails (caller should fall through to execution).

        ``run.est_fit`` names the estimator-fit receivers whose fitted state
        must be transferred onto the EXISTING object rather than rebound, so
        every alias observes the fit. It is recomputed each call from
        the live namespace (never read from ``mutation_verdicts``, which is empty
        right after a kernel restart).
        """
        cache_key, inputs, metrics, process_start = run.cache_key, run.inputs, run.metrics, run.process_start
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[CACHE_HIT_DEBUG] Cache hit for key: %s...", cache_key[:20])
                logger.debug(
                    "%s Input lineages used: %s",
                    "[CACHE_HIT_DEBUG]",
                    [
                        (v, self.tracking_state.variable_lineage.get(v, "NONE")[:16] + "...")
                        for v in inputs
                        if v not in ["get_ipython", "__builtins__", "print"]
                    ],
                )
                if metadata:
                    logger.debug(
                        "%s Stored lineages in cache: %s",
                        "[CACHE_HIT_DEBUG]",
                        [(k, v[:16] + "...") for k, v in (metadata.output_lineages or {}).items()],
                    )
            self._restorer.restore_from_cache(
                self.tracking_state, cached_data, metadata, run.silent, process_start, run.est_fit
            )

            metrics["status"] = CacheStatus.RESTORED
            metrics["saved_time"] = (metadata.execution_time or 0.0) if metadata else 0.0
            metrics["restored_vars"] = (metadata.outputs or []) if metadata else []
            # Carry the stored input list through so provenance/audit can
            # reconstruct the dependency graph on a cache hit, not just on
            # a fresh compute.
            metrics["inputs"] = list((metadata.inputs or []) if metadata else [])
            metrics["total_time"] = time.time() - process_start

            if metadata:
                if metadata.source is not None:
                    metrics["source"] = metadata.source
                    metrics["storage"] = [metadata.source]
                elif metadata.storage is not None:
                    metrics["storage"] = metadata.storage
                for k in COST_MODEL_KEYS:
                    value = getattr(metadata, k)
                    if value is not None:
                        metrics[k] = value
                where = [metadata.source, *(metadata.storage or ())]
                self._rebuild_cost.note(
                    cache_key,
                    inputs,
                    metadata.outputs or (),
                    metadata.execution_time or 0.0,
                    on_disk=any(s not in (None, "RAM") for s in where),
                )

            payload = cached_data
            if isinstance(payload, dict) and "variables" in payload:
                metrics["stdout"] = payload.get("stdout", "")
                metrics["stderr"] = payload.get("stderr", "")
                metrics["rich_outputs"] = payload.get("rich_outputs", [])
            else:
                metrics["stdout"] = ""
                metrics["stderr"] = ""
                metrics["rich_outputs"] = []

            return metrics
        except (
            CacheBackendError,
            CacheSerializationError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            OSError,
            pickle.UnpicklingError,
        ) as e:
            logger.warning("[CACHE] Restoration failed (%s), falling back to execution.", e, exc_info=True)
            return None
