"""Serving a statement from its cache entry.

:class:`CacheHitServer` restores a hit's outputs and replays its output
(through :class:`StatementRestorer`), then fills the statement's metrics from
the entry the way a computed statement's metrics are filled from its run.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cash import rewrite_in_place
from cash._clock import perf_counter as _perf_counter
from cash.exceptions import CacheBackendError, CacheSerializationError
from cash.notebook import helper_memos
from cash.notebook.anchors import link_anchors, resolve_anchors
from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import in_place
from cash.notebook.statement.control_body import is_one_structure
from cash.notebook.statement.results import COST_MODEL_KEYS
from cash.notebook.statement.run import ECHO_FIELD, MEMO_CHANGES_FIELD, echoes

if TYPE_CHECKING:
    from cash.notebook.statement._metadata import StatementCacheMetadata
    from cash.notebook.statement.rebuild_cost import RebuildCostLedger
    from cash.notebook.statement.restore import StatementRestorer
    from cash.notebook.statement.results import ProcessResult
    from cash.notebook.statement.run import StatementRun
    from cash.notebook.tracking_state import TrackingState

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
        #: Why the last `serve` that gave None did not serve the entry: the
        #: statement's miss reason, which the badge shows.
        self.last_refusal: str | None = None

    def _refused(self, reason: str) -> None:
        """Not served, for *reason*: logged and kept as `last_refusal`."""
        logger.debug("[CACHE] %s, running it instead.", reason)
        self.last_refusal = reason

    def serve(
        self,
        run: StatementRun,
        cached_data: Any,
        metadata: StatementCacheMetadata | None,
        seed_epochs: Mapping[str, str],
    ) -> ProcessResult | None:
        """Restore from cache and populate *metrics* for a cache-hit path.

        Returns the completed *metrics* dict on success, or ``None`` if
        restoration fails (caller should fall through to execution).

        *seed_epochs* is the seeding regime in force now
        (``StatementRandomness.seed_epochs``).
        """
        self.last_refusal = None
        cache_key, inputs, metrics, process_start = run.cache_key, run.inputs, run.metrics, run.process_start
        holds_echo = isinstance(cached_data, dict) and ECHO_FIELD in cached_data
        if echoes(run.code, run.tree, run.is_last) and not holds_echo:
            # Stored where it was not the cell's result: the entry has no value
            # for IPython's output history (``_``, ``Out``) to take.
            return self._refused("the entry holds no value for the cell's result")
        if holds_echo and not echoes(run.code, run.tree, run.is_last):
            # The other way round: stored as the cell's last line, it replays
            # the display of its value, which here (``slow(5)`` followed by
            # another statement) a plain run does not show.
            return self._refused("the entry shows a value this statement does not show")
        if metadata is not None and metadata.restored_into:
            # Restored into the live objects: each must be one of the kind stored.
            variables = cached_data.get("variables") if isinstance(cached_data, dict) else None
            if not isinstance(variables, dict) or not in_place.ready(
                variables, metadata.restored_into, self._restorer.shell.user_ns
            ):
                return self._refused("a name it restores into is bound to another kind of object")
            # A weak reference or an id() key held now finds a copy after the
            # restore, whenever it was made: asked of the live objects, as
            # when the loop was stored.
            user_ns = self._restorer.shell.user_ns
            statement = not is_one_structure(run.tree)
            for name in sorted(metadata.restored_into):
                live = user_ns.get(name)
                why = in_place.refusal(live, user_ns)
                if why is None and (statement or in_place.written_into(live)):
                    # Checked for every name before any is written into, so
                    # a refused hit leaves them as they are for the run.
                    why = rewrite_in_place.mismatch(live, variables[name])
                if why is not None:
                    return self._refused(f"it cannot restore into {name} ({why})")
        if metadata is not None and metadata.bound_copies:
            # Bound to the stored copy: a weak reference or an id() made
            # since points at the live object, which the copy is not.
            user_ns = self._restorer.shell.user_ns
            for name in metadata.bound_copies:
                if name in user_ns and not in_place.bound_as_a_copy(user_ns[name], user_ns):
                    return self._refused(f"something points at {name} now")
        if metadata is not None and metadata.anchors:
            # The variables' objects the entry holds by reference: the live
            # ones, or a miss when one changed (`anchors`).
            try:
                cached_data = resolve_anchors(
                    cached_data,
                    metadata.anchors,
                    self._restorer.shell.user_ns,
                    self._restorer.compute_hash,
                    metadata.anchor_holds,
                )
            except (KeyError, pickle.UnpicklingError) as e:
                return self._refused(str(e))
        put_back = None
        if metadata is not None and helper_memos.carries_changes(metadata.helper_memos):
            # What the run put into the helper memos it filled goes back into
            # them once the outputs are restored, as one graph with them.
            changes = cached_data.get(MEMO_CHANGES_FIELD) if isinstance(cached_data, dict) else None
            outputs = (
                (cached_data.get("variables"), cached_data.get(ECHO_FIELD)) if isinstance(cached_data, dict) else None
            )
            put_back = helper_memos.put_back(metadata.helper_memos, changes, self._restorer.shell.user_ns, outputs)
            if isinstance(put_back, str):
                return self._refused(put_back)
        try:
            self._log_hit(cache_key, inputs, metadata)
            self._restorer.restore_from_cache(
                self.tracking_state,
                cached_data,
                metadata,
                run.silent,
                process_start,
                seed_epochs=seed_epochs,
            )
            if put_back is not None:
                put_back()

            if metadata is not None and metadata.anchor_links:
                user_ns = self._restorer.shell.user_ns
                link_anchors(
                    self.tracking_state.derivation_edges,
                    [name for name in metadata.outputs or () if name in user_ns],
                    [name for name in metadata.anchor_links if name in user_ns],
                    metadata.anchors,
                )
            self._report_hit(run, cached_data, metadata)
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
            self.last_refusal = f"restoring it failed ({e})"
            return None

    def _log_hit(self, cache_key: str, inputs: Any, metadata: StatementCacheMetadata | None) -> None:
        """Log the hit's key and the input and stored lineages it matched on."""
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

    def _report_hit(self, run: StatementRun, cached_data: Any, metadata: StatementCacheMetadata | None) -> None:
        """Fill *run*'s metrics for a restored hit: status, saved time, what
        it restored and from where, and the output it replays."""
        cache_key, inputs, metrics, process_start = run.cache_key, run.inputs, run.metrics, run.process_start
        metrics["status"] = CacheStatus.RESTORED
        metrics["saved_time"] = (metadata.execution_time or 0.0) if metadata else 0.0
        metrics["restored_vars"] = (metadata.outputs or []) if metadata else []
        # Carry the stored input list through so provenance can
        # reconstruct the dependency graph on a cache hit, not just on
        # a fresh compute.
        metrics["inputs"] = list((metadata.inputs or []) if metadata else [])
        metrics["total_time"] = _perf_counter() - process_start

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
            if ECHO_FIELD in payload:
                metrics[ECHO_FIELD] = payload[ECHO_FIELD]
        else:
            metrics["stdout"] = ""
            metrics["stderr"] = ""
            metrics["rich_outputs"] = []
