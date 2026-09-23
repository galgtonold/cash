"""Variable-granular cache restoration.

Owns the operation "make this variable present in ``shell.user_ns`` by
fetching it from the cache, validating its file dependencies, and
reconstituting its lineage in :class:`TrackingState`."

Single public entry: :meth:`Restorer.restore_variable`.  Recurses on
missing inputs so a freshly restored variable always has its upstream
dependencies present too.

**Anti-god-class rule (load-bearing):** this module does *not* orchestrate
cells.  It restores **one** variable + its missing inputs.  If a
multi-variable use case appears, the caller composes :meth:`restore_variable`
calls; the Restorer does not grow a "restore many" loop with cell-aware
logic.  Cell-level orchestration is the ``CellExecutor``'s job.
"""

from __future__ import annotations

import logging
import pickle
import types
from typing import Any

from ..object_hashing import compute_hash
from ..tracking.file_dep_snapshot import snapshot_is_fresh
from ._protocols import ShellProtocol, TrackingState
from .cache_status import CacheStatus
from .call_refs import resolve_call_refs
from .statement import ProcessResult

logger = logging.getLogger(__name__)


class Restorer:
    """Restore one variable from cache, recursively ensuring inputs first.

    Variable-granular (one variable + its inputs), distinct from
    ``StatementProcessor`` which is statement-granular (one statement +
    its outputs).  Both operate on the same cache backend but at
    different units of work.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        backend: Any,
        tracking_state: TrackingState,
    ) -> None:
        self.shell = shell
        self._backend = backend
        self.tracking_state = tracking_state
        # Variables currently being restored, to break dependency cycles. A
        # self-referential statement (``u = u * 0.99``) or mutually-referential
        # multi-output one (``u, v = f(u, v)``) makes a variable depend on
        # itself (directly or via a co-output), so naive recursive
        # input-restoration would loop forever. The cached output already
        # incorporates those inputs, so once a variable is mid-restore we skip
        # re-entering it.
        self._restoring: set[str] = set()

    def restore_variable(self, var_name: str) -> list[ProcessResult]:
        """Restore a single variable from cache, recursively ensuring dependencies first."""
        if var_name in self._restoring:
            # Cycle: this variable is already mid-restore further up the stack
            # (a self- or mutually-referential statement). Its cached output
            # already folds in the inputs, so don't re-enter - the outer call
            # will restore the value once its inputs are current.
            return []
        result = self._fetch_cached_payload(var_name)
        if result is None:
            return []
        metadata, cached_data = result
        restored_metrics: list[ProcessResult] = []

        self._restoring.add(var_name)
        try:
            if not isinstance(cached_data, dict) or "variables" not in cached_data:
                logger.debug("[STATE] Invalid payload format for '%s'", var_name)
                return []

            self._ensure_inputs_current(var_name, metadata, restored_metrics)

            restored_vars = cached_data["variables"]
            if var_name in restored_vars:
                self.shell.user_ns[var_name] = restored_vars[var_name]
                self._restore_tracking_state(var_name, metadata, restored_vars)
                logger.debug("[STATE] Restored '%s' from cache", var_name)
                restored_metrics.append(self._build_restore_metric(var_name, metadata, restored_vars))
            else:
                logger.debug("[STATE] Variable '%s' not in cached payload", var_name)

        except (KeyError, TypeError, ValueError, AttributeError, OSError, pickle.UnpicklingError) as e:
            logger.debug("[STATE] Error restoring '%s': %s", var_name, e)
        finally:
            self._restoring.discard(var_name)

        return restored_metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_available_in_builtins(self, var_name: str) -> bool:
        """Check if variable is available in __builtins__."""
        builtins_ns = self.shell.user_ns.get("__builtins__")
        if not builtins_ns:
            return False

        if isinstance(builtins_ns, dict):
            return var_name in builtins_ns
        if isinstance(builtins_ns, types.ModuleType):
            return hasattr(builtins_ns, var_name)
        return False

    def _validate_file_deps(self, var_name: str, metadata: dict) -> None:
        """Raise NameError if any file dependency has changed since caching.

        Called before restoring a cached variable — if files have changed the
        cached value is stale and must be recomputed.
        """
        # Content-authoritative freshness: a touch keeps the restore valid, a
        # same-size sub-resolution edit invalidates it.
        fresh, stale = snapshot_is_fresh(metadata.get("file_dependencies"))
        if fresh:
            return
        logger.debug("[STATE] Cannot restore '%s': file dependency stale (%s)", var_name, stale)
        if stale.reason in ("missing", "unreadable"):
            raise NameError(f"name '{var_name}' is not defined (file dependency missing)")
        raise NameError(f"name '{var_name}' is not defined (file dependency changed)")

    def _restore_tracking_state(self, var_name: str, metadata: dict, restored_vars: dict) -> None:
        """Update TrackingState after writing a restored variable into user_ns."""
        restored_hash = compute_hash(restored_vars[var_name])
        hashes = self.tracking_state.variable_hashes
        if var_name not in hashes:
            hashes[var_name] = set()
        hashes[var_name].add(restored_hash)

        output_lineages = metadata.get("output_lineages", {})
        if var_name in output_lineages:
            self.tracking_state.lineage.record(
                var_name,
                output_lineages[var_name],
                value=self.shell.user_ns.get(var_name),
            )

        stored_code = metadata.get("code")
        if stored_code:
            self.tracking_state.executed_cell_codes[var_name] = stored_code

        stored_hash = metadata.get("source_hash")
        if stored_hash:
            self.tracking_state.executed_cell_hashes.setdefault(var_name, set()).add(stored_hash)

        file_deps = metadata.get("file_dependencies", {})
        if file_deps:
            file_dep_set = self.tracking_state.executed_file_deps
            if var_name not in file_dep_set:
                file_dep_set[var_name] = set()
            file_dep_set[var_name].update(file_deps.keys())

    @staticmethod
    def _build_restore_metric(var_name: str, metadata: dict, restored_vars: dict) -> ProcessResult:
        """Build a ProcessResult entry for a successfully restored variable.

        Carries through the cached ``inputs`` list so downstream observability
        (provenance.record, audit log) can reconstruct the dependency chain
        even when the variable was hydrated from disk rather than freshly
        computed in this session.
        """
        saved_time = metadata.get("execution_time", 0.0)
        source = metadata.get("source", metadata.get("storage", "Disk"))
        if isinstance(source, list):
            source = source[0] if source else "Disk"
        return {
            "code": metadata.get("code", f"# defined {var_name}"),
            "status": CacheStatus.RESTORED,
            "execution_time": 0.0,
            "total_time": saved_time,
            "saved_time": saved_time,
            "error": None,
            "restored_vars": list(restored_vars.keys()),
            "inputs": list(metadata.get("inputs", [])),
            "uncacheable_reasons": [],
            "source": source,
            "is_upstream": True,
            "storage": [source],
        }

    def _ensure_inputs_current(
        self,
        var_name: str,
        metadata: dict,
        restored_metrics: list[ProcessResult],
    ) -> None:
        """Recursively restore any stale input variables required by var_name."""
        for input_var in metadata.get("inputs", []):
            if input_var in (var_name, "get_ipython", "__builtins__"):
                continue
            if input_var not in self.shell.user_ns:
                restored_metrics.extend(self.restore_variable(input_var))
            elif input_var in self.tracking_state.variable_hashes:
                current_hash = compute_hash(self.shell.user_ns.get(input_var))
                if current_hash not in self.tracking_state.variable_hashes[input_var]:
                    restored_metrics.extend(self.restore_variable(input_var))

    def _fetch_cached_payload(self, var_name: str) -> tuple[dict, dict] | None:
        """Look up cache entry for var_name and validate it.

        Returns ``(metadata, cached_data)`` if found and valid, or ``None``
        to signal that the caller should return early.

        Raises ``NameError`` when var_name has no known source and is not a builtin.
        """
        if var_name not in self.tracking_state.variable_sources:
            if self._is_available_in_builtins(var_name):
                logger.debug("[STATE] '%s' not in cache, but found in built-ins. Using built-in.", var_name)
                return None
            logger.debug("[STATE] Cannot restore '%s': no cached source found", var_name)
            raise NameError(f"name '{var_name}' is not defined")

        cache_key = self.tracking_state.variable_sources[var_name]
        metadata, cached_data = self._backend.get(cache_key)
        if cached_data:
            cached_data = resolve_call_refs(cached_data, self._backend)
        if not cached_data:
            logger.debug("[STATE] Cannot restore '%s': cache miss for key %s...", var_name, cache_key[:16])
            return None

        self._validate_file_deps(var_name, metadata)
        return metadata, cached_data
