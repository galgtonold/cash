"""Writing a statement's result to the cache, and deciding whether to.

:class:`StatementStore` owns the store half of the statement pipeline: the
"too cheap to cache" floor, the size-aware restore-cost gate (Gate A of the
cost model), the refusals that route a result to a metadata-only entry, and
the write itself -- the payload, the metadata and the files it was built from.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
import pickle
import time
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cash import cost_model
from cash.notebook._memo import LruMemo
from cash.notebook.statement._metadata import StatementCacheMetadata
from cash.notebook.statement.miss_guard import GUARD_SKIP_REASON
from cash.object_hashing import estimate_object_size
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_dependencies
from cash.tracking.randomness import capture_object_rng_states, capture_rng_state

from ..call_refs import REF_BYTES_FIELD, REFS_FIELD

if TYPE_CHECKING:
    from cash.notebook._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
    from cash.notebook.statement.amplification import AmplificationGuard
    from cash.notebook.statement.call_routing import CallRouting
    from cash.notebook.statement.lineage import StatementLineageBuilder
    from cash.notebook.statement.rebuild_cost import RebuildCostLedger
    from cash.notebook.statement.restore import StatementRestorer
    from cash.notebook.statement.run import StatementExecution, StatementRun

logger = logging.getLogger(__name__)

__all__ = ["StatementStore", "config_float"]


def config_float(config: Any, attr: str, default: float) -> float:
    """Read a float-valued config attribute defensively.

    Returns ``default`` when the config is None, missing the attribute,
    or holding a value that can't be converted to float (e.g. a
    MagicMock in tests).
    """
    if config is None:
        return default
    try:
        return float(getattr(config, attr, default))
    except (TypeError, ValueError):
        return default


def _snapshot_with_inherited(
    file_dependencies: set[str],
    accessed_remote: set[str],
    inherited_snapshots: dict[str, dict] | None,
) -> dict[str, dict]:
    """Snapshot the files a statement read itself; take the ones it only
    inherited from its inputs' producers as they recorded them."""
    inherited_snapshots = inherited_snapshots or {}
    own = set(file_dependencies) - inherited_snapshots.keys()
    snapshot = snapshot_dependencies(own, accessed_remote)
    for path, recorded in inherited_snapshots.items():
        snapshot.setdefault(path, recorded)
    return snapshot


def _version_slot(source_hash: str, outputs: set[str]) -> str:
    """What makes two stored results versions of one statement: its source and
    the names it binds. Two cells with the same statement share a slot; a
    version one of them reads in this process is never pruned for the other."""
    return hashlib.sha256(f"{source_hash}|{','.join(sorted(outputs))}".encode()).hexdigest()[:32]


def _is_only_definitions(code: str) -> bool:
    """Whether *code* is nothing but ``def``/``class`` statements."""
    try:
        body = ast.parse(code).body
    except (SyntaxError, ValueError):
        return False
    return bool(body) and all(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in body)


class StatementStore:
    """Decides whether a statement's result is worth storing, and stores it.

    Collaborators are injected: the lineage builder writes the entry's
    lineages, the restorer writes metadata-only records, the amplification
    guard and the rebuild-cost ledger can refuse or force a write, and the
    call router swaps served call results for references.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        cash_instance: CashInstanceProtocol,
        *,
        restorer: StatementRestorer,
        lineage_builder: StatementLineageBuilder,
        amplification: AmplificationGuard,
        rebuild_cost: RebuildCostLedger,
        calls: CallRouting,
        seed_epochs: Mapping[str, Any],
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.cash_instance = cash_instance
        self._restorer = restorer
        self._lineage_builder = lineage_builder
        self._amplification = amplification
        self._rebuild_cost = rebuild_cost
        self._calls = calls
        #: The seeding regime each RNG was last seeded under, stored with the
        #: RNG state (``StatementRandomness.seed_epochs``, shared, not copied).
        self._seed_epochs = seed_epochs
        #: Names a later top-level statement of the running cell writes
        #: (:meth:`set_written_later_in_cell`).
        self.written_later_in_cell: frozenset[str] = frozenset()
        #: The file snapshots each producing entry recorded, by cache key,
        #: under the ``"__epoch__"`` of the hash scheme they were taken with.
        self._producer_snapshots: LruMemo[str, dict[str, dict]] = LruMemo(64)
        self._producer_snapshots_epoch: Any = None

    def set_written_later_in_cell(self, names: frozenset[str]) -> None:
        """Record the names a later top-level statement of the cell writes.

        An intermediate value of the cell stays in RAM; only the cell's final
        version of a name is persisted, at the cell's end.
        """
        self.written_later_in_cell = names

    #: The perpetual-miss guard spares a statement whose value is written in at
    #: most this share of what computing it cost: each write it wastes is then
    #: nearly free, and one later hit repays all of them.
    CHEAP_WRITE_SHARE = 0.1

    def write_is_cheap(self, outputs: set[str], captured_vars: dict[str, Any], execution_time: float) -> bool:
        """Whether writing *outputs* costs little next to *execution_time*.

        The guard exists for values whose every write is wasted money -- a
        bare fit re-serialising a large model, -25 s a session. Five upstream
        edits in a row also churn a key, and that is an ordinary morning of
        model tuning: a cross-validation, seconds to compute and a few
        numbers to store, stopped being saved and the next restart ran every
        CV again. Estimated with the cost model the size-aware skip uses.
        """

        if execution_time <= 0:
            return False
        try:
            write = 0.0
            for name in outputs:
                if name not in captured_vars:
                    continue
                value = captured_vars[name]
                write += cost_model.estimated_serialize_time(type(value).__name__, estimate_object_size(value), "disk")
        except Exception:  # noqa: BLE001 - an estimate it cannot make guards as before
            return False
        return write <= self.CHEAP_WRITE_SHARE * execution_time

    def save(
        self,
        run: StatementRun,
        execution: StatementExecution,
        captured_vars: dict[str, Any],
        miss_guarded: bool = False,
    ) -> StatementCacheMetadata | None:
        if getattr(execution.result, "skipped", False):
            return None

        accessed_files = execution.accessed_files
        all_file_deps = set(accessed_files) if accessed_files else set()
        inherited_snapshots: dict[str, dict] = {}

        # CRITICAL: Include inherited file dependencies from input variables
        # This ensures that when Cell 3 (`df`) is cached, it stores the CSV file's mtime
        # even though Cell 3 didn't directly read the file. This allows proper invalidation.
        #
        # An inherited file is recorded as the input's PRODUCER recorded it --
        # the state the value was built from -- rather than read and hashed
        # again here. Re-snapshotted per statement, every statement derived from
        # a frame read out of 5,222 files re-read all 5,222.
        # A later lookup still checks the real file against it.
        direct = all_file_deps.copy()
        for input_var in run.inputs:
            inherited = self.tracking_state.executed_file_deps.get(input_var)
            if not inherited:
                continue
            all_file_deps.update(inherited)
            recorded = self._producer_file_snapshots(input_var)
            for path in inherited:
                if path in recorded and path not in direct:
                    inherited_snapshots.setdefault(path, recorded[path])

        return self._store(
            run,
            execution,
            captured_vars,
            file_dependencies=all_file_deps,
            miss_guarded=miss_guarded,
            inherited_snapshots=inherited_snapshots,
        )

    def _producer_file_snapshots(self, var_name: str) -> dict[str, dict]:
        """The file snapshots *var_name*'s producing statement stored, or {}."""
        key = self.tracking_state.variable_sources.get(var_name)
        backend = getattr(self.cash_instance, "backend", None) if self.cash_instance else None
        if not key or backend is None:
            return {}

        memo = self._producer_snapshots
        if self._producer_snapshots_epoch != file_dep_snapshot.HASH_EPOCH:
            memo.clear()
            self._producer_snapshots_epoch = file_dep_snapshot.HASH_EPOCH
        cached = memo.get(key)
        if cached is not None:
            return cached
        try:
            meta = backend.peek_metadata(key)
        except Exception:  # noqa: BLE001 - a snapshot it cannot read is taken afresh
            meta = None
        snaps = (meta or {}).get("file_dependencies") or {} if isinstance(meta, dict) else {}
        memo[key] = snaps
        return snaps

    def should_skip_large_object_caching(
        self,
        captured_vars: dict[str, Any],
        execution_time: float,
        force_persist: bool = False,
        has_file_dependencies: bool = False,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Decide whether caching a set of output variables is worthwhile.

        The guiding principle is **expected time savings**: caching should only
        be skipped when the overhead of storing and later restoring the result
        is so high relative to re-computing that the user wouldn't benefit.

        The cost model is **backend-aware**:

        * **RAM (InMemoryBackend)** – no serialisation, only ``deepcopy``.
          Estimated at ~2 GB/s for pandas types, ~500 MB/s for generic objects.
          Overhead = 2 × copy time (store + restore).  This is very cheap, so
          RAM caching is almost always worthwhile.

        * **Disk / remote (FileBackend, Redis, S3, TieredBackend)** – needs
          pickle + I/O.  Restore cost is predicted by the fitted cost model
          in ``cash.cost_model`` (per-family ``a + b·size_bytes``).
          Overhead = 2 × serialise time (store + restore).

        Caching is skipped when the expected time savings would be less than
        ``min_cache_savings_pct`` (default 20 %) of the original execution time::

            expected_savings = execution_time - est_restore_time
            skip  ⟺  expected_savings < min_cache_savings_pct × execution_time

        Equivalently::

            skip  ⟺  est_restore_time > (1 - min_cache_savings_pct) × execution_time

        Special cases that always allow caching (never skip):
        - ``force_persist`` is set (user explicitly wants caching via ``@cash:persist``)
        - The statement has direct file dependencies (I/O-bound; caching avoids
          re-reading from disk)

        Returns:
            ``(should_skip, reason, prediction)`` where *reason* is a human-readable
            explanation when skipping (else ``None``), and *prediction* is the cost-model
            dict for the largest variable seen (keys: ``size_bytes``, ``restore_seconds``,
            ``type_name``, ``family``), or ``None`` when estimation failed for all vars.

        The early-return paths (``force_persist``, ``has_file_dependencies``) still
        compute and return the prediction so downstream observability (cost-model
        validation, residual reports) sees consistent family attribution regardless
        of which gate fired the cache decision.
        """
        # NOTE: the "too cheap to cache" floor (statements whose own compute
        # is below ``min_execution_time_to_cache_seconds``) is checked
        # earlier in ``_store`` — earlier than this method — so
        # that no metadata entry is written at all. That keeps subsequent
        # warm lookups as fast cache misses rather than slow
        # metadata-only hits.

        # --- Determine cost model based on backend type -----------------------
        backend = getattr(self.cash_instance, "backend", None)
        backend_type = type(backend).__name__ if backend else ""

        # For TieredBackend the first (fastest) tier determines the restore cost
        # because that's where the data will be read from on cache hit.
        if backend_type == "TieredBackend" and backend.backends:
            primary_backend_type = type(backend.backends[0]).__name__
        else:
            primary_backend_type = backend_type

        is_ram_backend = primary_backend_type == "InMemoryBackend"

        config = getattr(self.cash_instance, "config", None)
        min_savings_pct = config_float(config, "min_cache_savings_pct", 0.20)
        fixed_budget = config_float(config, "min_cache_fixed_budget_seconds", 0.05)

        # Track the prediction for the largest variable (by size_bytes) seen so
        # far. Computed even on the early-return paths so observability is
        # consistent — the file_dependencies / force_persist gates decide
        # whether to cache, not whether to predict.
        largest_prediction: dict[str, Any] | None = None

        # Compute predictions for all vars; collect skip-causing var separately.
        skip_decision: tuple[str | None, dict[str, Any] | None] | None = None
        for var_name, var_value in captured_vars.items():
            skip, reason, prediction = self._check_var_restore_budget(
                var_name,
                var_value,
                execution_time,
                is_ram_backend,
                min_savings_pct,
                fixed_budget,
            )
            # Keep track of the largest variable's prediction for exposure.
            if prediction is not None:
                if largest_prediction is None or prediction["size_bytes"] > largest_prediction["size_bytes"]:
                    largest_prediction = prediction
            # Only the FIRST skip-causing var matters for the decision; remember it.
            if skip and skip_decision is None:
                skip_decision = (reason, prediction)

        # Never skip caching for statements that directly read files (file I/O
        # is inherently expensive; the "fast computation" heuristic doesn't apply)
        # or when force_persist is set by a user annotation. The prediction is
        # still returned so observability stays consistent.
        if force_persist or has_file_dependencies:
            return False, None, largest_prediction

        if skip_decision is not None:
            reason, skip_prediction = skip_decision
            return True, reason, skip_prediction

        return False, None, largest_prediction

    def _check_var_restore_budget(
        self,
        var_name: str,
        var_value: Any,
        execution_time: float,
        is_ram_backend: bool,
        min_savings_pct: float,
        fixed_budget: float,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Return (skip, reason, prediction) for a single variable based on the
        predicted restore cost from the fitted cost model.

        The skip decision uses ``max(fixed_budget, (1 - min_savings_pct) ×
        execution_time)`` as the budget. The fixed budget covers the
        per-call overhead of cheap caches (e.g. opening a file) so trivial
        cells aren't refused; the ratio kicks in once compute is large
        enough to dominate. This matches the policy framing: small fixed
        overhead is fine, what we want to avoid is doubling a long cell.

        ``prediction`` is a dict with keys ``size_bytes``, ``restore_seconds``,
        ``type_name``, ``family``; or ``None`` if size estimation raises.
        """

        try:
            obj_size = estimate_object_size(var_value)
            type_name = type(var_value).__name__
            backend_kind = "ram" if is_ram_backend else "disk"
            family = cost_model.resolve_family(type_name)
            est_restore_time = cost_model.estimated_restore_time(type_name, obj_size, backend_kind)
            prediction: dict[str, Any] = {
                "size_bytes": obj_size,
                "restore_seconds": est_restore_time,
                "type_name": type_name,
                "family": family,
            }
            max_acceptable_restore = max(
                fixed_budget,
                (1.0 - min_savings_pct) * execution_time,
            )

            if execution_time > 0 and est_restore_time > max_acceptable_restore:
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copying" if is_ram_backend else "serializing"
                pct_label = f"{min_savings_pct * 100:.0f}%"
                reason = (
                    f"Restoring '{var_name}' ({size_mb:.0f} MB {type_name}) would take "
                    f"~{est_restore_time:.2f}s vs {execution_time:.2f}s compute "
                    f"({backend_label}, <{pct_label} savings) — "
                    f"use @cash:persist to force"
                )
                logger.debug("[SIZE_AWARE] %s", reason)
                return True, reason, prediction
            if obj_size > 10 * 1024 * 1024:
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copy" if is_ram_backend else "serialize"
                logger.debug(
                    "[SIZE_AWARE] Caching '%s' (%.1fMB %s) — est. %s %.2fs vs %.2fs compute",
                    var_name,
                    size_mb,
                    type_name,
                    backend_label,
                    est_restore_time,
                    execution_time,
                )
            return False, None, prediction
        except (TypeError, ValueError, AttributeError, OSError, RecursionError):
            logger.debug("[SIZE_AWARE] Failed to estimate object size, allowing caching")
        return False, None, None

    @staticmethod
    def _filter_safe_vars(captured_vars: dict[str, Any]) -> dict[str, Any]:
        """Every captured variable but a module. Whether a value pickles is the
        backend's to find out when it stores it."""
        return {k: v for k, v in captured_vars.items() if not isinstance(v, types.ModuleType)}

    def _store(
        self,
        run: StatementRun,
        execution: StatementExecution,
        captured_vars: dict[str, Any],
        *,
        file_dependencies: set[str],
        miss_guarded: bool = False,
        inherited_snapshots: dict[str, dict] | None = None,
    ) -> StatementCacheMetadata | None:
        """Store execution results and metadata in the cache. Returns
        metadata, or ``None`` when the statement was so cheap to compute
        that we don't even write a metadata-only entry (the next lookup
        will miss cleanly rather than hit a metadata-only entry and
        pay a per-file read just to decide 'recompute')."""
        cache_key, code, source_hash = run.cache_key, run.code, run.source_hash
        inputs, outputs, ttl = run.inputs, run.outputs, run.effective_ttl
        force_persist = run.force_persist
        execution_time = execution.cost
        captured_output = execution.captured
        accessed_remote = execution.accessed_remote
        t_store = time.time()
        # What this key recorded is about to change (``_producer_file_snapshots``).
        self._producer_snapshots.pop(cache_key, None)

        # "Too cheap to cache" floor — checked here (not inside
        # ``should_skip_large_object_caching``) so we can skip writing
        # even a metadata-only entry. Without this, a notebook with many
        # trivial statements (e.g. 100 `a_i = i + 1`) would write 100
        # metadata-only files on the first run; every subsequent run
        # would pay ~1ms/statement of cache-lookup overhead reading
        # them only to discover they're skipped entries. By writing
        # nothing, the next lookup is a fast clean miss.
        # The restore-cost check below is waived only for a statement that
        # READS a file itself: reading is the expensive part then.
        # `file_dependencies` also holds every file the inputs were built from,
        # and waiving on those exempted everything downstream of a load:
        # ~400 MiB frames restoring slower than they computed, served as hits.
        #
        # The too-cheap FLOOR keeps the old rule on purpose. A first version
        # narrowed it too, and a cheap reader of a file HANDLE
        # (`lines = [l for l in fh]`) stopped being stored, re-ran on the second
        # Run All against the handle its skipped producer had left at EOF, and
        # printed [] (test_file_handle_iteration_second_run_all).
        reads_files = bool(execution.accessed_files or accessed_remote)
        if not force_persist and not file_dependencies and not accessed_remote:
            config_obj = getattr(self.cash_instance, "config", None)
            min_exec_time = config_float(config_obj, "min_execution_time_to_cache_seconds", 0.01)
            # ``execution_time`` is wall clock (``time.time()``), not CPU time,
            # so it charges the statement for any scheduling stall too. On
            # Windows the clock can report exactly 0.0 for a genuinely
            # instantaneous statement (a = 1) because its resolution is coarser
            # than the operation; treat 0 the same as "below the floor" — both
            # mean "too cheap to cache". The converse also holds: on a heavily
            # contended machine a trivial statement can measure tens of ms and
            # legitimately clear the floor, so nothing may assume this branch is
            # taken for a given statement (see the floor-exit test, which pins
            # the threshold rather than trusting the machine to be fast).
            if execution_time < min_exec_time and not self._rebuild_cost.final_over_costly_inputs(
                inputs, outputs, in_loop=self._calls.in_loop, written_later=self.written_later_in_cell
            ):
                logger.debug(
                    "[SIZE_AWARE] Compute took only %.1fms, below %.0fms floor — not writing cache entry",
                    execution_time * 1000,
                    min_exec_time * 1000,
                )
                return None

        # Size-aware caching: skip storing large objects when serialization overhead dominates
        should_skip, skip_reason, prediction = self.should_skip_large_object_caching(
            captured_vars,
            execution_time,
            force_persist,
            has_file_dependencies=reads_files,
        )

        # Statements whose outputs include a __main__-defined function or
        # class are never VALUE-cached: those values pickle BY
        # REFERENCE to a binding that won't exist in the next session, so a
        # value entry would either crash the lookup (dangling find_class ->
        # AttributeError) or "restore" nothing and wrongly skip the defining
        # statement. Route them through the metadata-only path (same as the
        # size-aware skip): output lineages persist for the upstream
        # simulation, while the value lookup misses cleanly and the cheap
        # statement (lambda assign, `g = f` alias) re-executes.
        if not should_skip:
            _unrestorable = sorted(
                _name
                for _name, _v in captured_vars.items()
                if (inspect.isfunction(_v) or inspect.isclass(_v)) and getattr(_v, "__module__", None) == "__main__"
            )
            if _unrestorable:
                should_skip = True
                skip_reason = (
                    f"__main__ function/class output(s) "
                    f"{', '.join(_unrestorable)} are unrestorable by value; "
                    f"statement re-executes (lineage persists)"
                )
                if _is_only_definitions(code):
                    # Nothing to report for a ``def``: it always re-runs at no
                    # cost. The reason is for ``f = make_fn()``. A def reading
                    # file-loaded data got here past the too-cheap floor and
                    # its badge row said NOT CACHED.
                    skip_reason = None
        # Perpetual-miss guard. Placed LAST so it can override the
        # exemptions above: ``has_file_dependencies`` waives the whole size-aware
        # cost model, and that waiver is precisely how it shipped — a fit
        # on a CSV-derived frame inherits the read's file deps, so the cost model
        # never got a vote and the frame was re-serialised every run for a cache
        # that could never hit. An unstable key does not become stable because the
        # statement touched a file. ``force_persist`` is checked by the caller and
        # is the one thing that outranks this.
        if not should_skip and miss_guarded:
            should_skip = True
            skip_reason = GUARD_SKIP_REASON

        # Loop-persist amplification guard. Placed after every other
        # gate so it only accounts writes that would ACTUALLY have happened, and
        # so it can override ``force_persist`` -- which is the whole point: a
        # user asking to persist one value must not silently get every
        # intermediate state of it written to their disk. It is the last word
        # because it is a disk-safety guard, not a cost heuristic.
        if not should_skip:
            amplified, amplified_reason = self._amplification.check(
                code,
                prediction,
                annotated=force_persist,
            )
            if amplified:
                should_skip = True
                skip_reason = amplified_reason

        # Cost-model prediction fields are shared by both the skip and the
        # full-store branches; build them once.
        cost_fields: dict[str, Any] = {}
        if prediction is not None:
            cost_fields = {
                "cost_model_size_bytes": prediction["size_bytes"],
                "cost_model_restore_seconds": prediction["restore_seconds"],
                "cost_model_type_name": prediction["type_name"],
                "cost_model_family": prediction["family"],
            }

        if should_skip:
            skip_metadata = StatementCacheMetadata(
                timestamp=time.time(),
                inputs=list(inputs),
                outputs=list(outputs),
                execution_time=execution_time,
                source_hash=source_hash,
                code=code,
                key=cache_key,
                skipped_reason=skip_reason,
                metadata_only=True,
                output_lineages=self._lineage_builder.build_output_lineages(self.tracking_state, outputs),
                input_lineages=self._lineage_builder.build_input_lineages(self.tracking_state, inputs),
                **cost_fields,
            )
            try:
                backend = self.cash_instance.backend if self.cash_instance else None
                if backend is not None:
                    self._restorer.persist_metadata_only(backend, cache_key, skip_metadata.to_dict())
            except (OSError, TypeError, ValueError, AttributeError):
                logger.debug("[PROCESSOR] Best-effort metadata persistence failed")
            return skip_metadata

        metadata = StatementCacheMetadata(
            timestamp=time.time(),
            inputs=list(inputs),
            outputs=list(outputs),
            execution_time=execution_time,
            source_hash=source_hash,
            code=code,
            key=cache_key,
            file_dependencies=_snapshot_with_inherited(file_dependencies, accessed_remote, inherited_snapshots),
            force_persist=force_persist,
            output_lineages=self._lineage_builder.build_output_lineages(self.tracking_state, outputs),
            input_lineages=self._lineage_builder.build_input_lineages(self.tracking_state, inputs),
            ttl=ttl,
            version_slot=_version_slot(source_hash, outputs),
            **cost_fields,
        )

        variables = self._filter_safe_vars(captured_vars)
        referenced: dict[str, int] = {}
        variables = self._calls.with_call_refs(variables, code, referenced)
        payload = {
            "variables": variables,
            "stdout": captured_output.stdout,
            "stderr": captured_output.stderr,
            # Rich-display output (RichOutput objects). The 'outputs' key in
            # the sibling ``metadata`` dict holds variable NAMES — two distinct
            # concepts; keep them on different keys here too.
            "rich_outputs": captured_output.outputs,
            "rng_state": capture_rng_state(),
            # The seeding regime this state was captured under, so a later
            # restore can tell whether replaying it would clobber a re-seed
            # rather than continue the stream.
            "rng_epochs": dict(self._seed_epochs),
        }

        # the module-global RNG post-state above misses generators the
        # user holds in a variable (``rng = np.random.default_rng(42)``).
        # Capture those too, scoped to this statement's inputs so the cost
        # stays proportional to what the statement actually reads.  Omitted
        # entirely when there are none, keeping the payload shape unchanged for
        # the overwhelming majority of statements.
        try:
            object_rng_states = capture_object_rng_states(inputs, self.shell.user_ns)
            if object_rng_states:
                payload["rng_object_states"] = object_rng_states
        except (TypeError, AttributeError) as e:
            logger.debug("[RANDOMNESS] Object RNG capture skipped: %s", e)

        # Dict-on-the-wire: the backend round-trips a plain dict and may
        # inject the resolved ``storage`` destinations back into it. We
        # re-wrap that mutated dict at the end so the returned view carries
        # the storage info on to the badge metrics.
        wire = metadata.to_dict()
        if referenced:
            wire[REFS_FIELD] = sorted(referenced)
            wire[REF_BYTES_FIELD] = sum(referenced.values())
        # An intermediate of this cell (``cell_executor._written_later_in_cell``)
        # stays in RAM; the cell's final version is persisted at its end.
        later = self.written_later_in_cell
        if not force_persist and outputs and later and set(outputs) <= later:
            wire["defer_persist"] = True

        try:
            self.cash_instance.backend.set(cache_key, payload, wire)
        except (OSError, TypeError, ValueError, pickle.PicklingError, RuntimeError) as e:
            logger.warning("[CACHE] Failed to write to cache backend: %s", e)
        else:
            # Charge this write to its statement's amplification budget, now
            # that the backend has reported which tiers actually took it
            # . Only durable destinations count.
            self._amplification.account(code, prediction, wire)

        # The metadata-only record keeps a RAM-only value's lineage across a
        # restart. A value written to a persistent tier carries its metadata
        # already, and asking for the record made the disk tier wait for that
        # write (``FileBackend.set_metadata_only``) only to skip it: a
        # cleaning cell spent 5.8 s of its cold run there, on ~500 MB frames
        # whose write was meant to happen in the background.
        persisted = any(d != "RAM" for d in (wire.get("storage") or ()))
        try:
            backend = self.cash_instance.backend
            if backend is not None and not persisted:
                self._restorer.persist_metadata_only(backend, cache_key, wire)
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("[PROCESSOR] Best-effort metadata persistence failed")

        store_time = time.time() - t_store
        total_time = time.time() - run.process_start

        logger.debug("[TIMING] Store: %.1fms | OVERALL: %.1fms", store_time * 1000, total_time * 1000)
        logger.debug("[CACHE DEBUG] Stored in cache: %s", cache_key)

        return StatementCacheMetadata.from_dict(wire)
