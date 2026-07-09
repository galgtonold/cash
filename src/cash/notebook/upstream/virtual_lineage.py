from __future__ import annotations

"""Phase 1 of the notebook simulator: forward simulation + cache probing.

Extracted from ``NotebookSimulator``. Owns the simulator-internal caches
(``_simulation_cache``, ``_ast_cache``, ``_simulation_cell_hashes``,
``_cell_id_to_last_index``) and shares ``tracking_state`` dict references
with :class:`NotebookSimulator` and :class:`MismatchClassifier`. Pure-phase
invariants land in a later refactor.
"""

import ast
import hashlib
import logging
import os
import re
import time as time_module
import types
from collections.abc import Callable
from typing import Any

from ...utils import resolve_file_dep_path
from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from ..analysis import CodeAnalyzer
from ..cacheability import (
    KNOWN_PURE_METHODS,
    standalone_method_call_receivers,
    standalone_method_mutation_receivers,
)
from ..file_dep_snapshot import split_file_dep_value
from ..cache_key import CacheKeyContext, compute_cache_key
from ..cache_status import CacheStatus
from ..control_structures import extract_target_names, get_control_structure_type, is_control_structure
from ..statement.derivation_edges import bump_derived_lineages
from ._types import (
    IncrementalStartResult as _IncrementalStartResult,
    RestoreCollector,
    SimulationCacheEntry as _SimulationCacheEntry,
    TraceEntry as _TraceEntry,
    apply_collected_mutations,
)

__all__ = ["VirtualLineage"]

logger = logging.getLogger(__name__)

# Canonical built-ins to skip during lineage tracking (mirrors upstream.py).
_BUILTIN_NAMES: frozenset[str] = frozenset({
    'get_ipython', '__builtins__', 'print', 'range',
    'len', 'enumerate', 'zip', 'map', 'filter',
    'sorted', 'reversed', 'list', 'dict', 'set',
    'str', 'int', 'float', 'bool', 'type', 'isinstance',
    'hasattr', 'getattr', 'setattr', 'open', 'sum', 'min', 'max',
    'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'AttributeError', 'RuntimeError', 'Exception',
    'True', 'False', 'None',
})


def _normalize_stmt(s: str) -> str:
    """Strip iteration-context comments and whitespace for code comparison."""
    s = re.sub(r'# __iteration_context__:.*?\n', '', s)
    return s.strip()


# Sentinel placed in user_ns by the forward-probe optimisation so that
# _check_input_lineage_skip sees the variable as "present".  Replaced by
# the real cached value when _restore_from_cache runs.
_FORWARD_PROBE_PLACEHOLDER = object()


class VirtualLineage:
    """Phase 1 of NotebookSimulator: forward simulation + cache probing.

    Owns the simulator-internal caches. Shares ``tracking_state`` dict
    references with :class:`NotebookSimulator` and
    :class:`MismatchClassifier`; pure-phase invariants land in a later
    refactor.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol | None,
        tracking_state: TrackingState,
        compute_hash_fn: Callable[[Any], str] | None = None,
        debug: bool = False,
    ) -> None:
        self.shell = shell
        self.cash_instance = cash_instance
        self.compute_hash_fn = compute_hash_fn
        self.debug = debug
        self.function_tracker: Any | None = None
        self._current_cell_id: str | None = None

        # Shared state refs (same dicts as NotebookSimulator / UpstreamChecker).
        self.set_tracking_state(tracking_state)

        # Simulator-owned caches.
        self._ast_cache: dict[str, ast.Module] = {}
        self._ast_cache_max_size: int = 200
        self._simulation_cache: list[_SimulationCacheEntry] = []
        self._simulation_cell_hashes: dict[int, str] = {}
        self._cell_id_to_last_index: dict[str, int] = {}

        # Buffered TrackingState mutations; orchestrator drains after the phase.
        self._restores = RestoreCollector()

        # Derivation-alias vars bumped during the most recent cache-hit
        # propagation (CAS-115 / CAS-89); read back by _update_virtual_lineage.
        self._last_hit_bumped: set[str] = set()

    def set_tracking_state(self, state: TrackingState) -> None:
        """Re-wire shared state refs (mirrors NotebookSimulator.set_tracking_state)."""
        self._tracking_state = state
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes
        self.variable_lineage = state.variable_lineage
        self.lineage = state.lineage
        self.executed_file_deps = state.executed_file_deps
        self.vars_with_mutation_lineage = state.vars_with_mutation_lineage
        self.executed_input_lineages = state.executed_input_lineages
        self.mutation_verdicts = state.mutation_verdicts

    def _mutation_receivers(self, stmt_code: str, tree: ast.Module) -> set[str]:
        """Receivers of standalone method calls in *tree* that mutate, per the
        runtime's broad-precise classification.

        Statically-known mutators (``MUTATING_METHODS`` / ``inplace=True``) and
        known-pure methods are decided the same way as the runtime, without a
        verdict. For everything else this reads ``mutation_verdicts`` (keyed by
        the statement's ``source_hash`` — the same SHA-256 of the code the
        runtime uses) so the simulation reproduces the runtime's observed
        decision; an unknown verdict (statement not yet executed) is treated as
        mutating (conservative).
        """
        candidates = standalone_method_call_receivers(tree)
        if not candidates:
            return set()
        tier1 = standalone_method_mutation_receivers(tree)
        receivers: set[str] = set()
        source_hash = hashlib.sha256(stmt_code.encode('utf-8')).hexdigest()
        verdict = self.mutation_verdicts.get(source_hash)
        for base, method in candidates:
            if isinstance(self.shell.user_ns.get(base), types.ModuleType):
                continue  # module function call, not a method mutation
            if base in tier1:
                receivers.add(base)
                continue
            if method in KNOWN_PURE_METHODS:
                continue
            if verdict is not None:
                if base in verdict:
                    receivers.add(base)
            else:
                receivers.add(base)  # unknown -> conservative
        return receivers

    def reset_caches(self) -> None:
        """Clear simulation and AST caches."""
        self._simulation_cache.clear()
        self._simulation_cell_hashes.clear()
        self._ast_cache.clear()

    def _get_metadata_only(self, cache_key: str) -> dict | None:
        """Get only metadata for a cache key without deserializing the full value.

        Delegates to ``backend.get_metadata()``, which every
        :class:`cash.backends.CacheBackend` provides (the base supplies a
        ``get()``-discard-value fallback; ``FileBackend`` overrides for a
        cheaper metadata-only read path). Lets callers skip the expensive
        deserialization of large cached objects (e.g. DataFrames) when
        only metadata is needed.
        """
        backend = self.cash_instance.backend if self.cash_instance else None
        if backend is None:
            return None
        return backend.get_metadata(cache_key)

    def _get_cached_ast(self, code: str) -> ast.Module | None:
        """Parse code with AST caching. Returns None on SyntaxError."""
        if code in self._ast_cache:
            return self._ast_cache[code]
        try:
            tree = ast.parse(code)
        except SyntaxError:
            logger.debug("AST parse failed for code: %.80s...", code)
            return None
        if len(self._ast_cache) >= self._ast_cache_max_size:
            keys = list(self._ast_cache.keys())
            for evict_key in keys[:len(keys) // 4]:
                del self._ast_cache[evict_key]
        self._ast_cache[code] = tree
        return tree

    def _check_cell_file_deps(
        self,
        cached_file_deps: dict[str, float],
        idx: int,
    ) -> bool:
        """Return True if any file dep for the cached cell at *idx* has changed."""
        for fpath, stored_mtime in cached_file_deps.items():
            try:
                resolved = resolve_file_dep_path(fpath)
                if resolved is None:
                    return True
                current_mtime = os.path.getmtime(resolved)
                if abs(current_mtime - stored_mtime) > 0.01:
                    if self.debug:
                        logger.debug(
                            "[UPSTREAM_DEBUG] File dependency changed: %s "
                            "(cached mtime=%s, current=%s)",
                            resolved, stored_mtime, current_mtime,
                        )
                    return True
            except OSError:
                return True
        return False

    def _scan_main_cache_for_changes(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> tuple[int, bool]:
        """Scan the main simulation cache to find the first changed cell.

        Returns ``(first_changed_cell, cache_had_hash_mismatch)``.
        """
        first_changed_cell = 0
        cache_had_hash_mismatch = False
        for idx in range(min(current_cell_idx, len(self._simulation_cache))):
            cell_code = notebook_cells[idx].replace('\r\n', '\n')
            cell_hash = hashlib.sha256(cell_code.encode('utf-8')).hexdigest()
            cached = self._simulation_cache[idx]
            if cached.cell_code_hash != cell_hash:
                cache_had_hash_mismatch = True
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] Hash mismatch in cell %d "
                        "(cached=%s, current=%s). Re-simulating from here.",
                        idx, cached.cell_code_hash[:12], cell_hash[:12],
                    )
                break
            cached_file_deps = cached.cell_file_deps
            if cached_file_deps and self._check_cell_file_deps(cached_file_deps, idx):
                cache_had_hash_mismatch = True
                break
            first_changed_cell = idx + 1
        return first_changed_cell, cache_had_hash_mismatch

    def _check_lightweight_hash_cache(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> bool:
        """Check the lightweight hash cache for changes beyond the main cache range.

        Returns True if a hash mismatch was detected.
        """
        cache_range_end = min(current_cell_idx, len(self._simulation_cache)) if self._simulation_cache else 0
        for idx in range(cache_range_end, current_cell_idx):
            if idx not in self._simulation_cell_hashes:
                continue
            cell_code = notebook_cells[idx].replace('\r\n', '\n')
            cell_hash = hashlib.sha256(cell_code.encode('utf-8')).hexdigest()
            if self._simulation_cell_hashes[idx] != cell_hash:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] Hash mismatch in cell %d "
                        "(detected via lightweight hash cache, main cache truncated)",
                        idx,
                    )
                return True
        return False

    def _restore_cached_state(
        self,
        first_changed_cell: int,
    ) -> tuple[dict[str, str], set[str], list, set[str], set[str]]:
        """Restore virtual state from cached entries up to *first_changed_cell*.

        Returns ``(virtual_lineage, virtual_modules, simulation_trace,
        vars_mutated_by_loops, vars_with_stale_files)``.
        """
        cached_entry = self._simulation_cache[first_changed_cell - 1]
        virtual_lineage = dict(cached_entry.virtual_lineage)
        virtual_modules = set(cached_entry.virtual_modules)
        simulation_trace: list = []
        vars_mutated_by_loops: set[str] = set()
        vars_with_stale_files: set[str] = set()
        for ci in range(first_changed_cell):
            simulation_trace.extend(self._simulation_cache[ci].trace_segment)
            vars_mutated_by_loops.update(self._simulation_cache[ci].vars_mutated_by_loops)
            vars_with_stale_files.update(self._simulation_cache[ci].vars_with_stale_files)
        if self.debug:
            logger.debug(
                "[UPSTREAM_DEBUG] Incremental simulation: reusing cache for cells 0-%d, simulating from cell %d",
                first_changed_cell - 1, first_changed_cell,
            )
        return virtual_lineage, virtual_modules, simulation_trace, vars_mutated_by_loops, vars_with_stale_files

    def _find_incremental_start(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> _IncrementalStartResult:
        """Find the first upstream cell that changed since last simulation.

        Compares cached simulation hashes with current notebook cells and checks
        file dependency mtimes. Returns the index to start re-simulation from,
        along with restored cached state (virtual lineage, modules, trace, etc.).
        """
        simulation_trace: list = []
        virtual_lineage: dict[str, str] = {}
        virtual_modules: set[str] = set()
        vars_mutated_by_loops: set[str] = set()
        vars_with_stale_files: set[str] = set()

        first_changed_cell = 0
        had_prior_cache = bool(self._simulation_cache)
        cache_had_hash_mismatch = False

        if self.debug:
            logger.debug(
                "[UPSTREAM_DEBUG] _simulate_and_find_changes: current_cell_idx=%d, "
                "had_prior_cache=%s, cache_size=%d, cell_hashes_size=%d",
                current_cell_idx, had_prior_cache,
                len(self._simulation_cache) if self._simulation_cache else 0,
                len(self._simulation_cell_hashes),
            )

        if self._simulation_cache:
            first_changed_cell, cache_had_hash_mismatch = self._scan_main_cache_for_changes(
                current_cell_idx, notebook_cells
            )

        # Check the lightweight hash cache for cells beyond the main cache range.
        if not cache_had_hash_mismatch and self._simulation_cell_hashes:
            if self._check_lightweight_hash_cache(current_cell_idx, notebook_cells):
                cache_had_hash_mismatch = True

        # Restore cached state for cells before the first change, regardless of
        # what type of change was detected (code hash OR file dep staleness).
        # Without this, stale file deps would cause ALL cached state to be lost,
        # even for cells before the stale cell.
        if first_changed_cell > 0 and self._simulation_cache and first_changed_cell <= len(self._simulation_cache):
            (virtual_lineage, virtual_modules,
             simulation_trace, vars_mutated_by_loops,
             vars_with_stale_files) = self._restore_cached_state(first_changed_cell)

        new_cache_entries = list(self._simulation_cache[:first_changed_cell]) if self._simulation_cache else []

        return _IncrementalStartResult(
            first_changed_cell=first_changed_cell,
            had_prior_cache=had_prior_cache,
            cache_had_hash_mismatch=cache_had_hash_mismatch,
            simulation_trace=simulation_trace,
            virtual_lineage=virtual_lineage,
            virtual_modules=virtual_modules,
            new_cache_entries=new_cache_entries,
            vars_mutated_by_loops=vars_mutated_by_loops,
            vars_with_stale_files=vars_with_stale_files,
        )

    def _collect_notebook_statements(self, notebook_cells: list[str]) -> set[str]:
        """Collect all normalized statement codes from notebook cells.

        Used to distinguish downstream statements from unsaved extensions.
        """
        all_notebook_stmts: set[str] = set()
        for cell_code in notebook_cells:
            try:
                clean_code = CodeAnalyzer.strip_magics(cell_code.replace('\r\n', '\n'))
                if clean_code.strip():
                    tree = self._get_cached_ast(clean_code)
                    if tree is not None:
                        for node in tree.body:
                            try:
                                stmt_code = ast.unparse(node)
                                all_notebook_stmts.add(stmt_code.strip())
                            except (TypeError, ValueError):
                                logger.debug("Failed to unparse AST node in notebook cell")
            except (SyntaxError, ValueError):
                logger.debug("Failed to parse notebook cell for unsaved extension check")
        return all_notebook_stmts

    def _reapply_unsaved_extensions(
        self,
        broken_vars: set[str],
        vars_updated_by_trace: set[str],
        simulation_trace: list,
        notebook_cells: list[str],
        statements_to_reexecute: list[str],
    ) -> None:
        """Re-apply unsaved extension code for broken variables.

        If a broken variable's producing code is NOT in the notebook (unsaved
        extension), schedule it for re-execution â€” unless the trace already
        updated that variable.
        """
        all_notebook_stmts = self._collect_notebook_statements(notebook_cells)

        for var_name in broken_vars:
            if var_name in vars_updated_by_trace:
                continue

            if var_name in self.executed_cell_codes:
                mem_code = self.executed_cell_codes[var_name]

                is_in_trace = False
                for stmt, _, _, _, _, _ in simulation_trace:
                    if stmt.strip() == mem_code.strip():
                        is_in_trace = True
                        break

                if not is_in_trace:
                    if mem_code.strip() in all_notebook_stmts:
                        logger.debug("[UPSTREAM] Skipping downstream statement for '%s'", var_name)
                        continue

                    if mem_code not in statements_to_reexecute:
                         logger.debug("[UPSTREAM] Re-applying unsaved extension for '%s'", var_name)
                         statements_to_reexecute.append(mem_code)

    def _propagate_loop_derived_vars(
        self,
        vars_mutated_by_loops: set[str],
        simulation_trace: list,
    ) -> set[str]:
        """Walk forward from loop-mutated vars to include transitive dependents.

        Returns the full set of variables derived from loop mutations (including
        the original loop-mutated vars). These are trusted in memory rather than
        replaced with stale cached values.
        """
        if not vars_mutated_by_loops:
            return set()
        vars_derived = set(vars_mutated_by_loops)
        for _stmt_code, outputs, inputs, _, _, _ in simulation_trace:
            if inputs & vars_derived:
                vars_derived.update(outputs)
        if self.debug and vars_derived - vars_mutated_by_loops:
            logger.debug(
                "[UPSTREAM_DEBUG] Variables transitively derived from loop mutations: %s",
                vars_derived - vars_mutated_by_loops,
            )
        return vars_derived

    def _skipped_stmt_metric(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        inputs: set[str],
        input_hashes: dict[str, str],
        virtual_modules: set[str],
    ) -> dict | None:
        """Return a metric dict for a single skipped statement, or ``None`` on error."""
        if self.debug:
            logger.debug("[UPSTREAM] Checking skipped stmt [%d]: %.30s...", i, stmt_code)
        try:
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker if hasattr(self, 'function_tracker') else None,
                    virtual_lineage=input_hashes,
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                ),
                outputs=outputs,
            )
            metadata = self._get_metadata_only(cache_key)
            if metadata:
                saved_time = metadata.get('execution_time', 0.0)
                is_metadata_only = metadata.get('metadata_only', False)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] Skipped stmt [%d] hit cache. Saved: %ss (metadata_only=%s)",
                        i, saved_time, is_metadata_only,
                    )
                entry: dict = {
                    'code': stmt_code,
                    'status': CacheStatus.SKIPPED,
                    'saved_time': saved_time,
                    'is_upstream': True,
                    'source': 'Skipped',
                    'position': i,
                    'has_cache': True,
                }
                if 'storage' in metadata:
                    entry['storage'] = metadata['storage']
                return entry
            if self.debug:
                logger.debug("[UPSTREAM] Skipped stmt [%d] miss cache. Key: %s", i, cache_key)
            return {
                'code': stmt_code,
                'status': CacheStatus.SKIPPED,
                'saved_time': 0.0,
                'is_upstream': True,
                'source': 'Skipped',
                'position': i,
                'has_cache': False,
            }
        except (KeyError, TypeError, OSError, ValueError) as e:
            if self.debug:
                logger.debug("[UPSTREAM] Error checking skipped stmt: %s", e)
            return None

    def _collect_skipped_statement_metrics(
        self,
        simulation_trace: list,
        stmts_to_run_indices: list[int],
        restored_statements_info: list,
        virtual_modules: set[str],
        stmt_lookup_times: dict[str, float],
    ) -> list[dict]:
        """Identify implicitly skipped statements and collect their cache metrics.

        Skipped statements are dependencies of restored variables that were
        neither scheduled for execution nor explicitly restored. Returns a list
        of metric dicts to be appended to ``restored_statements_info``.
        """
        executed_indices = set(stmts_to_run_indices)
        restored_indices = set()
        restored_outputs = set()
        for info in restored_statements_info:
            if 'position' in info:
                restored_indices.add(info['position'])
                if 'restored_vars' in info:
                    restored_outputs.update(info['restored_vars'])

        dependency_chain: set[int] = set()
        if restored_outputs:
            needed = set(restored_outputs)
            for i in range(len(simulation_trace) - 1, -1, -1):
                _stmt_code, outputs, inputs, _ih, _pl, _ = simulation_trace[i]
                if outputs & needed:
                    dependency_chain.add(i)
                    needed.update(inputs)

        skipped_metrics: list[dict] = []
        for i, (stmt_code, outputs, inputs, input_hashes, _produced_lineages, _) in enumerate(simulation_trace):
            if i in executed_indices or i in restored_indices or i not in dependency_chain:
                continue
            entry = self._skipped_stmt_metric(i, stmt_code, outputs, inputs, input_hashes, virtual_modules)
            if entry is not None:
                skipped_metrics.append(entry)
        return skipped_metrics

    def _is_reinit_to_skip(
        self,
        idx: int,
        simulation_trace: list,
        scheduled_iteration_outputs: dict[str, list],
        vars_mutated_by_loops: set[str],
        iteration_context_pattern: re.Pattern[str],
        fully_rerun_mutated: set[str],
    ) -> bool:
        """Return True if the statement at *idx* is an accumulator init that should be skipped.

        Skips when the statement initialises to an empty container (e.g. ``x = {}``)
        but the accumulator already has data in memory, to avoid wiping state.
        """
        stmt_code, outputs, _inputs, _, _, _ = simulation_trace[idx]
        if iteration_context_pattern.search(stmt_code):
            return False
        if len(outputs) != 1:
            return False
        out_var = list(outputs)[0]
        if out_var in fully_rerun_mutated:
            # When the loop that mutates out_var is itself fully re-executed, the
            # init must run alongside it, else the accumulation doubles; skipping
            # is only safe for pure incremental extension of a cached loop.
            return False
        is_loop_updated = (out_var in scheduled_iteration_outputs or out_var in vars_mutated_by_loops)
        if not is_loop_updated:
            return False
        stripped = stmt_code.strip()
        empty_init_pattern = re.compile(
            rf'^{re.escape(out_var)}\s*=\s*(\{{\}}|\[\]|set\(\)|dict\(\)|list\(\)|frozenset\(\))$'
        )
        if not empty_init_pattern.match(stripped):
            return False
        if out_var not in self.shell.user_ns:
            return False
        existing_val = self.shell.user_ns[out_var]
        try:
            is_non_empty = bool(existing_val)
        except (ValueError, TypeError):
            is_non_empty = False
        if is_non_empty:
            if self.debug:
                logger.debug(
                    "[UPSTREAM] Skipping accumulator init '%.40s' - already has %d items in memory",
                    stmt_code, len(existing_val),
                )
            return True
        return False

    def _loop_vars_fully_rescheduled(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        vars_mutated_by_loops: set[str],
        iteration_context_pattern: re.Pattern[str],
    ) -> set[str]:
        """Loop-mutated vars whose mutation is scheduled OUTSIDE a cached iteration-context body (=> full re-run)."""
        if not vars_mutated_by_loops:
            return set()
        patterns = {
            mv: re.compile(rf'\b{re.escape(mv)}\s*(?:\.\s*\w+\s*\(|\[[^\]]*\]\s*=(?!=))')
            for mv in vars_mutated_by_loops
        }
        fully_rerun_mutated: set[str] = set()
        for idx in stmts_to_run_indices:
            stmt_code = simulation_trace[idx][0]
            if iteration_context_pattern.search(stmt_code):
                continue
            for mv, pat in patterns.items():
                if mv not in fully_rerun_mutated and pat.search(stmt_code):
                    fully_rerun_mutated.add(mv)
        return fully_rerun_mutated

    def _filter_accumulator_reinits(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        vars_mutated_by_loops: set[str],
    ) -> list[int]:
        """Remove accumulator initialization statements that would reset existing state.

        When adding new items to a cached loop, the backward scan may schedule
        the initialization (e.g. ``ticker_stats = {}``) for execution. If the
        accumulator already exists in memory with data, re-running the init would
        wipe accumulated state. Returns a filtered copy of *stmts_to_run_indices*.
        """
        iteration_context_pattern = re.compile(r'# __iteration_context__: ([a-f0-9]+)')
        scheduled_iteration_outputs: dict[str, list] = {}
        for idx in stmts_to_run_indices:
            stmt_code, outputs, *_ = simulation_trace[idx]
            match = iteration_context_pattern.search(stmt_code)
            if match:
                for out in outputs:
                    scheduled_iteration_outputs.setdefault(out, []).append(idx)

        fully_rerun_mutated = self._loop_vars_fully_rescheduled(
            stmts_to_run_indices, simulation_trace, vars_mutated_by_loops,
            iteration_context_pattern,
        )

        # A fully re-run loop replays its in-place mutations (.append / [k]=)
        # onto whatever the accumulator currently holds.  If the empty-container
        # init was never scheduled (the backward scan often schedules only the
        # loop body, treating the accumulator output as already satisfied), the
        # replay doubles the accumulated value.  Schedule the missing init so it
        # runs alongside the loop.  (Pure incremental extension keeps the init
        # unscheduled and is handled by the removal pass below.)
        stmts_to_run_indices = self._schedule_missing_accumulator_inits(
            stmts_to_run_indices, simulation_trace, fully_rerun_mutated,
        )

        indices_to_remove: set[int] = set()
        for idx in stmts_to_run_indices:
            if self._is_reinit_to_skip(
                idx, simulation_trace, scheduled_iteration_outputs,
                vars_mutated_by_loops, iteration_context_pattern,
                fully_rerun_mutated,
            ):
                indices_to_remove.add(idx)

        if indices_to_remove:
            return [idx for idx in stmts_to_run_indices if idx not in indices_to_remove]
        return stmts_to_run_indices

    def _schedule_missing_accumulator_inits(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        fully_rerun_mutated: set[str],
    ) -> list[int]:
        """Schedule empty-container inits for fully-re-run loop accumulators.

        For each var in *fully_rerun_mutated*, if its ``x = []`` / ``x = {}`` /
        ``x = set()`` init appears in the trace but is not already scheduled,
        add it. This prevents the loop's in-place mutations from replaying onto
        a stale value (doubling). Only single-output empty-container inits are
        added, so non-init assignments are never pulled in.
        """
        if not fully_rerun_mutated:
            return stmts_to_run_indices
        scheduled = set(stmts_to_run_indices)
        empty_init_patterns = {
            mv: re.compile(
                rf'^{re.escape(mv)}\s*=\s*(\{{\}}|\[\]|set\(\)|dict\(\)|list\(\)|frozenset\(\))$'
            )
            for mv in fully_rerun_mutated
        }
        additional: list[int] = []
        for idx, entry in enumerate(simulation_trace):
            if idx in scheduled:
                continue
            stmt_code, outputs = entry[0], entry[1]
            if len(outputs) != 1:
                continue
            out_var = next(iter(outputs))
            pat = empty_init_patterns.get(out_var)
            if pat is not None and pat.match(stmt_code.strip()):
                additional.append(idx)
                scheduled.add(idx)
        if additional:
            return stmts_to_run_indices + additional
        return stmts_to_run_indices

    def _check_loop_derived_trust_override(
        self,
        upstream_has_modifications: bool,
        vars_mutated_by_loops: set[str],
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if loop-derived variable trust should be overridden.

        This happens when a loop-mutated variable was produced by code that is
        NOT in the current simulation trace (unsaved edit or stale execution).
        """
        if upstream_has_modifications or not vars_mutated_by_loops:
            return False
        for mv in vars_mutated_by_loops:
            if mv not in self.executed_cell_codes:
                continue
            exec_code = re.sub(r'# __iteration_context__:.*?\n', '', self.executed_cell_codes[mv]).strip()
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Checking loop trust for '%s': exec_code=%s", mv, repr(exec_code[:60]))
                matching = [sc for sc in simulation_trace_codes if exec_code in sc or sc in exec_code]
                logger.debug("[UPSTREAM_DEBUG]   Partial matches in simulation_trace_codes: %s", [repr(m[:60]) for m in matching])
            if exec_code and exec_code not in simulation_trace_codes:
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Loop-mutated var '%s' was produced by code "
                          "not found on disk (unsaved edit or stale execution). Distrusting ALL loop-derived vars.", mv)
                return True
        return False

    def _build_loop_var_input_lineages(
        self,
        simulation_trace: list,
        vars_derived_from_loops: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> dict[str, dict[str, str]]:
        """Return a mapping of loop-derived variable â†’ its data-input virtual lineages.

        Used to detect when loop inputs change (e.g., N=10â†’20) even when the
        producing code is unchanged on disk.
        """
        loop_var_input_lineages: dict[str, dict[str, str]] = {}
        for _stmt_code, outputs, inputs, _input_hashes, _produced_lineages, _ in simulation_trace:
            for out in outputs:
                if out in vars_derived_from_loops:
                    data_input_lineages: dict[str, str] = {}
                    for inp in inputs:
                        if inp in virtual_modules:
                            continue
                        if inp in virtual_lineage:
                            data_input_lineages[inp] = virtual_lineage[inp]
                    loop_var_input_lineages[out] = data_input_lineages
        return loop_var_input_lineages

    def _build_simulation_trace_codes(self, simulation_trace: list) -> set[str]:
        """Return the set of normalised statement codes present in *simulation_trace*.

        Includes body-level statements from control structures so that
        per-iteration cache entries (which record body statements rather than
        the whole for-loop) are matched correctly.
        """
        simulation_trace_codes: set[str] = set()
        for stmt_code, _, _, _, _, _ in simulation_trace:
            normalized = re.sub(r'# __iteration_context__:.*?\n', '', stmt_code).strip()
            simulation_trace_codes.add(normalized)
            try:
                tree = self._get_cached_ast(normalized)
                if tree and len(tree.body) == 1 and is_control_structure(tree.body[0]):
                    for body_node in self._iter_body_nodes(tree.body[0]):
                        try:
                            body_code = ast.unparse(body_node).strip()
                            simulation_trace_codes.add(body_code)
                        except (ValueError, TypeError):
                            logger.debug("[UPSTREAM] Failed to unparse body node in simulation trace")
            except (SyntaxError, ValueError):
                logger.debug("[UPSTREAM] Failed to parse control structure for simulation trace codes")
        return simulation_trace_codes

    def _update_stale_file_deps(
        self,
        inputs: list[str],
        outputs: set[str],
        files_stale: bool,
        vars_with_stale_files: set[str],
    ) -> None:
        """Mark *outputs* as stale if the statement or any input is stale."""
        stmt_has_stale_deps = files_stale
        if not stmt_has_stale_deps:
            for inp in inputs:
                if inp in vars_with_stale_files:
                    stmt_has_stale_deps = True
                    break
        if stmt_has_stale_deps:
            vars_with_stale_files.update(outputs)

    def _simulate_one_node(
        self,
        i: int,
        node: ast.AST,
        cell_stmt_occurrence_counts: dict,
        virtual_lineage: dict,
        virtual_modules: set,
        simulation_trace: list,
        vars_mutated_by_loops: set,
        vars_with_stale_files: set,
        stmt_lookup_times: dict,
        loop_target_vars: set,
        cell_file_deps: dict,
    ) -> None:
        """Simulate a single AST statement node, updating all mutable state in-place.

        Returns without doing anything for control structures (they are handled
        by ``_simulate_control_structure`` directly).
        """
        try:
            if is_control_structure(node):
                self._simulate_control_structure(node, virtual_lineage, virtual_modules, simulation_trace, stmt_lookup_times, vars_mutated_by_loops, loop_target_vars=loop_target_vars)
                return

            stmt_code = ast.unparse(node)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("[UPSTREAM] Error processing node in cell %d: %s", i, e)
            raise

        occ = cell_stmt_occurrence_counts.get(stmt_code, 0)
        cell_stmt_occurrence_counts[stmt_code] = occ + 1
        occurrence_index = occ  # 0-based

        inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        input_hashes: dict[str, str] = {}
        for inp in inputs:
            if inp in virtual_lineage:
                input_hashes[inp] = virtual_lineage[inp]
            elif inp in self.variable_lineage:
                input_hashes[inp] = self.variable_lineage[inp]

        outputs, lookup_time, files_stale, stmt_file_deps = self._update_virtual_lineage(
            stmt_code, virtual_lineage, virtual_modules, occurrence_index=occurrence_index,
        )

        if stmt_file_deps:
            cell_file_deps.update(stmt_file_deps)

        self._update_stale_file_deps(inputs, outputs, files_stale, vars_with_stale_files)

        if outputs:
            produced_lineages = {out: virtual_lineage[out] for out in outputs if out in virtual_lineage}
            simulation_trace.append(_TraceEntry(stmt_code, outputs, inputs, input_hashes, produced_lineages, files_stale))
            if lookup_time > 0:
                stmt_lookup_times[stmt_code] = lookup_time
        else:
            # No-output statements normally stay out of the trace, but a bare
            # file-writing expression (``df.to_csv(p)``) IS upstream state a
            # reader depends on: without a trace entry the planner can never
            # schedule an edited/stale writer (CAS-81/82). Empty outputs keep
            # the backward scan indifferent to the entry.
            from ..cacheability import statement_writes_files
            if statement_writes_files(stmt_code):
                simulation_trace.append(_TraceEntry(stmt_code, outputs, inputs, input_hashes, {}, files_stale))

    def _simulate_one_cell(
        self,
        i: int,
        cell_code: str,
        simulation_trace: list,
        virtual_lineage: dict,
        virtual_modules: set,
        new_cache_entries: list,
        vars_mutated_by_loops: set,
        vars_with_stale_files: set,
        stmt_lookup_times: dict,
        loop_target_vars: set,
    ) -> None:
        """Simulate a single cell and append a cache entry to *new_cache_entries*.

        Mutates *simulation_trace*, *virtual_lineage*, *virtual_modules*,
        *new_cache_entries*, *vars_mutated_by_loops*, *vars_with_stale_files*,
        and *stmt_lookup_times* in-place.
        """
        cell_hash = hashlib.sha256(cell_code.encode('utf-8')).hexdigest()
        trace_start = len(simulation_trace)
        cell_file_deps: dict = {}

        # Model ``%reset`` / ``%reset -f`` as a full namespace wipe BEFORE the
        # strip_magics empty-cell short-circuit below (a reset cell strips to
        # empty). Like ``del`` it clears ``user_ns`` but not ``variable_lineage``;
        # position-scoping (the simulator only replays cells 0..current) means a
        # reset ABOVE the target wipes the virtual state so an above-the-reset
        # consumer's inputs are reconstructed, while a reset BELOW is never
        # simulated. Without this the liveness gate (CAS-94) would resurrect a
        # reset variable as a phantom restore (test_reset_magic_no_phantom_restore).
        for line in cell_code.split('\n'):
            if line.strip().startswith('%reset'):
                virtual_lineage.clear()
                virtual_modules.clear()

        try:
            clean_cell_code = CodeAnalyzer.strip_magics(cell_code)
            if not clean_cell_code.strip():
                new_cache_entries.append(_SimulationCacheEntry(
                    cell_code_hash=cell_hash,
                    virtual_lineage=dict(virtual_lineage),
                    virtual_modules=set(virtual_modules),
                    trace_segment=[],
                    vars_mutated_by_loops=set(),
                    vars_with_stale_files=set(),
                    cell_file_deps={},
                ))
                return

            tree = self._get_cached_ast(clean_cell_code)
            if tree is None:
                ast.parse(clean_cell_code)  # will raise SyntaxError

            cell_stmt_occurrence_counts: dict = {}

            for node in tree.body:
                # A top-level ``raise`` unconditionally aborts the cell — every
                # statement after it is dead code that never runs in a real
                # from-start execution. Stop here so the simulation does not
                # register a post-raise assignment (``z = 1; raise; z = 2``) as
                # the variable's producer and later reconstruct that dead value
                # (CAS-64).
                if isinstance(node, ast.Raise):
                    break
                self._simulate_one_node(
                    i, node, cell_stmt_occurrence_counts,
                    virtual_lineage, virtual_modules, simulation_trace,
                    vars_mutated_by_loops, vars_with_stale_files,
                    stmt_lookup_times, loop_target_vars, cell_file_deps,
                )

        except SyntaxError:
            logger.debug("[UPSTREAM] Syntax error in cell %d, raising error.", i)
            raise  # Re-raise syntax error to stop execution
        except (KeyError, TypeError, ValueError, OSError, AttributeError) as e:
            logger.debug("[UPSTREAM] Error simulating cell %d: %s", i, e)
            raise

        cell_trace_segment = simulation_trace[trace_start:]
        new_cache_entries.append(_SimulationCacheEntry(
            cell_code_hash=cell_hash,
            virtual_lineage=dict(virtual_lineage),
            virtual_modules=set(virtual_modules),
            trace_segment=cell_trace_segment,
            vars_mutated_by_loops=set(vars_mutated_by_loops),
            vars_with_stale_files=set(vars_with_stale_files),
            cell_file_deps=dict(cell_file_deps),
        ))

    def _simulate_cells_pass1(
        self,
        first_changed_cell: int,
        current_cell_idx: int,
        notebook_cells: list[str],
        simulation_trace: list,
        virtual_lineage: dict,
        virtual_modules: set,
        new_cache_entries: list,
        vars_mutated_by_loops: set,
        vars_with_stale_files: set,
        stmt_lookup_times: dict,
        loop_target_vars: set,
    ) -> None:
        """Run pass-1 simulation for cells *first_changed_cell*..*current_cell_idx* and update caches."""
        for i in range(first_changed_cell, current_cell_idx):
            cell_code = notebook_cells[i].replace('\r\n', '\n')
            self._simulate_one_cell(
                i, cell_code,
                simulation_trace, virtual_lineage, virtual_modules,
                new_cache_entries, vars_mutated_by_loops, vars_with_stale_files,
                stmt_lookup_times, loop_target_vars,
            )

        # Update simulation cache for future incremental simulation.
        # NOTE: We only store entries for cells 0..(current_cell_idx-1).
        # Entries beyond that are discarded to avoid stale lineage data.
        # For hash change detection across intermediate cell runs, we use
        # _simulation_cell_hashes (a separate lightweight structure).
        self._simulation_cache = new_cache_entries

        # This persists across intermediate cell runs so that a later cell can
        # detect code changes in cells that were truncated from the main cache.
        for idx, entry in enumerate(new_cache_entries):
            self._simulation_cell_hashes[idx] = entry.cell_code_hash

        # Also record the CURRENT cell's hash so that a later cell (e.g., cell 3
        # running after cell 2 in a run_all()) sees the up-to-date hash and
        # doesn't falsely detect a modification from a stale hash left over
        # from a previous run_all().
        if current_cell_idx < len(notebook_cells):
            current_cell_code = notebook_cells[current_cell_idx].replace('\r\n', '\n')
            self._simulation_cell_hashes[current_cell_idx] = hashlib.sha256(
                current_cell_code.encode('utf-8')
            ).hexdigest()

    def _collect_loop_mutation_info(
        self,
        node: ast.AST,
        loop_target_vars: set[str],
        vars_mutated_by_loops: set[str],
    ) -> set[str]:
        """Collect control-body mutation info and return the mutated vars for this node.

        Updates *loop_target_vars* (for ``ast.For``) and *vars_mutated_by_loops*
        in place. Covers ALL control structures, not just loops: a var mutated in
        place inside an ``if`` / ``with`` / ``try`` body (``if cond:
        items.append(x)``) is not reported as a static output by
        ``CodeAnalyzer``, so without this its virtual lineage would stay stale and
        a downstream cell reading it would serve a pre-mutation value (CAS-66).
        Treated like a loop mutation so it is trusted in memory and its lineage is
        bumped, matching the runtime's ``update_lineage_after_execution``.
        """
        mutated_vars: set[str] = set()
        if isinstance(node, ast.For):
            target_names = extract_target_names(node.target)
            loop_target_vars.update(target_names)
            mutated_vars = self._find_loop_mutated_vars(node.body, set(target_names))
            vars_mutated_by_loops.update(mutated_vars)
        elif isinstance(node, ast.While):
            mutated_vars = self._find_loop_mutated_vars(node.body, set())
            vars_mutated_by_loops.update(mutated_vars)
        elif isinstance(node, (ast.If, ast.With, ast.AsyncWith, ast.Try)):
            direct_body: list = []
            for attr in ('body', 'orelse', 'finalbody'):
                direct_body.extend(getattr(node, attr, []) or [])
            for handler in getattr(node, 'handlers', []) or []:
                direct_body.extend(handler.body)
            mutated_vars = self._find_loop_mutated_vars(direct_body, set())
            vars_mutated_by_loops.update(mutated_vars)
        return mutated_vars

    def _apply_loop_mutation_lineages(
        self,
        mutated_vars: set[str],
        outputs: set[str],
        inputs: set[str],
        stmt_code: str,
        input_hashes: dict[str, str],
        virtual_lineage: dict[str, str],
    ) -> set[str]:
        """Update virtual lineage for loop-mutated vars and return the extra output set."""
        extra_outputs: set[str] = set()
        if not mutated_vars:
            return extra_outputs
        source_hash = hashlib.sha256(stmt_code.encode('utf-8')).hexdigest()
        input_lineages_sorted = sorted(input_hashes.values())
        for mv in mutated_vars:
            if mv not in outputs and mv in inputs:
                combined = source_hash + ':' + ':'.join(input_lineages_sorted)
                new_lineage = hashlib.sha256(combined.encode('utf-8')).hexdigest()
                virtual_lineage[mv] = new_lineage
                extra_outputs.add(mv)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] Loop-mutated var '%s' virtual lineage updated to %s...",
                        mv, new_lineage[:12],
                    )
        return extra_outputs

    def _simulate_control_structure(
        self,
        node: ast.AST,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        simulation_trace: list[tuple],
        stmt_lookup_times: dict[str, float],
        vars_mutated_by_loops: set[str] = None,
        parent_context: dict[str, Any] | None = None,
        loop_target_vars: set[str] = None
    ) -> None:
        """
        Simulate execution of a control structure as a single unit.

        Control structures are no longer decomposed per-iteration/per-branch.
        The entire control structure is treated as one statement for simulation
        purposes, matching the runtime behavior in ControlStructureProcessor.
        """
        if vars_mutated_by_loops is None:
            vars_mutated_by_loops = set()
        if loop_target_vars is None:
            loop_target_vars = set()

        # Treat the entire control structure as a single statement
        stmt_code = ast.unparse(node)

        inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        input_hashes = {}
        for inp in inputs:
            if inp in virtual_lineage:
                input_hashes[inp] = virtual_lineage[inp]
            elif inp in self.variable_lineage:
                input_hashes[inp] = self.variable_lineage[inp]

        outputs, lookup_time, files_stale, _ = self._update_virtual_lineage(stmt_code, virtual_lineage, virtual_modules)

        mutated_vars = self._collect_loop_mutation_info(node, loop_target_vars, vars_mutated_by_loops)

        # CRITICAL FIX: Update virtual lineage for variables mutated inside loops.
        # CodeAnalyzer doesn't detect loop-mutated vars (like `groups` in
        # `for k, v in data: groups.setdefault(k, []).append(v)`) as outputs,
        # so their virtual lineage stays stale.  We compute a new lineage hash
        # that depends on the loop's code and input lineages, ensuring downstream
        # consumers (like `sums = {k: sum(v) for k, v in groups.items()}`) get
        # a different cache key when the loop's inputs change.
        extra_outputs = self._apply_loop_mutation_lineages(
            mutated_vars, outputs, inputs, stmt_code, input_hashes, virtual_lineage
        )

        all_outputs = outputs | extra_outputs

        if self.debug:
            cs_type = get_control_structure_type(node) if node else 'unknown'
            logger.debug("[UPSTREAM_DEBUG] Simulating %s as single unit: %s... Outputs: %s", cs_type, stmt_code[:60], all_outputs)

        if all_outputs:
            produced_lineages = {out: virtual_lineage[out] for out in all_outputs if out in virtual_lineage}
            simulation_trace.append(_TraceEntry(stmt_code, all_outputs, inputs, input_hashes, produced_lineages, files_stale))
            if lookup_time > 0:
                stmt_lookup_times[stmt_code] = lookup_time

    # -- Helpers for _update_virtual_lineage ----------------------------------

    @staticmethod
    def _validate_file_freshness(
        hist_files: dict[str, Any], debug: bool = False
    ) -> bool:
        """Return True if all historical file dependencies are still fresh.

        Each entry is ``{path: {'mtime': ..., 'size': ...}}``. When ``size``
        is recorded it is checked too — that catches rewrites within a
        single mtime tick on coarse-resolution filesystems (HFS+/APFS,
        some ext4 configs).
        """
        for fpath, stored in hist_files.items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                if debug:
                    logger.debug("[UPSTREAM] Forward prop failed: Miss file %s", fpath)
                return False
            stored_mtime, stored_size = split_file_dep_value(stored)
            try:
                cur_stat = os.stat(resolved)
            except OSError:
                return False
            delta = abs(cur_stat.st_mtime - stored_mtime)
            if delta > 0.01:
                if debug:
                    logger.debug("[UPSTREAM] Forward prop failed: Stale file %s (delta=%.4fs)", resolved, delta)
                return False
            if stored_size is not None and cur_stat.st_size != stored_size:
                if debug:
                    logger.debug("[UPSTREAM] Forward prop failed: Resized file %s (%d -> %d)", resolved, stored_size, cur_stat.st_size)
                return False
        return True

    def _resolve_input_lineage(
        self,
        inp: str,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> str | None:
        """Resolve the lineage hash for a single input variable.

        Priority: virtual_lineage â†’ variable_lineage â†’ hash from user_ns.
        Returns ``None`` if the input cannot be resolved.
        """
        if inp in virtual_lineage:
            return virtual_lineage[inp]
        if inp in self.variable_lineage:
            return self.variable_lineage[inp]

        val = self.shell.user_ns.get(inp)
        if val is None:
            return None

        try:
            if self.compute_hash_fn:
                return self.compute_hash_fn(val)
            return hashlib.sha256(str(val).encode('utf-8')).hexdigest()
        except (TypeError, ValueError):
            logger.debug("[UPSTREAM] Failed to compute hash for input '%s'", inp)
            return None

    def _hash_module_with_deps(self, out: str, mod_file: str, function_tracker: Any) -> str:
        """Hash the source of module *out* plus its tracked dependency files.

        Returns a lineage component string like ``:mod_src:<hex>`` on success,
        or an empty string on I/O failure.
        """
        try:
            hasher = hashlib.sha256()
            with open(mod_file, 'rb') as mf:
                hasher.update(mf.read())
            dep_files: set = set()
            for dep_path, parent_mods in getattr(function_tracker, '_dep_file_to_parents', {}).items():
                if out in parent_mods:
                    dep_files.add(dep_path)
            for dep_path in sorted(dep_files):
                if os.path.isfile(dep_path):
                    try:
                        with open(dep_path, 'rb') as df:
                            hasher.update(df.read())
                    except OSError:
                        logger.debug("Cannot read dependency file for module hash: %s", dep_path)
            return f":mod_src:{hasher.hexdigest()}"
        except OSError:
            logger.debug("Cannot read module file for source hash: %s", mod_file)
            return ""

    def _compute_module_source_hash(self, outputs: set[str]) -> str:
        """Return a module-source lineage component string for module outputs.

        Scans *outputs* for module-type objects whose source is tracked by the
        function_tracker and hashes their file contents (plus transitive deps).
        """
        function_tracker = self.function_tracker if hasattr(self, 'function_tracker') else None
        if function_tracker is None:
            return ""

        for out in outputs:
            val = self.shell.user_ns.get(out)
            if val is None or not isinstance(val, types.ModuleType):
                continue
            mod_file = getattr(val, '__file__', None)
            if not mod_file or not os.path.isfile(mod_file):
                continue
            if out not in getattr(function_tracker, '_tracked_modules', set()):
                continue
            result = self._hash_module_with_deps(out, mod_file, function_tracker)
            if result:
                return result
        return ""

    def _resolve_virtual_input_lineages(
        self, stmt_code: str, inputs: set[str], virtual_lineage: dict[str, str], virtual_modules: set[str]
    ) -> list[str]:
        """Resolve input lineage hashes for all inputs of a statement.

        Returns a list of lineage hashes for all resolved inputs (including modules),
        matching the order used by _capture_variables at runtime.
        """
        input_lineages_all = []
        sorted_inputs = sorted(inputs)

        if self.debug:
            logger.debug("[LINEAGE_DEBUG] Statement: %s...", stmt_code[:50])
            logger.debug("[LINEAGE_DEBUG] Detected inputs: %s", sorted_inputs)

        for inp in sorted_inputs:
            if inp in {'get_ipython', '__builtins__'}:
                continue

            is_module = inp in virtual_modules

            val = None
            in_user_ns = inp in self.shell.user_ns
            if in_user_ns:
                val = self.shell.user_ns[inp]

            if val is not None and not is_module:
                try:
                    if isinstance(val, types.ModuleType) or callable(val) and (inp.startswith('_') or hasattr(val, '__self__')):
                        is_module = True
                except (TypeError, AttributeError):
                    logger.debug("Type check failed for input variable %s", inp)

            lineage = self._resolve_input_lineage(inp, virtual_lineage, virtual_modules)

            if lineage:
                input_lineages_all.append(lineage)

        if self.debug:
            logger.debug("[LINEAGE_DEBUG] input_lineages_all (%s): %s", len(input_lineages_all), [ln[:12]+'...' for ln in input_lineages_all])

        return input_lineages_all

    @staticmethod
    def _stat_file_deps(hist_files: dict[str, float]) -> dict[str, float]:
        """Stat each path in *hist_files* and return ``{path: mtime}`` for existing files."""
        result: dict[str, float] = {}
        for fpath in hist_files:
            try:
                resolved = resolve_file_dep_path(fpath)
                if resolved is not None:
                    result[fpath] = os.path.getmtime(resolved)
            except OSError:
                logger.debug("Cannot stat file dependency %s", fpath)
        return result

    def _apply_cache_hit_propagation(
        self,
        stmt_code: str,
        cache_key: str,
        outputs: set[str],
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        is_import: bool,
        metadata: dict,
        hist_files: dict[str, float],
        output_lineages: dict[str, str],
    ) -> tuple[str, float, dict[str, float]]:
        """Apply a cache-hit forward propagation and return the 'hit' sentinel tuple.

        Updates *virtual_lineage* (and optionally *self.variable_lineage* for imports)
        in place.  Returns ``('hit', 0.0, stmt_file_deps)`` where the caller
        should substitute the real ``cache_lookup_time``.
        """
        if self.debug:
            print(f"[UPSTREAM] Forward propagating cached lineages for {stmt_code[:30]}...")
        for var, h in output_lineages.items():
            virtual_lineage[var] = h
        # Even on a cache hit, replay the derivation-alias bump so a mutation of
        # a base/frame (its own lineage restored from cache here) still bumps its
        # live-alias derivatives (CAS-115 / CAS-89). Same skip-inputs rule and
        # deterministic formula as the runtime and the miss path. Bumped vars are
        # threaded back so the caller can union them into ``outputs``.
        self._last_hit_bumped = bump_derived_lineages(
            self._tracking_state.derivation_edges,
            virtual_lineage,
            outputs,
            inputs,
            record=lambda t, h: virtual_lineage.__setitem__(t, h),
            present=lambda t: True,
        )
        if is_import:
            for out in outputs:
                if out in virtual_modules and out not in self.variable_lineage:
                    lineage_val = output_lineages.get(out)
                    if lineage_val:
                        self._restores.record_restore(var_name=out, lineage_hash=lineage_val)
                        if self.debug:
                            logger.debug(
                                "[LINEAGE_DEBUG] Propagated module '%s' lineage (from cache): %s...",
                                out, lineage_val[:12],
                            )
        # Mid-simulation drain: same reasoning as in _propagate_import_lineage.
        apply_collected_mutations(self._restores, self._tracking_state)
        stmt_file_deps = self._stat_file_deps(hist_files)
        return ('hit', 0.0, stmt_file_deps)

    def _collect_historical_file_deps(
        self,
        hist_files: dict[str, float],
    ) -> tuple[set[str], dict[str, float]]:
        """Collect file dependency sets when cache propagation is aborted.

        Returns ``(file_deps_to_check, stmt_file_deps)``.
        """
        file_deps_to_check: set[str] = set(hist_files.keys())
        stmt_file_deps: dict[str, float] = {}
        for fpath in hist_files:
            try:
                resolved = resolve_file_dep_path(fpath)
                if resolved is not None:
                    stmt_file_deps[fpath] = os.path.getmtime(resolved)
            except OSError:
                logger.debug("Cannot stat historical file dependency %s", fpath)
        if self.debug:
            logger.debug(
                "[UPSTREAM] Found historical file deps (validation failed/skipped): %s",
                list(hist_files.keys()),
            )
        return file_deps_to_check, stmt_file_deps

    def _try_virtual_cache_propagation(
        self,
        stmt_code: str,
        cache_key: str,
        outputs: set[str],
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        is_import: bool,
    ) -> tuple[float, bool, dict[str, float], set[str]] | None:
        """Try to forward-propagate lineages from a cached entry.

        Returns (cache_lookup_time, files_stale, stmt_file_deps, file_deps_to_check)
        on cache miss or failed validation, or None-wrapped early-return tuple isn't usedâ€”
        instead returns a special sentinel. On successful propagation, returns with
        file_deps_to_check as empty set (caller should return early).

        Actually returns:
        - On cache HIT with valid files: ('hit', cache_lookup_time, stmt_file_deps)
        - On cache miss or stale: ('miss', cache_lookup_time, files_stale, stmt_file_deps, file_deps_to_check)
        """
        cache_lookup_time = 0.0
        files_stale = False
        stmt_file_deps = {}
        file_deps_to_check = set()

        if not self.cash_instance:
            return ('miss', cache_lookup_time, files_stale, stmt_file_deps, file_deps_to_check)

        try:
            if self.debug:
                logger.debug("[UPSTREAM] Virtual lookup Key: %s", cache_key)

            t_lookup = time_module.time()
            metadata = self._get_metadata_only(cache_key)
            cache_lookup_time = time_module.time() - t_lookup

            if metadata:
                hist_files = metadata.get('file_dependencies', {})
                output_lineages = metadata.get('output_lineages', {})
                files_valid = not hist_files or self._validate_file_freshness(hist_files, self.debug)

                if files_valid and output_lineages:
                    self._last_hit_bumped = set()
                    _sentinel, _, hit_file_deps = self._apply_cache_hit_propagation(
                        stmt_code, cache_key, outputs, inputs, virtual_lineage, virtual_modules,
                        is_import, metadata, hist_files, output_lineages,
                    )
                    return ('hit', cache_lookup_time, hit_file_deps, self._last_hit_bumped)

                if not files_valid:
                    files_stale = True

                if self.debug:
                    logger.debug(
                        "[UPSTREAM] Forward prop aborted. files_valid=%s, output_lineages keys=%s",
                        files_valid, list(output_lineages.keys()) if output_lineages else 'None/Empty',
                    )

                if hist_files:
                    extra_fdeps, extra_stmt_deps = self._collect_historical_file_deps(hist_files)
                    file_deps_to_check.update(extra_fdeps)
                    stmt_file_deps.update(extra_stmt_deps)
        except (KeyError, TypeError, OSError, ValueError) as e:
            if self.debug:
                logger.debug("[UPSTREAM] Virtual lookup failed: %s", e)

        return ('miss', cache_lookup_time, files_stale, stmt_file_deps, file_deps_to_check)

    def _build_file_hash_component(self, file_deps_to_check: set[str], stmt_file_deps: dict[str, float]) -> str:
        """Build the file hash component string from file dependencies.

        Also updates stmt_file_deps with current mtimes for tracked files.
        """
        if not file_deps_to_check:
            return ""

        file_components = []
        for file_path in sorted(file_deps_to_check):
            if os.path.exists(file_path):
                try:
                    stat = os.stat(file_path)
                    file_components.append(f"{file_path}:{stat.st_mtime}:{stat.st_size}")
                    stmt_file_deps[file_path] = stat.st_mtime
                except OSError:
                    logger.debug("Cannot stat file for hash component: %s", file_path)
        if file_components:
            return ":" + hashlib.sha256(",".join(file_components).encode('utf-8')).hexdigest()
        return ""

    def _compute_virtual_output_lineage(
        self,
        source_hash: str,
        input_lineages_all: list[str],
        file_hash_component: str,
        inputs: set[str],
        outputs: set[str],
    ) -> str:
        """Compute the output lineage hash for a simulated statement.

        Includes function source hashes and module source hashes to match
        the lineage computation in statement_processor._capture_variables.
        """
        # Function source hashes for callable inputs
        func_lineage_component = ""
        function_tracker = self.function_tracker if hasattr(self, 'function_tracker') else None
        if function_tracker is not None:
            try:
                func_source_hashes = function_tracker.get_callable_source_hashes(inputs, self.shell.user_ns)
                if func_source_hashes:
                    func_parts = [f"{k}:{v}" for k, v in sorted(func_source_hashes.items())]
                    func_lineage_component = ":" + ":".join(func_parts)
            except (TypeError, ValueError, AttributeError):
                logger.debug("[UPSTREAM] Failed to compute function source hashes for capture")

        # Module source hash for module outputs
        module_lineage_component = self._compute_module_source_hash(outputs)

        lineage_str = f"{source_hash}:{':'.join(sorted(input_lineages_all))}{file_hash_component}{func_lineage_component}{module_lineage_component}"
        return hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()

    def _collect_session_file_deps(self, outputs: set[str]) -> set[str]:
        """Return file dependencies from the current session for *outputs*."""
        file_deps: set[str] = set()
        if hasattr(self, 'executed_file_deps') and self.executed_file_deps:
            for out in outputs:
                if out in self.executed_file_deps:
                    file_deps.update(self.executed_file_deps[out])
        return file_deps

    def _propagate_import_lineage(
        self,
        outputs: set[str],
        virtual_modules: set[str],
        lineage_hash: str,
    ) -> None:
        """Propagate module lineages to ``self.variable_lineage`` for import statements.

        Called after computing the lineage hash for an import so that
        ``compute_cache_key`` can find the module in ``variable_lineage`` and
        include it in the module component â€” preventing cache key mismatches.
        """
        for out in outputs:
            if out in virtual_modules and out not in self.variable_lineage:
                self._restores.record_restore(var_name=out, lineage_hash=lineage_hash)
                if self.debug:
                    logger.debug(
                        "[LINEAGE_DEBUG] Propagated module '%s' lineage to variable_lineage: %s...",
                        out, lineage_hash[:12],
                    )
        # Mid-simulation drain: subsequent statements' compute_cache_key reads
        # variable_lineage to include module components, so the write must be
        # visible before the next _update_virtual_lineage call.
        apply_collected_mutations(self._restores, self._tracking_state)

    def _update_virtual_lineage(self, stmt_code: str, virtual_lineage: dict[str, str], virtual_modules: set[str] = None, occurrence_index: int = 0) -> tuple[set[str], float, bool, dict[str, float]]:
        """
        Update virtual lineage based on statement execution.
        Returns tuple of (output variables, cache_lookup_time_seconds, files_stale, file_deps).
        files_stale is True if this statement had stale file dependencies.
        file_deps is a dict of {filepath: mtime} for file dependencies found during lookup.

        Parameters
        ----------
        occurrence_index : int
            Zero-based occurrence index for duplicate statements within a cell.
        """
        try:
            if virtual_modules is None:
                virtual_modules = set()

            # Analyze statement
            inputs, outputs = CodeAnalyzer.analyze_code_block(stmt_code)

            # Mirror the runtime: a top-level bare-Expr method call
            # (lst.append(x), bus.on(fn)) carries no Store target, so
            # analyze_code_block never surfaces the receiver as an output. Union
            # in the receivers the runtime treats as mutated (statically known,
            # or per the recorded broad-precise verdict) so the simulated lineage
            # is bumped with the SAME source-based formula -- keeping the engines
            # in sync (a runtime-only bump desyncs cross-cell restore).
            mutation_tree = self._get_cached_ast(stmt_code)
            if mutation_tree is not None:
                outputs = outputs | self._mutation_receivers(stmt_code, mutation_tree)

                # Model bare-name ``del x`` as a namespace removal so the
                # position-scoped liveness check downstream reconstructs an
                # above-the-del consumer's inputs (CAS-94). Only ``ast.Name``
                # targets remove a lineage entry; ``del d[k]`` / ``del obj.attr``
                # are container mutations handled at cacheability.py:233, so they
                # must NOT pop the base's lineage here.
                for node in mutation_tree.body:
                    if not isinstance(node, ast.Delete):
                        continue
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            virtual_lineage.pop(tgt.id, None)
                            virtual_modules.discard(tgt.id)

            stripped = stmt_code.strip()
            is_import = stripped.startswith(('import ', 'from '))
            if is_import:
                virtual_modules.update(outputs)

            if not outputs:
                return set(), 0.0, False, {}

            source_hash = hashlib.sha256(stmt_code.encode('utf-8')).hexdigest()

            # Resolve input lineages (includes ALL inputs for output lineage computation)
            input_lineages_all = self._resolve_virtual_input_lineages(
                stmt_code, inputs, virtual_lineage, virtual_modules
            )

            # Compute cache key using the unified function
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker if hasattr(self, 'function_tracker') else None,
                    virtual_lineage=virtual_lineage,
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                    debug=self.debug,
                    debug_print_fn=print,
                ),
                outputs=outputs,
                occurrence_index=occurrence_index,
            )

            # Collect file deps from current session
            file_deps_to_check = self._collect_session_file_deps(outputs)

            # Try forward-propagation from cache
            cache_result = self._try_virtual_cache_propagation(
                stmt_code, cache_key, outputs, inputs, virtual_lineage, virtual_modules, is_import
            )

            if cache_result[0] == 'hit':
                _, cache_lookup_time, stmt_file_deps, hit_bumped = cache_result
                # Union derivation-bumped vars so this cached mutation statement
                # is still recorded as a producer of the aliased base.
                outputs = outputs | hit_bumped
                return outputs, cache_lookup_time, False, stmt_file_deps

            _, cache_lookup_time, files_stale, stmt_file_deps, extra_file_deps = cache_result
            file_deps_to_check.update(extra_file_deps)

            # Build file hash component
            file_hash_component = self._build_file_hash_component(file_deps_to_check, stmt_file_deps)

            # Compute output lineage hash
            lineage_hash = self._compute_virtual_output_lineage(
                source_hash, input_lineages_all, file_hash_component, inputs, outputs
            )

            if self.debug and ('sort' in stmt_code or 'VolAdj' in stmt_code or 'read_csv' in stmt_code or 'exists' in stmt_code):
                logger.debug("[LINEAGE_CALC] Statement: %s...", stmt_code[:40])
                logger.debug("[LINEAGE_CALC]   source_hash: %s...", source_hash[:16])
                logger.debug("[LINEAGE_CALC]   sorted(input_lineages_all): %s", [h[:12]+'...' for h in sorted(input_lineages_all)])
                logger.debug("[LINEAGE_CALC]   file_hash_component: %s...", file_hash_component[:20] if file_hash_component else '(empty)')
                logger.debug("[LINEAGE_CALC]   => lineage_hash: %s...", lineage_hash[:16])

            # Update virtual state
            for out in outputs:
                virtual_lineage[out] = lineage_hash

            # Mirror the runtime derivation-alias bump (CAS-115 / CAS-89): when
            # a base/frame is mutated in place, bump its live-alias derivatives.
            # The simulator cannot observe ``.base`` / ``.obj`` identity, so it
            # only REPLAYS the runtime-recorded edge map with the SAME
            # skip-inputs rule and the SAME deterministic derived-hash formula,
            # keeping runtime and simulation byte-identical. No live namespace,
            # so every edge target counts as present. Union bumped vars into
            # ``outputs`` so the reexecution planner records THIS statement as a
            # producer of the aliased base and reschedules it on an isolated
            # re-run (the base's own cache is the stale pre-mutation value).
            bumped = bump_derived_lineages(
                self._tracking_state.derivation_edges,
                virtual_lineage,
                outputs,
                inputs,
                record=lambda t, h: virtual_lineage.__setitem__(t, h),
                present=lambda t: True,
            )
            outputs = outputs | bumped

            # CRITICAL: Propagate module lineages to self.variable_lineage immediately.
            # Without this, compute_cache_key won't find the module in variable_lineage
            # and will exclude it from module_component, causing key mismatches.
            if is_import:
                self._propagate_import_lineage(outputs, virtual_modules, lineage_hash)

            return outputs, cache_lookup_time, files_stale, stmt_file_deps

        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.error("[UPSTREAM] Error simulating statement '%s...': %s", stmt_code[:20], e)
            raise

    def _check_file_deps_for_restore(
        self, file_deps: dict[str, Any], start_time: float
    ) -> tuple[set, float, float] | None:
        """Validate file deps for a virtual restore.  Returns failure tuple or None.

        Each entry is ``{'mtime': ..., 'size': ...}`` — see
        :meth:`_validate_file_freshness`.
        """
        for fpath, stored in file_deps.items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                if self.debug:
                    print(f"[UPSTREAM] Restore failed: Miss file {fpath}")
                return set(), time_module.time() - start_time, 0.0
            stored_mtime, stored_size = split_file_dep_value(stored)
            try:
                cur_stat = os.stat(resolved)
            except OSError:
                return set(), time_module.time() - start_time, 0.0
            if abs(cur_stat.st_mtime - stored_mtime) > 0.01:
                if self.debug:
                    print(f"[UPSTREAM] Restore failed: Stale file {resolved} (delta={abs(cur_stat.st_mtime - stored_mtime):.4f}s)")
                return set(), time_module.time() - start_time, 0.0
            if stored_size is not None and cur_stat.st_size != stored_size:
                if self.debug:
                    print(f"[UPSTREAM] Restore failed: Resized file {resolved} ({stored_size} -> {cur_stat.st_size})")
                return set(), time_module.time() - start_time, 0.0
        return None  # All deps fresh

    def _check_lineage_consistency(
        self,
        metadata: dict,
        file_deps: dict[str, float],
        expected_lineages: dict[str, str] | None,
        start_time: float,
    ) -> tuple[set, float, float] | None:
        """Check output lineage consistency.  Returns failure tuple or None."""
        if not file_deps and expected_lineages and 'output_lineages' in metadata:
            for var, expected_hash in expected_lineages.items():
                cached_hash = metadata['output_lineages'].get(var)
                if cached_hash and cached_hash != expected_hash:
                    if self.debug:
                        logger.debug(
                            "[UPSTREAM] Restore failed: Lineage mismatch for %s. Exp: %s, Cached: %s",
                            var, expected_hash[:8], cached_hash[:8],
                        )
                    return set(), time_module.time() - start_time, 0.0
        return None

    def _restore_vars_from_cache(
        self,
        variables_to_restore: dict,
        metadata: dict,
    ) -> set[str]:
        """Restore variables into shell namespace.  Returns the set of restored var names."""
        restored_vars: set[str] = set()
        for var, val in variables_to_restore.items():
            if var in self.shell.user_ns:
                existing = self.shell.user_ns[var]
                try:
                    if len(existing) > 0 and len(val) == 0:
                        if self.debug:
                            logger.debug(
                                "[UPSTREAM] Restore BLOCKED for '%s': cached value is empty "
                                "but in-memory has %d items. Keeping in-memory value.",
                                var, len(existing),
                            )
                        continue
                except (TypeError, AttributeError):
                    pass
            self.shell.user_ns[var] = val
            restored_vars.add(var)
            if 'output_lineages' in metadata:
                new_lineage = metadata['output_lineages'].get(var)
                if var in self.lineage and new_lineage is not None:
                    # Buffer a value-coupled restore so apply_collected_mutations
                    # routes through lineage.record, attaching _cash_lineage_hash
                    # to the live object. Drain immediately so the attribute is
                    # visible before _update_tracking_after_restore runs.
                    self._restores.record_restore(
                        var_name=var,
                        lineage_hash=new_lineage,
                        value=val,
                    )
                    apply_collected_mutations(self._restores, self._tracking_state)
                else:
                    # Variable wasn't tracked in the lineage store before, but
                    # we still want the attribute attached so future cache-key
                    # computation finds it via the ladder fallback.
                    try:
                        val._cash_lineage_hash = new_lineage
                    except (AttributeError, TypeError):
                        logger.debug(
                            "Cannot attach _cash_lineage_hash to restored variable %s",
                            var,
                        )
        return restored_vars

    def _update_tracking_after_restore(
        self,
        restored_vars: set[str],
        metadata: dict,
        input_hashes: dict[str, str],
    ) -> None:
        """Buffer one CacheRestore per restored var.

        The orchestrator drains the collector and applies writes to
        executed_cell_codes, executed_cell_hashes, executed_input_lineages,
        executed_file_deps, and variable_lineage.
        """
        output_lineages = metadata.get('output_lineages', {}) if 'output_lineages' in metadata else {}
        stored_code = metadata.get('code', metadata.get('cell_code'))
        stored_hash = metadata.get('source_hash', metadata.get('cell_hash'))

        # Resolve file deps once.
        resolved_paths: set[str] = set()
        file_deps_meta = metadata.get('file_dependencies', {})
        if file_deps_meta:
            for stored_path in file_deps_meta:
                resolved = resolve_file_dep_path(stored_path)
                if resolved is not None:
                    resolved_paths.add(resolved)

        for var in restored_vars:
            lin = output_lineages.get(var) if output_lineages else None
            self._restores.record_restore(
                var_name=var,
                lineage_hash=lin,  # may be None — apply step skips lineage write if so
                code=stored_code if stored_code else None,
                code_hash=stored_hash if stored_hash else None,
                input_lineages=dict(input_hashes) if input_hashes else None,
                file_deps=set(resolved_paths) if resolved_paths else None,
            )

    def _eliminate_broken_vars_via_current_cell_probe(
        self,
        broken_vars: set[str],
        notebook_cells: list[str],
        current_cell_idx: int,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> None:
        """Remove variables from *broken_vars* that would be restored by current cell cache hits.

        When a broken variable (e.g. ``df``) is absent from memory but the
        current cell contains a statement that both uses it as input AND
        produces it as output (e.g. ``df['col'] = heavy_computation(df)``),
        and that statement would be a cache hit on DISK, then the cache
        restore will inject both the output variable and its data into
        memory.  In that case we do NOT need upstream re-execution to
        produce the broken variable â€” the cache restore will provide it.

        This avoids expensive upstream re-execution for scenarios like
        kernel restarts where heavy current-cell statements are on disk.
        """
        if not self.cash_instance or not broken_vars:
            return

        try:
            current_cell_code = notebook_cells[current_cell_idx].replace('\r\n', '\n')
            tree = ast.parse(current_cell_code)
        except (IndexError, SyntaxError):
            return

        # Track which broken vars are resolved by forward cache hits.
        # We simulate forward through the current cell's statements:
        # if a statement (a) would cache-hit and (b) its outputs overlap
        # with broken_vars, those outputs become available in memory.
        resolved_by_cache = set()

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.stmt):
                continue
            if is_control_structure(node):
                continue  # Control structures are too complex to probe

            try:
                stmt_code = ast.unparse(node)
            except (ValueError, TypeError):
                continue

            inputs, outputs = CodeAnalyzer.analyze_code_block(stmt_code)
            if not outputs:
                continue

            # Check if this statement uses any broken variable
            uses_broken = inputs & (broken_vars - resolved_by_cache)
            if not uses_broken:
                # Statement doesn't need any broken vars â€” skip probe
                continue

            # Build input hashes from virtual lineage (same as simulation)
            input_hashes: dict[str, str] = {}
            for inp in inputs:
                if inp in virtual_lineage:
                    input_hashes[inp] = virtual_lineage[inp]
                elif inp in self.variable_lineage:
                    input_hashes[inp] = self.variable_lineage[inp]

            # Probe the cache (read-only â€” don't restore anything yet)
            try:
                cache_key, _, _, _, _ = compute_cache_key(
                    stmt_code,
                    inputs,
                    ctx=CacheKeyContext(
                        variable_lineage=self.variable_lineage,
                        user_ns=self.shell.user_ns,
                        function_tracker=self.function_tracker if hasattr(self, 'function_tracker') else None,
                        virtual_lineage=input_hashes,
                        virtual_modules=virtual_modules,
                        compute_hash_fn=self.compute_hash_fn,
                        debug=False,
                        debug_print_fn=print,
                    ),
                    outputs=outputs,
                )

                metadata, cached_data = self.cash_instance.backend.get(cache_key)
                if metadata and cached_data is not None:
                    # Verify file deps are still valid (mtime + size, both
                    # forms â€” see _validate_file_freshness for rationale).
                    file_deps = metadata.get('file_dependencies', {})
                    deps_valid = self._validate_file_freshness(file_deps, self.debug)

                    if deps_valid:
                        # Cache hit! This statement's restore will put its
                        # outputs (including any broken vars) into memory.
                        produced = outputs & (broken_vars - resolved_by_cache)
                        if produced:
                            resolved_by_cache.update(produced)
                            # Populate variable_lineage and user_ns so the
                            # statement processor's _check_input_lineage_skip
                            # doesn't bail out before computing the cache key.
                            # The placeholder will be overwritten by the real
                            # cached value when _restore_from_cache runs.
                            for var in produced:
                                if var in virtual_lineage:
                                    self._restores.record_restore(
                                        var_name=var, lineage_hash=virtual_lineage[var],
                                    )
                                    # Drain so subsequent statements probing the
                                    # cache see the placeholder lineage.
                                    apply_collected_mutations(
                                        self._restores, self._tracking_state,
                                    )
                                if var not in self.shell.user_ns:
                                    self.shell.user_ns[var] = _FORWARD_PROBE_PLACEHOLDER
                            if self.debug:
                                logger.debug(
                                    "[UPSTREAM] Forward probe: cache hit for '%s' "
                                    "resolves broken vars: %s",
                                    stmt_code[:50], produced,
                                )

            except (KeyError, TypeError, ValueError, OSError):
                continue

        if resolved_by_cache:
            broken_vars -= resolved_by_cache
            if self.debug:
                logger.debug(
                    "[UPSTREAM] Forward probe eliminated %d broken vars: %s. "
                    "Remaining: %s",
                    len(resolved_by_cache), resolved_by_cache, broken_vars,
                )

    def _try_virtual_restore(self, stmt_code: str, outputs: set[str], inputs: set[str], input_hashes: dict[str, str], virtual_modules: set[str] | None = None, expected_lineages: dict[str, str] | None = None) -> tuple[set[str], float, float]:
        """Attempt to restore a statement using virtual input hashes.

        Directly queries backend and updates memory if successful.

        Returns:
            Tuple of (set of variables successfully restored, restore_time_seconds, saved_time_seconds).
        """
        start_time = time_module.time()

        if not self.cash_instance:
            return set(), 0.0, 0.0

        if virtual_modules is None:
            virtual_modules = set()

        try:
            # 1. Reconstruct Cache Key using the unified function.
            # Pass input_hashes as virtual_lineage so the unified function
            # can look up lineages for inputs that aren't in variable_lineage yet.
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker if hasattr(self, 'function_tracker') else None,
                    virtual_lineage=input_hashes,
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                    debug=self.debug,
                    debug_print_fn=print,
                ),
                outputs=outputs,
            )

            if self.debug:
                logger.debug("[UPSTREAM] Attempting virtual restore Key: %s", cache_key)

            # 2. Query Memory Backend first (fastest) - Or just generic backend
            metadata, cached_data = self.cash_instance.backend.get(cache_key)

            # Extract saved execution time
            saved_time = metadata.get('execution_time', 0.0) if metadata else 0.0

            if metadata and cached_data is not None:
                # 3. Check file dependencies (Critical!)
                file_deps = metadata.get('file_dependencies', {})
                fail = self._check_file_deps_for_restore(file_deps, start_time)
                if fail is not None:
                    return fail

                # 3.5 Check Lineage Consistency (Fix for Stale Cache Loops)
                # CRITICAL: We skip this strict check if file dependencies are present.
                # File hashing (mtime based) uses a strict threshold (0.01s) for detecting changes.
                # If we enforce strict lineage string equality here, we reject valid cache entries where mtime changed slightly.
                fail = self._check_lineage_consistency(metadata, file_deps, expected_lineages, start_time)
                if fail is not None:
                    return fail

                # 4. Success! Restore into shell.
                # Cache stores variables under 'variables' key (see _store_in_cache)
                variables_to_restore = cached_data.get('variables', {})
                restored_vars = self._restore_vars_from_cache(variables_to_restore, metadata)
                self._update_tracking_after_restore(restored_vars, metadata, input_hashes)
                return restored_vars, time_module.time() - start_time, saved_time

        except (KeyError, TypeError, ValueError, OSError) as e:
            if self.debug:
                logger.debug("[UPSTREAM] Virtual restore error: %s", e)

        return set(), time_module.time() - start_time, 0.0

    def _code_exists_in_notebook(self, mem_code: str, notebook_cells: list[str]) -> bool:
        """Return True if the normalized form of *mem_code* appears as a top-level
        statement in any notebook cell.

        Used when upstream modifications are detected to verify that an
        extension-validated variable's producing code has not been deleted or
        replaced.  On any parse/IO failure, returns False (conservative).
        """
        try:
            normalized_mem_code = _normalize_stmt(mem_code)
            for cell_code in notebook_cells:
                clean_cell = CodeAnalyzer.strip_magics(cell_code.replace('\r\n', '\n'))
                if not clean_cell.strip():
                    continue
                try:
                    cell_tree = self._get_cached_ast(clean_cell)
                    if cell_tree is None:
                        continue
                    for node in cell_tree.body:
                        try:
                            node_code = _normalize_stmt(ast.unparse(node))
                            if node_code == normalized_mem_code:
                                return True
                        except (ValueError, TypeError):
                            logger.debug("[UPSTREAM] Failed to unparse node during notebook code search")
                except (SyntaxError, ValueError):
                    logger.debug("[UPSTREAM] Failed to parse cell during notebook code search")
        except (OSError, SyntaxError, ValueError, ImportError):
            logger.debug("[UPSTREAM] Notebook code search failed, assuming code is gone")
        return False

    def _find_directly_mismatched_vars(
        self,
        virtual_lineage: dict[str, str],
        simulation_trace_codes: set[str],
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> set[str]:
        """Return variables whose in-memory lineage differs from virtual lineage.

        A variable is included when its producing code is on disk (traceable),
        so its mismatch represents an unsaved upstream edit rather than a
        downstream or external mutation.
        """
        directly_mismatched: set[str] = set()
        for vname in virtual_lineage:
            if vname not in self.variable_lineage:
                continue
            if virtual_lineage[vname] == self.variable_lineage[vname]:
                continue
            producing_code = self.executed_cell_codes.get(vname)
            if producing_code is None:
                continue
            normalized_prod = re.sub(r'# __iteration_context__:.*?\n', '', producing_code).strip()
            if normalized_prod in simulation_trace_codes:
                directly_mismatched.add(vname)
            elif vname in virtual_lineage:
                is_downstream = any(
                    normalized_prod in notebook_cells[di]
                    for di in range(current_cell_idx + 1, len(notebook_cells))
                )
                if not is_downstream:
                    directly_mismatched.add(vname)
        return directly_mismatched

    def _compute_tainted_vars_from_unsaved_edits(
        self,
        virtual_lineage: dict[str, str],
        simulation_trace: list,
        simulation_trace_codes: set[str],
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> set[str]:
        """Identify variables transitively tainted by an unsaved upstream edit.

        When a user executes modified code without saving, the in-memory variable
        lineage diverges from the simulation's virtual lineage.  This method:

        1. Finds variables whose memory lineage differs from the virtual lineage
           AND whose producing code is traceable to disk (directly mismatched).
        2. Propagates that taint forward through the simulation trace so that
           downstream dependents are also flagged as stale.

        The directly-mismatched variables themselves are NOT included in the
        returned set â€” they are handled by the backward scan which has full
        unsaved-edit context.  Only their transitive dependents are returned.
        """
        directly_mismatched_vars = self._find_directly_mismatched_vars(
            virtual_lineage, simulation_trace_codes, current_cell_idx, notebook_cells
        )

        if not directly_mismatched_vars:
            return set()

        # Walk simulation trace forward: taint outputs of any statement whose
        # inputs include a directly-mismatched or already-tainted variable.
        # Directly-mismatched vars themselves stay out of the taint set so the
        # backward scan handles them with proper unsaved-edit trust logic.
        vars_tainted: set[str] = set()
        propagation_sources = set(directly_mismatched_vars)
        for _stmt_code_t, outputs_t, inputs_t, _, _, _ in simulation_trace:
            if inputs_t & propagation_sources:
                new_tainted = outputs_t - directly_mismatched_vars
                vars_tainted.update(new_tainted)
                propagation_sources.update(outputs_t)

        if self.debug and vars_tainted:
            logger.debug(
                "[UPSTREAM_DEBUG] Transitive mismatch propagation (unsaved edit): "
                "root mismatches = %s, tainted dependents = %s",
                directly_mismatched_vars, vars_tainted,
            )
        return vars_tainted

    def _is_valid_extension(self, code: str, actual_lineage: str, virtual_lineage: dict[str, str], required_dependency: str | None = None) -> bool:
        """
        Check if the 'actual_lineage' is a valid derivation from the 'virtual_lineage' state.
        This verifies if executing 'code' using 'virtual_lineage' inputs produces 'actual_lineage'.
        True implies the actual state is a valid extension (e.g. unsaved downstream cell) of the notebook state.

        If required_dependency is provided, ensures that 'code' actually takes that variable as input.
        This distinguishes valid extensions (x = x + 1) from conflicting redefinitions (x = 5).
        """
        try:
             inputs, outputs = CodeAnalyzer.analyze_code_block(code)

             if required_dependency and required_dependency not in inputs:
                 return False

             input_lineages = []
             # Note: CodeAnalyzer inputs are a set. We sort for deterministic hashing order.
             # However, the lineage hash construction below sorts them anyway.
             sorted_inputs = sorted(inputs)
             for inp in sorted_inputs:
                 if inp in _BUILTIN_NAMES:
                     continue

                 if inp in virtual_lineage:
                     input_lineages.append(virtual_lineage[inp])
                 elif inp in self.variable_lineage:
                     # Fallback to memory if virtual missing (external var not in notebook)
                     input_lineages.append(self.variable_lineage[inp])
                 else:
                     # Input missing entirely. Cannot verify.
                     return False

             source_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
             # Route through the shared func-inclusive projection (matches the
             # recorder in statement/lineage.py) so an unsaved edit that calls a
             # user-defined function is not spuriously rejected for lacking the
             # function-source component. A hand-rolled sha256(code)+input_lineages
             # omitted it, so any function-routed edit always projected != recorded
             # and was wrongly discarded. [CAS-88 layer 1]
             projected_hash = self._compute_virtual_output_lineage(
                 source_hash, input_lineages, "", inputs, outputs
             )

             return projected_hash == actual_lineage

        except (KeyError, TypeError, ValueError, SyntaxError):
            return False

    @staticmethod
    def _iter_body_nodes(node: ast.AST):
        """Yield all body statements of a control structure (recursively)."""
        for attr in ('body', 'orelse', 'finalbody'):
            for child in getattr(node, attr, []) or []:
                yield child
                if is_control_structure(child):
                    yield from VirtualLineage._iter_body_nodes(child)
        # ast.Try handlers
        for handler in getattr(node, 'handlers', []) or []:
            for child in handler.body:
                yield child
                if is_control_structure(child):
                    yield from VirtualLineage._iter_body_nodes(child)

    def _recurse_control_structure_mutations(
        self, body_node: ast.AST, loop_targets: set[str]
    ) -> set[str]:
        """Recurse into a nested control structure and return its mutated vars."""
        if isinstance(body_node, ast.For):
            nested_targets = extract_target_names(body_node.target)
            return self._find_loop_mutated_vars(body_node.body, loop_targets | set(nested_targets))
        if isinstance(body_node, ast.While):
            return self._find_loop_mutated_vars(body_node.body, loop_targets)
        if isinstance(body_node, ast.If):
            result = self._find_loop_mutated_vars(body_node.body, loop_targets)
            if body_node.orelse:
                result |= self._find_loop_mutated_vars(body_node.orelse, loop_targets)
            return result
        if isinstance(body_node, ast.With):
            return self._find_loop_mutated_vars(body_node.body, loop_targets)
        if isinstance(body_node, ast.Try):
            result = self._find_loop_mutated_vars(body_node.body, loop_targets)
            for handler in body_node.handlers:
                result |= self._find_loop_mutated_vars(handler.body, loop_targets)
            if body_node.orelse:
                result |= self._find_loop_mutated_vars(body_node.orelse, loop_targets)
            if body_node.finalbody:
                result |= self._find_loop_mutated_vars(body_node.finalbody, loop_targets)
            return result
        return set()

    def _find_loop_mutated_vars(self, body_nodes: list, loop_targets: set[str]) -> set[str]:
        """
        Find variables that are *actually* mutated inside loop body.

        Uses ``MutationDetector`` for precise detection of in-place mutations
        (subscript assignment, method calls like ``.append()``, augmented
        assigns, attribute assignments).  This avoids false positives from the
        old ``inputs - outputs`` heuristic, which incorrectly marked
        read-only variables (e.g. ``df`` in ``ticker_data = df[...]``) as
        mutated.

        Excludes loop target variables and built-ins.
        """
        from ..cacheability import analyze_statement

        mutated_vars: set[str] = set()

        for body_node in body_nodes:
            if is_control_structure(body_node):
                # Recurse into nested control structures
                mutated_vars.update(
                    self._recurse_control_structure_mutations(body_node, loop_targets)
                )
            else:
                # Use analyze_statement for precise in-place mutation detection.
                # This catches: .append(), .update(), [key]=val, +=, obj.attr=val
                try:
                    stmt_code = ast.unparse(body_node)
                    detected = analyze_statement(stmt_code, None).all_mutated_vars
                    mutated_vars.update(detected)
                except (SyntaxError, ValueError, TypeError):
                    logger.debug("analyze_statement failed for AST node in loop body")

        # Filter out built-ins and loop targets
        return mutated_vars - _BUILTIN_NAMES - loop_targets
