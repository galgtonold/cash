from __future__ import annotations

"""Notebook simulator: pure-AST + cache-probing replay of upstream cells.

Extracted from ``UpstreamChecker`` so the simulation logic has a clear test
surface independent of the orchestrator. See CONTEXT.md entry: *NotebookSimulator*
and ``docs/architecture_decisions.md`` ADR-009.

The simulator never executes user code via the IPython kernel. It simulates
statement-by-statement using AST analysis and the cache backend, producing a
plan of statements to re-execute and a list of restored statements. The
orchestrator (``UpstreamChecker``) takes that plan and runs it via the real
``process_statement_callback``.
"""

import ast
import hashlib
import logging
import os
import re
import time as time_module
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..exceptions import UpstreamStateError
from ..utils import resolve_file_dep_path
from ._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from .analysis import CodeAnalyzer
from .cache_key import CacheKeyContext, compute_cache_key
from .cache_status import CacheStatus
from .control_structures import extract_target_names, get_control_structure_type, is_control_structure
from .server_discovery import get_notebook_cells

if TYPE_CHECKING:
    from .statement_processor import ProcessResult

__all__ = ["NotebookSimulator"]

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

from .simulator_types import (
    IncrementalStartResult as _IncrementalStartResult,
    SimulationCacheEntry as _SimulationCacheEntry,
    TraceEntry as _TraceEntry,
)


class NotebookSimulator:
    """Replays upstream cells via AST simulation and cache probing.

    Constructed and owned by :class:`UpstreamChecker`. Shares mutable state
    references (lineage dicts, ``executed_*`` trackers) so writes are visible
    to both. The simulator owns its own caches (``_simulation_cache``,
    ``_ast_cache``, ``_simulation_cell_hashes``, ``_cell_id_to_last_index``).
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

        # Shared state refs (same dicts as UpstreamChecker / StatementProcessor).
        self.set_tracking_state(tracking_state)

        # Simulator-owned caches.
        self._ast_cache: dict[str, ast.Module] = {}
        self._ast_cache_max_size: int = 200
        self._simulation_cache: list[_SimulationCacheEntry] = []
        self._simulation_cell_hashes: dict[int, str] = {}
        self._cell_id_to_last_index: dict[str, int] = {}

    def set_tracking_state(self, state: TrackingState) -> None:
        """Re-wire shared state refs (mirrors UpstreamChecker.set_tracking_state)."""
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes
        self.variable_lineage = state.variable_lineage
        self.lineage = state.lineage
        self.executed_file_deps = state.executed_file_deps
        self.vars_with_mutation_lineage = state.vars_with_mutation_lineage
        self.executed_input_lineages = state.executed_input_lineages

    def reset_caches(self) -> None:
        """Clear simulation and AST caches."""
        self._simulation_cache.clear()
        self._simulation_cell_hashes.clear()
        self._ast_cache.clear()

    def _get_metadata_only(self, cache_key: str) -> dict | None:
        """Get only metadata for a cache key without deserializing the full value.

        Uses backend.get_metadata() if available (FileBackend), otherwise
        falls back to backend.get() which deserializes everything.
        This optimization avoids expensive deserialization of large cached
        objects (e.g., DataFrames) when only metadata is needed.
        """
        backend = self.cash_instance.backend if self.cash_instance else None
        if backend is None:
            return None
        if hasattr(backend, 'get_metadata'):
            return backend.get_metadata(cache_key)
        # Fallback for backends without get_metadata (e.g., InMemoryBackend)
        metadata, _ = backend.get(cache_key)
        return metadata

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

        indices_to_remove: set[int] = set()
        for idx in stmts_to_run_indices:
            if self._is_reinit_to_skip(
                idx, simulation_trace, scheduled_iteration_outputs,
                vars_mutated_by_loops, iteration_context_pattern,
            ):
                indices_to_remove.add(idx)

        if indices_to_remove:
            return [idx for idx in stmts_to_run_indices if idx not in indices_to_remove]
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

    def _check_loop_var_inputs_changed(
        self,
        var_name: str,
        input_lineages_for_var: dict[str, str],
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
    ) -> bool:
        """Return True if any non-loop data input for *var_name* has changed lineage."""
        for inp_name, expected_lineage in input_lineages_for_var.items():
            if inp_name in vars_derived_from_loops:
                continue  # Skip other loop-derived vars (their lineages are inherently mismatched)
            if inp_name in loop_target_vars:
                continue  # Skip loop iteration vars (e.g., 'x' in 'for x in data')
            actual_inp_lineage = self.variable_lineage.get(inp_name)
            if actual_inp_lineage and actual_inp_lineage != expected_lineage:
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Loop input '%s' lineage changed: virtual=%s, actual=%s",
                          inp_name, expected_lineage[:8], actual_inp_lineage[:8])
                return True
        return False

    def _collect_non_module_inputs(
        self,
        stmt_inputs: list[str],
        virtual_modules: set[str],
    ) -> set[str]:
        """Return the subset of *stmt_inputs* that are data variables (not modules)."""
        data_inputs: set[str] = set()
        for inp in stmt_inputs:
            if inp in virtual_modules:
                continue
            val = self.shell.user_ns.get(inp)
            if val is not None and isinstance(val, types.ModuleType):
                continue
            data_inputs.add(inp)
        return data_inputs

    def _find_mismatched_data_inputs(
        self,
        var_name: str,
        data_inputs: set[str],
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
    ) -> set[str]:
        """Return data inputs whose virtual and actual lineages differ."""
        mismatched: set[str] = set()
        for inp in data_inputs:
            if inp == var_name:
                continue  # skip self-referential input (e.g., df['x'] = f(df))
            if inp in vars_derived_from_loops:
                continue  # loop-derived, expected mismatch
            if inp in loop_target_vars:
                continue  # loop iteration target, expected mismatch
            if inp in virtual_lineage and inp in self.variable_lineage and virtual_lineage[inp] != self.variable_lineage[inp]:
                mismatched.add(inp)
        return mismatched

    def _check_code_matches_loop_trust(
        self,
        var_name: str,
        last_stmt_for_var: str,
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
    ) -> bool:
        """Return True if var_name should be trusted in-memory when code matches trace.

        Called only when upstream_has_modifications is False and loop-derived
        check applies.  Checks that no non-loop data inputs have a lineage
        mismatch.
        """
        try:
            stmt_inputs, _ = CodeAnalyzer.analyze_code_block(last_stmt_for_var)
            data_inputs = self._collect_non_module_inputs(stmt_inputs, virtual_modules)
            mismatched_inputs = self._find_mismatched_data_inputs(
                var_name, data_inputs, vars_derived_from_loops, loop_target_vars, virtual_lineage,
            )
            if not mismatched_inputs:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG]   -> Code matches, required input '%s' mismatch due to "
                        "loop-derived inputs %s. "
                        "Trusting in-memory value (upstream unchanged).",
                        var_name, data_inputs & vars_derived_from_loops,
                    )
                return True
        except (KeyError, ValueError, TypeError):
            logger.debug("[UPSTREAM] Failed to check loop-derived inputs for '%s'", var_name)
        return False

    def _check_var_extension_valid(
        self,
        var_name: str,
        actual_lineage: str,
        virtual_lineage: dict[str, str],
        upstream_has_modifications: bool,
        notebook_cells: list[str],
    ) -> bool:
        """Return True if the in-memory value is a valid downstream extension."""
        if var_name not in self.executed_cell_codes:
            return False
        mem_code = self.executed_cell_codes[var_name]
        if not self._is_valid_extension(mem_code, actual_lineage, virtual_lineage, required_dependency=var_name):
            return False
        if upstream_has_modifications:
            code_still_in_notebook = self._code_exists_in_notebook(mem_code, notebook_cells)
            if code_still_in_notebook:
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG]   -> Valid extension (code still exists in notebook), keeping")
                return True
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Extension code no longer exists in notebook (modified/deleted upstream). Rejecting.")
            return False
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG]   -> Valid extension (no upstream modifications), keeping")
        logger.debug("[UPSTREAM] Variable '%s' is a valid extension of notebook state. Keeping.", var_name)
        return True

    def _handle_mismatch_code_matches(
        self,
        var_name: str,
        last_stmt_for_var: str | None,
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        required_inputs: set[str] | None,
        broken_vars: set[str],
    ) -> bool:
        """Handle the code-matches-but-lineage-differs case.

        Returns True if the caller should stop processing this variable (either
        the variable was trusted or marked broken), False if processing should
        continue (code did not match).
        """
        if var_name not in self.executed_cell_codes:
            return False
        last_stmt_for_var_real = last_stmt_for_var
        if last_stmt_for_var_real is None:
            return False
        sim_code = _normalize_stmt(last_stmt_for_var_real)
        mem_code = _normalize_stmt(self.executed_cell_codes[var_name])
        if sim_code != mem_code:
            return False

        # Code matches simulation but lineage differs.
        if (required_inputs and var_name in required_inputs
                and vars_derived_from_loops and not upstream_has_modifications
                and not loop_derived_trust_overridden):
            if self._check_code_matches_loop_trust(
                var_name, last_stmt_for_var_real, vars_derived_from_loops, loop_target_vars,
                virtual_lineage, virtual_modules, upstream_has_modifications, loop_derived_trust_overridden,
            ):
                return True
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Code matches but '%s' is a REQUIRED INPUT with lineage mismatch. Marking as broken.", var_name)
            logger.debug("[UPSTREAM] Variable '%s' is a required input with lineage mismatch. Must re-execute.", var_name)
            broken_vars.add(var_name)
            return True
        # For non-required variables, trust the memory
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG]   -> Lineage mismatch but Code Matches trace (%s). Assuming valid extension due to cache miss. Keeping.", var_name)
        logger.debug("[UPSTREAM] Variable '%s' mismatch but code matches trace. Keeping.", var_name)
        return True

    def _classify_one_broken_var(
        self,
        var_name: str,
        vars_derived_from_loops: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        loop_var_input_lineages: dict[str, dict[str, str]],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_with_stale_files: set[str],
        vars_mutated_by_loops: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        simulation_trace: list,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
        simulation_trace_codes: set[str] | None = None,
    ) -> None:
        """Classify a single variable and add to *broken_vars* if needed."""
        # Skip loop-derived vars when upstream is unchanged AND producing code
        # is on disk (not overridden by unsaved edit). FAST MODE can't track
        # per-iteration lineage, so we trust in-memory state.
        if var_name in vars_derived_from_loops and not upstream_has_modifications and not loop_derived_trust_overridden:
            # Don't trust if the variable was overwritten by a downstream cell.
            # Check that executed_cell_codes for this var matches an upstream statement.
            overwritten_downstream = False
            exec_code = self.executed_cell_codes.get(var_name)
            if exec_code and simulation_trace_codes is not None:
                if exec_code not in simulation_trace_codes:
                    overwritten_downstream = True
                    if self.debug:
                        logger.debug(
                            "[UPSTREAM_DEBUG] NOT trusting loop-derived '%s' â€” "
                            "executed code '%.40s' not in upstream simulation",
                            var_name, exec_code,
                        )

            if not overwritten_downstream:
                input_lineages_for_var = loop_var_input_lineages.get(var_name, {})
                inputs_changed = self._check_loop_var_inputs_changed(
                    var_name, input_lineages_for_var, vars_derived_from_loops, loop_target_vars,
                )
                if not inputs_changed:
                    if self.debug:
                        source = "directly mutated by loop" if var_name in vars_mutated_by_loops else "transitively derived from loop mutation"
                        logger.debug("[UPSTREAM_DEBUG] Skipping mismatch check for '%s' - %s, trusting in-memory state (upstream unchanged, inputs consistent)", var_name, source)
                    return
                if self.debug:
                    source = "directly mutated by loop" if var_name in vars_mutated_by_loops else "transitively derived from loop mutation"
                    logger.debug("[UPSTREAM_DEBUG] NOT trusting '%s' (%s) â€” loop input lineage changed, will check lineage", var_name, source)

        actual_lineage = self.variable_lineage[var_name]
        if var_name not in virtual_lineage:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Variable '%s' is in memory but not in virtual state (downstream or external)", var_name)
            return

        final_virtual_hash = virtual_lineage[var_name]
        if actual_lineage == final_virtual_hash:
            if var_name in vars_tainted_by_upstream_mismatch:
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Lineage matches for '%s' but tainted by upstream mismatch. Marking broken.", var_name)
                broken_vars.add(var_name)
            return

        self._handle_lineage_mismatch(
            var_name, actual_lineage, final_virtual_hash,
            vars_derived_from_loops, upstream_has_modifications, loop_derived_trust_overridden,
            loop_target_vars, virtual_lineage, virtual_modules,
            vars_with_stale_files, simulation_trace, required_inputs, current_cell_outputs,
            notebook_cells, broken_vars,
        )

    def _handle_mismatch_prereqs(
        self,
        var_name: str,
        actual_lineage: str,
        final_virtual_hash: str,
        virtual_lineage: dict[str, str],
        vars_with_stale_files: set[str],
        upstream_has_modifications: bool,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
    ) -> bool:
        """Check early-exit conditions for a lineage mismatch.

        Returns True if the caller should stop processing this variable
        (it was already handled â€” marked broken, kept, or lineage reset).
        """
        # Read-only input: reject downstream mutations (e.g., df['SMA']=...)
        if required_inputs and var_name in required_inputs and current_cell_outputs is not None and var_name not in current_cell_outputs:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> '%s' is a READ-ONLY required input "
                      "(not in current cell outputs). Rejecting downstream extension "
                      "to force restoration to upstream state.", var_name)
            broken_vars.add(var_name)
            return True

        if self._check_var_extension_valid(
            var_name, actual_lineage, virtual_lineage, upstream_has_modifications, notebook_cells,
        ):
            return True

        if var_name in vars_with_stale_files:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Variable '%s' has stale file dependencies. Forcing re-execution.", var_name)
            logger.debug("[UPSTREAM] Variable '%s' has stale file dependencies. Must re-execute.", var_name)
            broken_vars.add(var_name)
            return True

        # Downstream advancement: if var is also a current-cell output reset lineage.
        if required_inputs and var_name in required_inputs and current_cell_outputs and var_name in current_cell_outputs:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> '%s' is also an OUTPUT of the current cell. "
                      "Lineage is ahead due to downstream advancement. "
                      "Resetting lineage from %s to virtual %s.",
                      var_name, actual_lineage[:8], final_virtual_hash[:8])
            self.lineage.reset_to(var_name, final_virtual_hash)
            return True

        return False

    def _handle_lineage_mismatch(
        self,
        var_name: str,
        actual_lineage: str,
        final_virtual_hash: str,
        vars_derived_from_loops: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_with_stale_files: set[str],
        simulation_trace: list,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
    ) -> None:
        """Handle a confirmed lineage mismatch for *var_name*."""
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Lineage mismatch for '%s': virtual=%s, actual=%s", var_name, final_virtual_hash[:8], actual_lineage[:8])

        if self._handle_mismatch_prereqs(
            var_name, actual_lineage, final_virtual_hash, virtual_lineage,
            vars_with_stale_files, upstream_has_modifications, required_inputs,
            current_cell_outputs, notebook_cells, broken_vars,
        ):
            return

        last_stmt_for_var = None
        for stmt, outputs, _, _, _, _ in reversed(simulation_trace):
            if var_name in outputs:
                last_stmt_for_var = stmt
                break

        if self._handle_mismatch_code_matches(
            var_name, last_stmt_for_var, vars_derived_from_loops, loop_target_vars,
            virtual_lineage, virtual_modules, upstream_has_modifications,
            loop_derived_trust_overridden, required_inputs, broken_vars,
        ):
            return

        if required_inputs and var_name in required_inputs:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Required input mismatch and INVALID extension. Forcing strict restoration")
            logger.debug("[UPSTREAM] Variable '%s' is a required input mismatch. Forcing strict restoration.", var_name)
        logger.debug("[UPSTREAM] Variable '%s' is broken. Exp: %s, Act: %s", var_name, final_virtual_hash[:8], actual_lineage[:8])
        broken_vars.add(var_name)

    def _classify_broken_vars(
        self,
        vars_to_check: set[str],
        vars_derived_from_loops: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        loop_var_input_lineages: dict[str, dict[str, str]],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_with_stale_files: set[str],
        vars_mutated_by_loops: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        simulation_trace: list,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
        simulation_trace_codes: set[str] | None = None,
    ) -> None:
        """Classify each variable in *vars_to_check* and populate *broken_vars*.

        Also checks required inputs that are missing from memory entirely.
        """
        for var_name in vars_to_check:
            self._classify_one_broken_var(
                var_name, vars_derived_from_loops, upstream_has_modifications,
                loop_derived_trust_overridden, loop_var_input_lineages, loop_target_vars,
                virtual_lineage, virtual_modules, vars_with_stale_files, vars_mutated_by_loops,
                vars_tainted_by_upstream_mismatch, simulation_trace, required_inputs,
                current_cell_outputs, notebook_cells, broken_vars,
                simulation_trace_codes=simulation_trace_codes,
            )

        # Only required inputs matter here; temporary intermediates can stay missing.
        self._check_missing_required_inputs(
            required_inputs, virtual_lineage, virtual_modules, broken_vars,
        )

    def _run_pass2_identify_broken_vars(
        self,
        simulation_trace: list,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_mutated_by_loops: set[str],
        vars_with_stale_files: set[str],
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        upstream_has_modifications: bool,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        current_cell_idx: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Pass 2: identify broken variables.

        Returns (broken_vars, simulation_trace_codes, vars_tainted_by_upstream_mismatch).
        """
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Simulation complete. Virtual lineage keys: %s", list(virtual_lineage.keys()))
            logger.debug("[UPSTREAM_DEBUG] Actual variable_lineage keys: %s", list(self.variable_lineage.keys()))
            logger.debug("[UPSTREAM_DEBUG] Simulation trace has %s statements", len(simulation_trace))
            if vars_mutated_by_loops:
                logger.debug("[UPSTREAM_DEBUG] Variables mutated by loops (trusted): %s", vars_mutated_by_loops)

        vars_to_check: set[str] = set()
        if required_inputs:
            for var_name in required_inputs:
                if var_name in self.variable_lineage:
                    vars_to_check.add(var_name)

        simulation_trace_codes = self._build_simulation_trace_codes(simulation_trace)

        vars_tainted_by_upstream_mismatch: set[str] = set()
        if not upstream_has_modifications:
            vars_tainted_by_upstream_mismatch = self._compute_tainted_vars_from_unsaved_edits(
                virtual_lineage, simulation_trace, simulation_trace_codes,
                current_cell_idx, notebook_cells,
            )

        loop_derived_trust_overridden = self._check_loop_derived_trust_override(
            upstream_has_modifications, vars_mutated_by_loops, simulation_trace_codes,
        )

        loop_var_input_lineages = self._build_loop_var_input_lineages(
            simulation_trace, vars_derived_from_loops, virtual_lineage, virtual_modules,
        )

        broken_vars: set[str] = set()
        self._classify_broken_vars(
            vars_to_check, vars_derived_from_loops, upstream_has_modifications,
            loop_derived_trust_overridden, loop_var_input_lineages, loop_target_vars,
            virtual_lineage, virtual_modules, vars_with_stale_files, vars_mutated_by_loops,
            vars_tainted_by_upstream_mismatch, simulation_trace, required_inputs,
            current_cell_outputs, notebook_cells, broken_vars,
            simulation_trace_codes=simulation_trace_codes,
        )

        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Broken vars: %s", broken_vars)
            if not broken_vars:
                logger.debug("[UPSTREAM_DEBUG] No broken vars, nothing to re-execute")

        return broken_vars, simulation_trace_codes, vars_tainted_by_upstream_mismatch

    def _build_reexecution_plan(
        self,
        simulation_trace: list,
        broken_vars: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        simulation_trace_codes: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        vars_mutated_by_loops: set[str],
        upstream_has_modifications: bool,
        stmt_lookup_times: dict[str, float],
        notebook_cells: list[str],
    ) -> tuple[list[str], list[dict], float]:
        """Build the list of statements to re-execute and restored info.

        Returns (statements_to_reexecute, restored_statements_info, total_restore_time).
        """
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Simulation trace contents:")
            for i, (stmt, outputs, _, _, _, _) in enumerate(simulation_trace):
                logger.debug("[UPSTREAM_DEBUG]   [%s] outputs=%s: %s...", i, outputs, stmt[:60])

        loop_derived_trust_overridden = self._check_loop_derived_trust_override(
            upstream_has_modifications, vars_mutated_by_loops, simulation_trace_codes,
        )

        stmts_to_run_indices, restored_statements_info, total_restore_time = self._backward_scan_pass(
            simulation_trace, broken_vars, vars_tainted_by_upstream_mismatch,
            virtual_lineage, virtual_modules, vars_derived_from_loops,
            loop_derived_trust_overridden, upstream_has_modifications,
            simulation_trace_codes, stmt_lookup_times,
        )

        skipped_metrics = self._collect_skipped_statement_metrics(
            simulation_trace, stmts_to_run_indices, restored_statements_info,
            virtual_modules, stmt_lookup_times,
        )
        restored_statements_info.extend(skipped_metrics)

        stmts_to_run_indices = self._schedule_loop_var_contexts(stmts_to_run_indices, simulation_trace)
        stmts_to_run_indices = self._filter_accumulator_reinits(stmts_to_run_indices, simulation_trace, vars_mutated_by_loops)
        stmts_to_run_indices = self._dedup_sorted_indices(stmts_to_run_indices)

        statements_to_reexecute: list[str] = []
        for idx in stmts_to_run_indices:
            stmt_code = simulation_trace[idx][0]
            statements_to_reexecute.append(stmt_code)
            if self.debug:
                logger.debug("[UPSTREAM] Scheduled for execution: %s", stmt_code[:40])

        restored_statements_info.reverse()

        vars_updated_by_trace: set[str] = set()
        for stmt_code in statements_to_reexecute:
            try:
                _, outputs = CodeAnalyzer.analyze_code_block(stmt_code)
                vars_updated_by_trace.update(outputs)
            except (SyntaxError, ValueError):
                logger.debug("Failed to analyze statement for variable outputs: %.40s", stmt_code)

        self._reapply_unsaved_extensions(
            broken_vars, vars_updated_by_trace, simulation_trace,
            notebook_cells, statements_to_reexecute,
        )

        return statements_to_reexecute, restored_statements_info, total_restore_time

    def _dedup_sorted_indices(self, stmts_to_run_indices: list[int]) -> list[int]:
        """Return *stmts_to_run_indices* sorted and deduplicated while preserving order."""
        stmts_to_run_indices.sort()
        seen: set[int] = set()
        unique: list[int] = []
        for idx in stmts_to_run_indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        return unique

    def _is_loop_var_assignment_for_context(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        simulation_trace: list,
        scheduled_contexts: set[str],
        iteration_context_pattern: re.Pattern,
    ) -> bool:
        """Return True if *stmt* is a loop-var assignment for a scheduled iteration context."""
        for j in range(i + 1, min(i + 4, len(simulation_trace))):
            next_stmt = simulation_trace[j][0]
            match = iteration_context_pattern.search(next_stmt)
            if match and match.group(1) in scheduled_contexts:
                if len(outputs) == 1:
                    var_name = list(outputs)[0]
                    if f"{var_name} = " in stmt_code or f"{var_name}=" in stmt_code:
                        return True
            elif match:
                break
        return False

    def _schedule_loop_var_contexts(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
    ) -> list[int]:
        """Ensure loop variable assignments are scheduled alongside their iteration bodies.

        This prevents stale restorations from overwriting loop variable
        assignments when iteration bodies are re-executed.
        """
        iteration_context_pattern = re.compile(r'# __iteration_context__: ([a-f0-9]+)')
        stmts_set = set(stmts_to_run_indices)

        scheduled_contexts: set[str] = set()
        for idx in stmts_to_run_indices:
            stmt_code = simulation_trace[idx][0]
            match = iteration_context_pattern.search(stmt_code)
            if match:
                scheduled_contexts.add(match.group(1))

        if not scheduled_contexts:
            return stmts_to_run_indices

        additional_indices: list[int] = []
        for i, (stmt_code, outputs, _inputs, _, _, _) in enumerate(simulation_trace):
            if i in stmts_set or iteration_context_pattern.search(stmt_code):
                continue
            if self._is_loop_var_assignment_for_context(
                i, stmt_code, outputs, simulation_trace, scheduled_contexts, iteration_context_pattern,
            ):
                if self.debug:
                    logger.debug("[UPSTREAM] Adding loop var assignment for scheduled context: %s", stmt_code[:40])
                additional_indices.append(i)

        return stmts_to_run_indices + additional_indices

    def _check_tainted_input_valid(
        self,
        inp: str,
        virtual_lineage: dict[str, str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if *inp* is an unsaved-edit input that should be trusted.

        This is called when inp has a lineage mismatch to decide if we can
        trust the in-memory value (produced by unsaved code) rather than
        cascading into the old disk code.
        """
        if inp not in virtual_lineage or inp not in self.variable_lineage:
            return False
        if self.variable_lineage[inp] == virtual_lineage[inp]:
            return False
        if upstream_has_modifications:
            return False
        inp_producing_code = self.executed_cell_codes.get(inp)
        if inp_producing_code is None:
            return False
        normalized_inp_code = re.sub(r'# __iteration_context__:.*?\n', '', inp_producing_code).strip()
        return normalized_inp_code not in simulation_trace_codes

    def _all_tainted_inputs_valid(
        self,
        stmt_code: str,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if all inputs for a tainted statement are available and fresh."""
        stmt_inputs_check, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        for inp in stmt_inputs_check:
            if inp in virtual_modules:
                if inp in self.shell.user_ns:
                    continue
                return False
            if inp in _BUILTIN_NAMES:
                continue
            if inp not in self.shell.user_ns:
                return False
            if self._check_tainted_input_valid(inp, virtual_lineage, upstream_has_modifications, simulation_trace_codes):
                if self.debug:
                    logger.debug("[UPSTREAM] Input '%s' has different lineage but produced "
                          "by unsaved edit (code not on disk). Trusting in-memory value.", inp)
                continue
            if inp in virtual_lineage and inp in self.variable_lineage and self.variable_lineage[inp] != virtual_lineage[inp]:
                if self.debug:
                    logger.debug("[UPSTREAM] Tainted stmt input '%s' has stale lineage "
                          "(actual=%s, virtual=%s). Cascading.",
                          inp, self.variable_lineage[inp][:8], virtual_lineage[inp][:8])
                return False
        return True

    def _resolve_tainted_stmt(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        needed_outputs_pre: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        needed_vars: set[str],
        resolved_vars: set[str],
        stmts_to_run_indices: list[int],
    ) -> tuple[set[str], float, float, bool]:
        """Handle the force-reexecute branch for a tainted statement.

        Returns (restored_vars, restore_time, saved_time, handled) where
        *handled* is True when the statement was fully resolved (inputs valid),
        False when it needs cascading.
        """
        if self._all_tainted_inputs_valid(stmt_code, virtual_lineage, virtual_modules, upstream_has_modifications, simulation_trace_codes):
            stmts_to_run_indices.append(i)
            needed_vars.difference_update(outputs)
            resolved_vars.update(outputs - needed_outputs_pre)
            if self.debug:
                logger.debug("[UPSTREAM] Tainted stmt scheduled (inputs in memory): %s...", stmt_code[:60])
            return set(), 0.0, 0.0, True
        if self.debug:
            logger.debug("[UPSTREAM] Tainted stmt, inputs missing, cascading: %s...", stmt_code[:60])
        return set(), 0.0, 0.0, False

    def _check_inp_lineage_skip(
        self,
        inp: str,
        virtual_lineage: dict[str, str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if *inp* should be skipped based on lineage/unsaved-edit checks."""
        if inp not in self.variable_lineage or inp not in virtual_lineage:
            return False
        if self.variable_lineage[inp] == virtual_lineage[inp]:
            if inp in self.shell.user_ns:
                if self.debug:
                    logger.debug("[UPSTREAM] Input '%s' already valid in memory (lineage matches virtual). Skipping.", inp)
                return True
            return False
        # Lineage mismatch â€” check for unsaved edit
        if upstream_has_modifications or inp not in self.shell.user_ns:
            return False
        inp_prod_code = self.executed_cell_codes.get(inp)
        if inp_prod_code is None:
            return False
        norm_code = re.sub(r'# __iteration_context__:.*?\n', '', inp_prod_code).strip()
        if norm_code not in simulation_trace_codes:
            if self.debug:
                logger.debug("[UPSTREAM] Input '%s' lineage mismatch but produced by unsaved edit. Trusting in-memory.", inp)
            return True
        return False

    def _should_add_input_to_needed(
        self,
        inp: str,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        loop_derived_trust_overridden: bool,
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        needed_vars: set[str],
    ) -> bool:
        """Return True if *inp* should be added to *needed_vars* during cascade.

        Handles module, builtin, lineage-matching, unsaved-edit, and loop-derived
        special cases.  Side-effect: may add *inp* to *needed_vars* for modules.
        """
        if inp in virtual_modules:
            if inp not in self.shell.user_ns:
                needed_vars.add(inp)
                if self.debug:
                    logger.debug("[UPSTREAM] Module '%s' is in virtual_modules but NOT in memory. Scheduling re-import.", inp)
            return False  # handled (either added or skipped)
        if inp in _BUILTIN_NAMES:
            return False
        if self._check_inp_lineage_skip(inp, virtual_lineage, upstream_has_modifications, simulation_trace_codes):
            return False
        if inp in vars_derived_from_loops and not upstream_has_modifications and not loop_derived_trust_overridden:
            if inp in self.shell.user_ns:
                if self.debug:
                    logger.debug("[UPSTREAM] Input '%s' is loop-derived and code matches disk. Trusting in-memory.", inp)
                return False
            if self.debug:
                logger.debug("[UPSTREAM] Input '%s' is loop-derived but NOT in memory. Scheduling re-execution.", inp)
        return True

    def _cascade_failed_restore_inputs(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        needed_outputs: set[str],
        restored_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        loop_derived_trust_overridden: bool,
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        needed_vars: set[str],
        resolved_vars: set[str],
        stmts_to_run_indices: list[int],
    ) -> None:
        """Schedule *stmt* for re-execution and cascade its unresolved inputs."""
        stmts_to_run_indices.append(i)
        stmt_inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        for inp in stmt_inputs:
            if inp in resolved_vars or inp in needed_vars:
                continue
            if self._should_add_input_to_needed(
                inp, virtual_lineage, virtual_modules, vars_derived_from_loops,
                loop_derived_trust_overridden, upstream_has_modifications,
                simulation_trace_codes, needed_vars,
            ):
                needed_vars.add(inp)

        outputs_only = outputs - set(stmt_inputs)
        if outputs_only:
            removed = outputs_only & needed_vars
            if removed:
                needed_vars -= removed
                resolved_vars.update(removed)
                if self.debug:
                    logger.debug("[UPSTREAM] Scheduled stmt [%s] will produce %s. Removing from needed_vars.", i, removed)
        if self.debug:
            logger.debug("[UPSTREAM] Virtual Restore FAILED for: %s. Needed: %s, Restored: %s", stmt_code[:40], needed_outputs, restored_vars)

    def _backward_scan_pass(
        self,
        simulation_trace: list,
        broken_vars: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        loop_derived_trust_overridden: bool,
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        stmt_lookup_times: dict[str, float],
    ) -> tuple[list[int], list[dict], float]:
        """Scan the simulation trace backwards to build the re-execution schedule.

        Returns (stmts_to_run_indices, restored_statements_info, total_restore_time).
        """
        resolved_vars: set[str] = set()
        needed_vars: set[str] = set(broken_vars)
        stmts_to_run_indices: list[int] = []
        restored_statements_info: list[dict] = []

        stmt_positions: dict[str, int] = {}
        for i, (stmt_code, _, _, _, _, _) in enumerate(simulation_trace):
            stmt_positions[stmt_code] = i
        total_restore_time = 0.0

        for i in range(len(simulation_trace) - 1, -1, -1):
            stmt_code, outputs, inputs, input_hashes, produced_lineages, _ = simulation_trace[i]

            is_needed = any(out in needed_vars for out in outputs)
            if not is_needed:
                continue

            needed_outputs_pre = outputs.intersection(needed_vars)
            force_reexecute = bool(needed_outputs_pre & vars_tainted_by_upstream_mismatch)

            if force_reexecute:
                _, restore_time, saved_time, handled = self._resolve_tainted_stmt(
                    i, stmt_code, outputs, needed_outputs_pre, virtual_lineage, virtual_modules,
                    upstream_has_modifications, simulation_trace_codes, needed_vars, resolved_vars,
                    stmts_to_run_indices,
                )
                if handled:
                    continue
                restored_vars: set[str] = set()
                restore_time = 0.0
                saved_time = 0.0
            else:
                restored_vars, restore_time, saved_time = self._try_virtual_restore(
                    stmt_code, outputs, inputs, input_hashes, virtual_modules, expected_lineages=produced_lineages,
                )
            total_restore_time += restore_time

            needed_outputs = outputs.intersection(needed_vars)
            if needed_outputs and needed_outputs.issubset(restored_vars):
                if self.debug:
                    logger.debug("[UPSTREAM] Virtual Restore SUCCESS for: %s", stmt_code[:40])
                lookup_time_for_stmt = stmt_lookup_times.get(stmt_code, 0.0)
                restored_statements_info.append({
                    'code': stmt_code,
                    'restored_vars': list(restored_vars),
                    'status': CacheStatus.RESTORED,
                    'is_upstream': True,
                    'source': 'DISK',
                    'saved_time': saved_time,
                    'total_time': restore_time + lookup_time_for_stmt,
                    'position': stmt_positions.get(stmt_code, 999999),
                })
                needed_vars.difference_update(restored_vars)
                resolved_vars.update(restored_vars)
            else:
                self._cascade_failed_restore_inputs(
                    i, stmt_code, outputs, needed_outputs, restored_vars,
                    virtual_lineage, virtual_modules, vars_derived_from_loops,
                    loop_derived_trust_overridden, upstream_has_modifications,
                    simulation_trace_codes, needed_vars, resolved_vars, stmts_to_run_indices,
                )

        return stmts_to_run_indices, restored_statements_info, total_restore_time

    def _check_missing_required_inputs(
        self,
        required_inputs: set[str] | None,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        broken_vars: set[str],
    ) -> None:
        """Mark required inputs that exist in virtual lineage but are absent from memory."""
        utility_vars = {'ip', 'cash_magics', 'get_ipython', '__builtins__', 'In', 'Out'}
        for var_name in (required_inputs or []):
            if var_name not in virtual_lineage or var_name in self.variable_lineage:
                continue
            # Skip if this is a module â€” BUT only if it's actually in memory!
            # After a kernel restart, imported modules may be missing.
            if var_name in virtual_modules:
                if var_name in self.shell.user_ns:
                    if self.debug:
                        logger.debug("[UPSTREAM_DEBUG] Skipping missing module '%s' (already in memory)", var_name)
                    continue
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Module '%s' is not in memory. Marking as broken for re-import.", var_name)
                broken_vars.add(var_name)
                continue

            if var_name in utility_vars or var_name.startswith('_'):
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Skipping utility variable '%s'", var_name)
                continue

            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Required input '%s' should exist but is missing from memory. Virtual lineage: %s", var_name, virtual_lineage.get(var_name)[:8])
            logger.debug("[UPSTREAM] Variable '%s' should exist but is missing.", var_name)
            broken_vars.add(var_name)

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

    def _simulate_and_find_changes(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        current_cell_outputs: set[str] | None = None
    ) -> tuple[list[str], list[ProcessResult]]:
        """Simulate notebook execution statement-by-statement.

        Returns:
            Tuple of (list of statement codes that need re-execution,
            list of dicts with info about restored statements).
        """
        # Pass 1: Simulate ALL statements to build final virtual state
        stmt_lookup_times = {}  # stmt_code -> cache_lookup_time (disk I/O during simulation)
        loop_target_vars = set()  # Track loop iteration variables (e.g., 'item' in 'for item in data')

        (first_changed_cell, had_prior_cache, cache_had_hash_mismatch,
         simulation_trace, virtual_lineage, virtual_modules,
         new_cache_entries, vars_mutated_by_loops, vars_with_stale_files) = \
            self._find_incremental_start(current_cell_idx, notebook_cells)

        self._simulate_cells_pass1(
            first_changed_cell, current_cell_idx, notebook_cells,
            simulation_trace, virtual_lineage, virtual_modules,
            new_cache_entries, vars_mutated_by_loops, vars_with_stale_files,
            stmt_lookup_times, loop_target_vars,
        )

        # Detect whether any upstream cell was actually modified since last simulation.
        # This is True only when we had a prior simulation cache AND a cached cell's hash
        # changed (actual code modification).  NOT true when cells are simply not cached
        # yet (e.g., first time cell 3 runs, cache only has cell 1 â€” cell 2 is new to the
        # cache but wasn't modified).
        upstream_has_modifications = had_prior_cache and cache_had_hash_mismatch

        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] upstream_has_modifications=%s "
                  "(had_prior_cache=%s, cache_had_hash_mismatch=%s, first_changed_cell=%s)",
                  upstream_has_modifications, had_prior_cache, cache_had_hash_mismatch, first_changed_cell)

        vars_derived_from_loops = self._propagate_loop_derived_vars(
            vars_mutated_by_loops, simulation_trace
        )

        if self.debug and loop_target_vars:
            logger.debug("[UPSTREAM_DEBUG] Loop target variables (iteration vars): %s", loop_target_vars)

        broken_vars, simulation_trace_codes, vars_tainted = self._run_pass2_identify_broken_vars(
            simulation_trace, virtual_lineage, virtual_modules, vars_mutated_by_loops,
            vars_with_stale_files, vars_derived_from_loops, loop_target_vars,
            upstream_has_modifications, required_inputs, current_cell_outputs,
            notebook_cells, current_cell_idx,
        )

        if not broken_vars:
            return [], [], 0.0

        # Optimization: probe the current cell's statements to see if cache
        # hits would restore broken variables, making upstream re-execution
        # unnecessary.  For example, if df is broken but the current cell's
        # first df-consuming statement is a disk cache hit that restores df,
        # we don't need to re-execute upstream cells that produce df.
        self._eliminate_broken_vars_via_current_cell_probe(
            broken_vars, notebook_cells, current_cell_idx,
            virtual_lineage, virtual_modules,
        )

        if not broken_vars:
            if self.debug:
                logger.debug("[UPSTREAM] All broken vars resolved by current cell cache hits â€” skipping upstream")
            return [], [], 0.0

        return self._build_reexecution_plan(
            simulation_trace, broken_vars, vars_tainted, simulation_trace_codes,
            virtual_lineage, virtual_modules, vars_derived_from_loops,
            vars_mutated_by_loops, upstream_has_modifications,
            stmt_lookup_times, notebook_cells,
        )

    def _collect_loop_mutation_info(
        self,
        node: ast.AST,
        loop_target_vars: set[str],
        vars_mutated_by_loops: set[str],
    ) -> set[str]:
        """Collect loop mutation info and return the set of mutated vars for this node.

        Updates *loop_target_vars* (for ``ast.For``) and *vars_mutated_by_loops* in place.
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

        Tolerates both the new ``{path: {'mtime': ..., 'size': ...}}`` form and
        the legacy bare-float form (``{path: mtime}``) used by older cache
        entries.  When the size is recorded it is checked too â€” that catches
        rewrites that happen within a single mtime tick on coarse-resolution
        filesystems (HFS+/APFS, some ext4 configs).
        """
        for fpath, stored in hist_files.items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                if debug:
                    logger.debug("[UPSTREAM] Forward prop failed: Miss file %s", fpath)
                return False
            if isinstance(stored, dict):
                stored_mtime = float(stored.get('mtime', 0.0))
                stored_size = stored.get('size')
            else:
                stored_mtime = float(stored)
                stored_size = None
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
        if is_import:
            for out in outputs:
                if out in virtual_modules and out not in self.variable_lineage:
                    lineage_val = output_lineages.get(out)
                    if lineage_val:
                        self.variable_lineage[out] = lineage_val
                        if self.debug:
                            logger.debug(
                                "[LINEAGE_DEBUG] Propagated module '%s' lineage (from cache): %s...",
                                out, lineage_val[:12],
                            )
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
                    _sentinel, _, hit_file_deps = self._apply_cache_hit_propagation(
                        stmt_code, cache_key, outputs, virtual_lineage, virtual_modules,
                        is_import, metadata, hist_files, output_lineages,
                    )
                    return ('hit', cache_lookup_time, hit_file_deps)

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
                self.variable_lineage[out] = lineage_hash
                if self.debug:
                    logger.debug(
                        "[LINEAGE_DEBUG] Propagated module '%s' lineage to variable_lineage: %s...",
                        out, lineage_hash[:12],
                    )

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
                stmt_code, cache_key, outputs, virtual_lineage, virtual_modules, is_import
            )

            if cache_result[0] == 'hit':
                _, cache_lookup_time, stmt_file_deps = cache_result
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

        Tolerates both the new ``{'mtime': ..., 'size': ...}`` and the legacy
        bare-float forms â€” see :meth:`_validate_file_freshness`.
        """
        for fpath, stored in file_deps.items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                if self.debug:
                    print(f"[UPSTREAM] Restore failed: Miss file {fpath}")
                return set(), time_module.time() - start_time, 0.0
            if isinstance(stored, dict):
                stored_mtime = float(stored.get('mtime', 0.0))
                stored_size = stored.get('size')
            else:
                stored_mtime = float(stored)
                stored_size = None
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
                    # ``record(value=...)`` coordinates the dict entry and
                    # ``_cash_lineage_hash`` so they cannot drift. Previously
                    # these were two separate writes.
                    self.lineage.record(var, new_lineage, value=val)
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

    def _record_restored_cell_hash(self, var: str, stored_hash: str) -> None:
        """Add *stored_hash* to ``executed_cell_hashes[var]``, normalising legacy str values."""
        if var not in self.executed_cell_hashes:
            self.executed_cell_hashes[var] = set()
        elif isinstance(self.executed_cell_hashes[var], str):
            self.executed_cell_hashes[var] = {self.executed_cell_hashes[var]}
        self.executed_cell_hashes[var].add(stored_hash)

    def _update_tracking_after_restore(
        self,
        restored_vars: set[str],
        metadata: dict,
        input_hashes: dict[str, str],
    ) -> None:
        """Update executed_cell_codes, executed_cell_hashes, executed_input_lineages, and executed_file_deps."""
        if 'output_lineages' in metadata:
            for var, lin in metadata['output_lineages'].items():
                if var in restored_vars:
                    self.variable_lineage[var] = lin

        stored_code = metadata.get('code', metadata.get('cell_code'))
        stored_hash = metadata.get('source_hash', metadata.get('cell_hash'))
        for var in restored_vars:
            if stored_code:
                self.executed_cell_codes[var] = stored_code
            if stored_hash:
                self._record_restored_cell_hash(var, stored_hash)

        if input_hashes and restored_vars:
            for var in restored_vars:
                self.executed_input_lineages[var] = dict(input_hashes)

        # Propagate file dependencies from cache metadata so that downstream
        # statements (auto-executed by the statement processor) can inherit
        # them.  Without this, the file-dep inheritance chain breaks after a
        # kernel restart: _save_to_cache sees has_file_dependencies=False and
        # the cost model stores metadata-only entries for fast-but-large
        # intermediate transformations (e.g. sort_values on a read_csv df).
        file_deps = metadata.get('file_dependencies', {})
        if file_deps and hasattr(self, 'executed_file_deps'):
            resolved_paths: set[str] = set()
            for stored_path in file_deps:
                resolved = resolve_file_dep_path(stored_path)
                if resolved is not None:
                    resolved_paths.add(resolved)
            if resolved_paths:
                for var in restored_vars:
                    if var not in self.executed_file_deps:
                        self.executed_file_deps[var] = set()
                    self.executed_file_deps[var].update(resolved_paths)

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
                                    self.variable_lineage[var] = virtual_lineage[var]
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
                if not variables_to_restore:
                    # Legacy format (pre-v0.2): variables stored at top level
                    # instead of nested under 'variables' key.  Safe to remove
                    # once all caches created before v0.2 have expired.
                    variables_to_restore = {k: v for k, v in cached_data.items()
                                           if k not in ('stdout', 'stderr', 'outputs', 'rng_state')}

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
             inputs, _ = CodeAnalyzer.analyze_code_block(code)

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
             lineage_str = source_hash + ":" + ":".join(input_lineages)
             projected_hash = hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()

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
                    yield from UpstreamChecker._iter_body_nodes(child)
        # ast.Try handlers
        for handler in getattr(node, 'handlers', []) or []:
            for child in handler.body:
                yield child
                if is_control_structure(child):
                    yield from UpstreamChecker._iter_body_nodes(child)

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
        from .cacheability import analyze_statement

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
