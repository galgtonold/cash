from __future__ import annotations

"""Phase 1 of the notebook simulator: forward simulation + cache probing.

Extracted from ``NotebookSimulator``. Owns the simulator-internal caches
(``_simulation_cache``, ``_ast_cache``, ``_simulation_cell_hashes``,
``_cell_id_to_last_index``) and shares ``tracking_state`` dict references
with :class:`NotebookSimulator` and :class:`MismatchClassifier`. Pure-phase
invariants land in a later refactor.
"""

import inspect
import ast
import builtins
import hashlib
import logging
import os
import re
import time as time_module
import types
from collections.abc import Callable, Iterable
from typing import Any

from ...utils import resolve_file_dep_path
from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from ..analysis import CodeAnalyzer
from ..cacheability import (
    KNOWN_PURE_METHODS,
    RECEIVER_READONLY_WRITE_METHODS,
    assigned_method_call_receivers,
    called_function_global_mutations,
    fits_its_receiver,
    is_pandas_plot_call,
    top_level_call_argument_bases,
    function_arg_mutations,
    standalone_call_arg_targets,
    standalone_method_call_receivers,
    standalone_method_mutation_receivers,
)
from ..cacheability_decision import receiver_is_identity_coupled
from ..file_dep_snapshot import _LISTING_MIN_FILES, file_dep_is_fresh, stats_from_listings
from ...source_norm import source_identity_digest
from ..cache_key import (
    CacheKeyContext,
    VirtualCallable,
    called_function_dependencies,
    compute_cache_key,
    is_cash_instrumentation,
    is_module_like,
    virtual_namespace,
)
from ..cache_status import CacheStatus
from ..control_structures import extract_target_names, get_control_structure_type, is_control_structure
from ..lineage_formula import callable_source_component, module_source_component, output_lineage
from ..randomness import (
    observed_rng_reads,
    hidden_lineage_reads,
    hidden_lineage_writes,
    hidden_write_lineage,
)
from ..statement.derivation_edges import bump_derived_lineages
from ..statement.file_deps import compute_file_hash_component
from ..statement.processor import _is_control_body
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


class _InputHashes(dict):
    """A trace entry's input lineages, and those of its not-yet-defined callees' globals.

    ``input_hashes`` names the statement's own inputs, and other code copies
    it as such (a restore records it as the variable's input lineages). The
    globals a simulated-only callee reads (see ``VirtualCallable``) belong in
    the key, at the statement's position, but nowhere else -- so they ride
    alongside, read back only when the key is rebuilt from the trace.
    """
    __slots__ = ("callee_lineages",)

    def __init__(self, own: dict[str, str], callee_lineages: dict[str, str]) -> None:
        super().__init__(own)
        self.callee_lineages = callee_lineages


def _key_lineages(input_hashes: dict[str, str]) -> dict[str, str]:
    """*input_hashes* plus any callee lineages riding on it (``_InputHashes``)."""
    callee = getattr(input_hashes, "callee_lineages", None)
    return {**callee, **input_hashes} if callee else input_hashes


#: Cache keys whose file dependencies were found fresh in the current cell run
#: (see VirtualLineage._validate_file_freshness).
_FRESH_ENTRY_VERDICTS: dict = {}

#: Per cell run, per FILE: the (path, recorded snapshot) pairs found fresh, and
#: each path's resolution and mtime. The same run-long trust the entry verdicts
#: above already take, one level down: upstream entries share their files -- in
#: r23s4 every entry depended on the same 5,222 documents, in two spellings --
#: and each entry checked all of them again, twice (freshness, then mtime):
#: 7-11 s before every cell of a notebook that runs in 30 s uncached.
_FILE_STATE_THIS_RUN: dict = {}


def _file_state_this_run() -> dict | None:
    """This cell run's per-file memo, or None outside a run."""
    from .. import file_dep_snapshot as _fds
    epoch = _fds._HASH_EPOCH
    if epoch is None:
        return None
    if _FILE_STATE_THIS_RUN.get("epoch") != epoch:
        _FILE_STATE_THIS_RUN.clear()
        _FILE_STATE_THIS_RUN.update(epoch=epoch, fresh=set(), where={})
    return _FILE_STATE_THIS_RUN


def forget_file_state_this_run() -> None:
    """A statement of this cell run wrote files: answers taken before it are
    not answers for entries checked after it."""
    _FILE_STATE_THIS_RUN.clear()


def _snapshot_token(stored: Any) -> Any:
    """What identifies a recorded snapshot, cheaply."""
    if isinstance(stored, dict):
        return (stored.get("hash"), stored.get("size"), stored.get("mtime_ns", stored.get("mtime")),
                stored.get("remote"), stored.get("absent"))
    return repr(stored)


def _locate_files(paths: Iterable[str], run: dict | None) -> dict[str, tuple[str | None, Any]]:
    """``{path: (resolved or None, stat or None)}`` for *paths*, this run's answers first.

    A crowded directory is read with one listing (``stats_from_listings``); a
    listed path is where it was recorded. The rest go through
    ``resolve_file_dep_path``'s relocation fallbacks, as before.
    """
    where = run["where"] if run is not None else {}
    paths = list(paths)
    todo = [p for p in paths if p not in where]
    listed = stats_from_listings(todo) if len(todo) >= _LISTING_MIN_FILES else {}
    found: dict[str, tuple[str | None, Any]] = {}
    for p in todo:
        st = listed.get(p)
        found[p] = (p, st) if st is not None else (resolve_file_dep_path(p), None)
    if run is not None and len(where) < 200_000:
        where.update(found)
    return {p: found[p] if p in found else where[p] for p in paths}


def _stats_this_run(paths: Iterable[str]) -> dict[str, tuple[str | None, Any]]:
    """``{path: (resolved or None, stat or None)}``, one stat per path per cell run.

    The resolution is ``resolve_file_dep_path``'s (relocation fallbacks
    included); the stat is the listing's where the directory was listed.
    """
    run = _file_state_this_run()
    stats = run.setdefault("stat", {}) if run is not None else {}
    paths = list(paths)
    todo = [p for p in paths if p not in stats]
    if todo:
        for p, (resolved, listed) in _locate_files(todo, run).items():
            st = listed
            if st is None and resolved is not None:
                try:
                    st = os.stat(resolved)
                except OSError:
                    st = None
            stats[p] = (resolved, st)
    return {p: stats[p] for p in paths}


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
        # Resolved on first loop-split lookup; None means 'not yet
        # resolved', not 'no splits'. See ``_loop_split_k``.
        self._split_store = None
        self._ast_cache: dict[str, ast.Module] = {}
        self._ast_cache_max_size: int = 200
        self._simulation_cache: list[_SimulationCacheEntry] = []
        self._simulation_cell_hashes: dict[int, str] = {}
        self._cell_id_to_last_index: dict[str, int] = {}
        #: Simulated ``def``s by lineage (``VirtualCallable``). Content-
        #: addressed, so an entry never goes stale; the cap bounds memory.
        self._virtual_callables: dict[str, VirtualCallable] = {}

        # Buffered TrackingState mutations; orchestrator drains after the phase.
        self._restores = RestoreCollector()

        # Derivation-alias vars bumped during the most recent cache-hit
        # propagation; read back by _update_virtual_lineage.
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
        # Runtime-observed hidden RNG draws, keyed by statement source hash.
        # Read here for the same reason mutation_verdicts is: the simulation
        # must reproduce the runtime's key inputs EXACTLY, or the two disagree
        # and every affected statement looks changed.
        self.observed_rng_statement_draws = state.observed_rng_statement_draws

    def _observed_rng_reads(self, code: str) -> set[str]:
        """Delegates to the shared helper so all engines agree exactly."""
        return observed_rng_reads(self, code)

    @staticmethod
    def _build_function_sources(notebook_cells: list[str]) -> dict[str, str]:
        """``{function_name: source}`` for every top-level ``def`` across cells.

        Resolves from cell SOURCE (not ``inspect.getsource``, which has no
        linecache entry under nbclient) so ``function_arg_mutations`` can analyse
        a called function's body during the headless simulation. Later same-name
        defs win (last definition), matching the runtime namespace.
        """
        sources: dict[str, str] = {}
        for code in notebook_cells:
            try:
                tree = ast.parse(code.replace('\r\n', '\n'))
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    try:
                        sources[node.name] = ast.unparse(node)
                    except (ValueError, AttributeError):
                        continue
        return sources

    def _resolve_sim_function_source(self, name: str) -> str | None:
        """Source of function *name* for the headless mutation analysis.

        Cell-defined functions have no ``linecache`` entry under nbclient, so
        they come from ``_build_function_sources`` (cell text stashed by pass 1).
        A function IMPORTED from a real ``.py`` file DOES resolve via ``inspect``
        (linecache reads the file) -- cell text cannot cover it -- so without this
        fallback an imported helper that mutates its argument stays invisible to
        the simulation, desyncing it from the runtime (which already uses
        ``inspect``) and reverting the mutation on a cross-cell restore.
        """
        srcs = getattr(self, '_sim_func_sources', None)
        if srcs is not None:
            s = srcs.get(name)
            if s is not None:
                return s
        # Imported functions DO resolve via inspect (linecache reads the file).
        # Search *name* bound directly, then in the __globals__ of any imported
        # function -- so an imported ``build(ds)`` whose body calls ``clean(ds)``
        # (``clean`` living in build's module, not the notebook) resolves too.
        ns = self.shell.user_ns
        fn = ns.get(name)
        if callable(fn) and not isinstance(fn, type):
            try:
                return inspect.getsource(fn)
            except (OSError, TypeError):
                pass
        seen: set[int] = set()
        for value in ns.values():
            g = getattr(value, "__globals__", None)
            if not isinstance(g, dict) or id(g) in seen:
                continue
            seen.add(id(g))
            cand = g.get(name)
            if callable(cand) and not isinstance(cand, type):
                try:
                    return inspect.getsource(cand)
                except (OSError, TypeError):
                    continue
        return None

    def _callee_mutated_globals(self, stmt_code: str, tree: ast.Module | None) -> set[str]:
        """Simulation half of CAS-260: the globals a callee writes.

        Byte-for-byte the same derivation as
        ``StatementProcessor._callee_mutated_globals`` — same
        :func:`called_function_global_mutations` walk, same namespace filter —
        differing only in how the callee's source is found
        (:meth:`_resolve_sim_function_source` reads stashed cell text, the
        runtime reads the live object). That pairing is the one this module
        already uses for argument-mutation analysis, so it is not a new
        asymmetry.

        This has to exist. The runtime folds these names into the statement's
        key inputs; a simulation that did not would compute a DIFFERENT key for
        every statement calling a global-mutating helper, and ADR-007's whole
        point is that the two engines mint identical keys. The failure would
        not look like a crash — the simulation would simply never find the
        entry the runtime wrote, and reschedule work that was already cached.

        Control-structure bodies are excluded on the runtime's own rule (see
        :func:`~cash.notebook.statement.processor._is_control_body`): the
        simulation treats a loop as one unit, so a body statement must not
        claim the accumulator here either.
        """
        if tree is None or _is_control_body(stmt_code):
            return set()
        try:
            names = called_function_global_mutations(tree, self._resolve_sim_function_source)
        except (SyntaxError, ValueError, RecursionError):
            return set()
        ns = self.shell.user_ns
        return {
            n for n in names
            if n in ns and not isinstance(ns[n], types.ModuleType)
        }

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
        # Bare FUNCTION-arg mutations (``proc(d)`` whose body mutates its
        # parameter) mirror bare method calls: no Store target, so the mutated
        # arg must be surfaced as an output or the backward restore scan resolves
        # the var from its constructor (pre-mutation). Detected statically from
        # the called function's source, matching the runtime's inspect-based
        # detect (see ``_resolve_sim_function_source`` for cell-vs-import).
        fam: set[str] = set()
        try:
            if standalone_call_arg_targets(tree):
                fam = {
                    v for v in function_arg_mutations(tree, self._resolve_sim_function_source)
                    if not isinstance(self.shell.user_ns.get(v), types.ModuleType)
                }
        except (SyntaxError, ValueError, RecursionError):
            fam = set()
        candidates = standalone_method_call_receivers(tree)
        # captured-return draws are assignments, absent from the
        # bare-``Expr`` candidate set; keep the guard from short-circuiting them.
        assigned = assigned_method_call_receivers(tree)
        # Mirror the runtime: an Axes/Figure handed to a call is drawn on, by
        # a plain function (``forest(axes[0], df)``) too -- so it is decided
        # before the no-method-call early return, exactly as there.
        drawn_args = {name for name in top_level_call_argument_bases(tree)
                      if receiver_is_identity_coupled(self.shell.user_ns.get(name))}
        if not candidates and not assigned and not drawn_args:
            return fam
        tier1 = standalone_method_mutation_receivers(tree)
        receivers: set[str] = set()
        source_hash = hashlib.sha256(stmt_code.encode('utf-8')).hexdigest()
        verdict = self.mutation_verdicts.get(source_hash)
        for base, method in candidates:
            receiver = self.shell.user_ns.get(base)
            if isinstance(receiver, types.ModuleType):
                continue  # module function call, not a method mutation
            if base in tier1:
                receivers.add(base)
                continue
            # Mirror the runtime classifier (``_classify_method_mutations``) so the
            # simulation reproduces its decision exactly (unified-key rule).
            if method in RECEIVER_READONLY_WRITE_METHODS:
                continue  # df.to_csv reads the frame, writes a file
            if receiver_is_identity_coupled(receiver):
                receivers.add(base)  # Axes/Figure draw method mutates it
                continue
            if method in KNOWN_PURE_METHODS or is_pandas_plot_call(method, receiver):
                continue
            if verdict is not None:
                if base in verdict:
                    receivers.add(base)
            else:
                receivers.add(base)  # unknown -> conservative
        # mirror the runtime — a captured-return draw
        # (``counts, bins, _ = ax.hist(...)``) routes its receiver as a mutation
        # too, gated SOLELY by the identity-coupled check so the simulated lineage
        # bumps the same source-based receiver the runtime does (unified-key
        # rule; a runtime-only bump would desync cross-cell restore). A
        # non-coupled captured receiver (``m = df.mean()``) is never routed.
        for base, _method in assigned:
            if base in receivers:
                continue
            receiver = self.shell.user_ns.get(base)
            if isinstance(receiver, types.ModuleType):
                continue
            if receiver_is_identity_coupled(receiver) or fits_its_receiver(_method, receiver):
                receivers.add(base)
        return receivers | drawn_args | fam

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

    def record_replayed_file_deps(self, rerecorded: set[str]) -> None:
        """Add the files behind the *rerecorded* variables to the snapshots of
        the cells that produce them.

        A cell's snapshot records the files the simulation knew of. Before a
        replay -- after a restart above all -- the simulation cannot find a
        statement's cache entry (nothing is live yet to key it with), so the
        snapshot has no file dependency for a file read inside a helper; the
        replay then restores the statement, and the snapshot stays blind. A
        re-delivered file was never looked at again and the old result was
        served (round 22, r22s1). With the files the restore brought back
        (``executed_file_deps``) in the snapshot, a later change of one makes
        the next simulation redo the cell, as it always did without a restart.

        Only the file list changes. Re-simulating the replayed cells instead
        exposed lineages the simulation cannot rebuild (a view-of-view's bump
        of its root base), and the replay then rebuilt the base under a view
        that still pointed at the old one.
        """
        if not rerecorded:
            return
        for entry in self._simulation_cache:
            for trace_entry in entry.trace_segment:
                for var in set(trace_entry[1]) & rerecorded:
                    for path in self.executed_file_deps.get(var, ()):
                        if path in entry.cell_file_deps:
                            continue
                        resolved = resolve_file_dep_path(path)
                        if resolved is None:
                            continue
                        try:
                            entry.cell_file_deps[path] = os.path.getmtime(resolved)
                        except OSError:
                            continue

    def _check_cell_file_deps(
        self,
        cached_file_deps: dict[str, float],
        idx: int,
    ) -> bool:
        """Return True if any file dep for the cached cell at *idx* has changed.

        One stat per file per cell run (``_stats_this_run``)."""
        current = _stats_this_run(cached_file_deps)
        for fpath, stored_mtime in cached_file_deps.items():
            resolved, st = current[fpath]
            if resolved is None or st is None:
                return True
            if abs(st.st_mtime - stored_mtime) > 0.01:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] File dependency changed: %s "
                        "(cached mtime=%s, current=%s)",
                        resolved, stored_mtime, st.st_mtime,
                    )
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
                # A file-dep change (or a mere mtime touch of identical content)
                # must re-simulate FROM this cell so the reader re-populates
                # ``vars_with_stale_files`` (its dedicated invalidation channel,
                # mismatch_classifier ``_handle_mismatch_prereqs``). But it must
                # NOT set ``cache_had_hash_mismatch``: that flag becomes the global
                # ``upstream_has_modifications``, which disables loop-derived TRUST
                # for EVERY loop var in the notebook — so an unrelated loop chain
                # (whose runtime folds a value-hash and whose sim folds input-
                # lineages, structurally divergent) would be spuriously marked
                # broken and re-executed. Break to re-simulate; leave the flag
                # alone so file staleness stays decoupled from code modification.
                #
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

    # Control-structure wrapper prefixes. Delimited (space or colon) so a plain
    # identifier that merely begins with a keyword (``elsewhere = ...``,
    # ``exception = ...``) is NOT mistaken for a wrapper and wrongly skipped.
    _CTRL_PREFIXES = (
        'for ', 'while ', 'async for ', 'if ', 'elif ', 'else:',
        'with ', 'async with ', 'try:', 'except ', 'except:', 'finally:',
    )

    def _loop_accumulators_with_external_init(
        self,
        vars_mutated_by_loops: set[str],
        simulation_trace: list,
        loop_target_vars: set[str],
    ) -> set[str]:
        """Loop accumulators whose value ALSO derives from an external input.

        A reassignment accumulator (``result = result + 1``) enters the loop-trust
        set so that a no-change re-run is not re-executed (which would re-drain a
        one-shot iterable). But when the accumulator ALSO has an external
        dependency via a *non-loop* producing statement — typically an
        initializer like ``result = np.zeros(N)`` — editing that external input
        (``N``) and re-running the edited cell before the reader leaves
        ``upstream_has_modifications`` False, and every current-state input
        lineage is consistent, so the trust would serve a stale value. Only the
        accumulator's transitive lineage betrays the staleness. Such accumulators
        are excluded from the loop-trust set entirely, so they fall back to the
        normal lineage-mismatch path (baseline behaviour) that correctly
        re-executes; a constant-init accumulator (``total = 0``) has no external
        dependency and keeps the new trust.

        Detection scans the trace for a *non-self-referential*, *non-control*
        statement producing the accumulator (its init) that reads a data variable
        which is not a loop target, not a builtin and not a module.
        """
        tainted: set[str] = set()
        for acc in vars_mutated_by_loops:
            for entry in simulation_trace:
                stmt_code, outputs, inputs = entry[0], entry[1], entry[2]
                if acc not in outputs or acc in inputs:
                    continue  # not a producer, or self-referential (loop body)
                if stmt_code.lstrip().startswith(self._CTRL_PREFIXES):
                    continue  # loop/control wrapper; iterable feeds via the loop
                for inp in inputs:
                    if inp in loop_target_vars or inp in _BUILTIN_NAMES:
                        continue
                    val = self.shell.user_ns.get(inp)
                    if val is not None and isinstance(val, types.ModuleType):
                        continue
                    tainted.add(acc)
                    break
                if acc in tainted:
                    break
        return tainted

    def _loops_reading_changed_data(
        self,
        vars_mutated_by_loops: set[str],
        simulation_trace: list,
        loop_target_vars: set[str],
        vars_derived_from_loops: set[str],
    ) -> set[str]:
        """Loop-built variables whose loop reads data that has changed since it ran.

        Loop trust assumes the loop's inputs are what they were: with no code
        edit upstream, a loop-built value whose lineage disagrees with the
        simulation is trusted, because the two engines fold a loop differently.
        A new file in a folder the notebook globs is not a code edit. The
        frame read from it changed, the loop over it (``for k in grid:
        rows.append(score(raw, k))``) did not re-run, and everything derived
        from it -- the tuned parameter picked from ``rows`` -- was trusted and
        served from the old data (round 22, r22s4, 3/3).

        So compare each data input of a loop producing an accumulator, as the
        simulation has it at that point, with the lineage it had when the
        loop last ran (``TrackingState.control_outcomes``, recorded on entry).
        Inputs that are themselves loop-built or loop targets are skipped:
        their lineages disagree by construction.
        """
        changed: set[str] = set()
        outcomes = self._tracking_state.control_outcomes
        for entry in simulation_trace:
            stmt_code, outputs, inputs, input_hashes = entry[0], entry[1], entry[2], entry[3] or {}
            accs = outputs & vars_mutated_by_loops
            if not accs or not stmt_code.lstrip().startswith(self._CTRL_PREFIXES):
                continue
            recorded = outcomes.get(hashlib.sha256(stmt_code.encode('utf-8')).hexdigest())
            if recorded is None:
                continue
            for inp in inputs:
                if (inp in outputs or inp in loop_target_vars or inp in vars_derived_from_loops
                        or inp in _BUILTIN_NAMES):
                    continue
                if isinstance(self.shell.user_ns.get(inp), types.ModuleType):
                    continue
                now, then = input_hashes.get(inp), recorded[0].get(inp)
                if now is not None and then is not None and now != then:
                    changed |= accs
                    break
        return changed

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
                    virtual_lineage=_key_lineages(input_hashes),
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                    virtual_callables=self._virtual_callables,
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
        raw_cell: str | None = None,
    ) -> None:
        """Simulate a single AST statement node, updating all mutable state in-place.

        Returns without doing anything for control structures (they are handled
        by ``_simulate_control_structure`` directly).

        *raw_cell* is the text *node* was parsed from. The runtime keys an
        expression followed by ``;`` WITH the ``;`` (IPython's display
        suppression), which ``ast.unparse`` drops -- so ``ax.bar(...);
        ax.set_xlabel(...)`` on one line got another key and lineage here, and
        every chart drawn that way disagreed (round 21).
        """
        try:
            if is_control_structure(node):
                self._simulate_control_structure(node, virtual_lineage, virtual_modules, simulation_trace, stmt_lookup_times, vars_mutated_by_loops, loop_target_vars=loop_target_vars, vars_with_stale_files=vars_with_stale_files)
                return

            stmt_code = ast.unparse(node)
            if raw_cell is not None:
                from ..ipython.cell_executor import CellExecutor
                if CellExecutor._expr_has_trailing_semicolon(raw_cell, node):
                    stmt_code += ";"
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
        callee_lineages = self._virtual_callee_lineages(inputs, virtual_lineage, virtual_modules)
        if callee_lineages:
            input_hashes = _InputHashes(input_hashes, callee_lineages)

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
            # schedule an edited/stale writer. Empty outputs keep
            # the backward scan indifferent to the entry.
            #
            # Same for a bare CALL (``tot.plot(ax=axes[0])``): it may draw on an
            # object it was handed, which only the carrier-history pass can see
            # (``_fills_carrier``). The runtime's recorded mutation verdict
            # usually gives it an output, but that record dies with the
            # kernel, so after a restart the call vanished from the trace and
            # a figure was rebuilt without it (round 21, replay corpus).
            from ..cacheability import statement_writes_files
            if statement_writes_files(stmt_code) or (
                    isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
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
        # simulated. Without this the liveness gate would resurrect a
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
                #.
                if isinstance(node, ast.Raise):
                    break
                self._simulate_one_node(
                    i, node, cell_stmt_occurrence_counts,
                    virtual_lineage, virtual_modules, simulation_trace,
                    vars_mutated_by_loops, vars_with_stale_files,
                    stmt_lookup_times, loop_target_vars, cell_file_deps,
                    raw_cell=clean_cell_code,
                )

        except SyntaxError:
            # a single unparseable upstream cell (a half-written cell
            # the user has SAVED but not run) must NOT abort the whole
            # simulation and silently disable caching for every cell below it —
            # a notebook with one mid-edit cell is the normal state of the
            # workflow cash exists to speed up. An unparseable cell cannot have
            # executed, so it contributes no runtime state: treat it as a no-op
            # (carry the virtual state through unchanged, empty trace) by
            # falling through to the cache-entry append below, so cells that do
            # NOT depend on it keep their lineage and their cache. A cell that
            # DID depend on it then follows an ordinary lineage mismatch and
            # recomputes from the current (last-valid) memory — never a wrong
            # cache hit, because the broken cell never ran to change that
            # memory. The visible ``CashUpstreamSyntaxWarning`` naming the
            # offending cell is emitted by
            # ``UpstreamChecker._warn_broken_upstream_cells``; here we only keep
            # the simulation alive. Non-syntax errors still propagate below —
            # they signal a real bug, not a user typo. (Was: re-raise, which
            # poisoned every downstream cell silently.)
            logger.debug(
                "[UPSTREAM] Syntax error in cell %d; skipping it and continuing "
                "simulation so unrelated downstream cells keep caching.",
                i,
            )
        except (KeyError, TypeError, ValueError, OSError, AttributeError) as e:
            logger.debug("[UPSTREAM] Error simulating cell %d: %s", i, e)
            raise

        for entry in simulation_trace[trace_start:]:
            if isinstance(entry, _TraceEntry):
                entry.cell = i
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
        # Source of every top-level function across all cells, so
        # ``_mutation_receivers`` can decide which bare ``proc(d)`` calls mutate
        # their argument (headless: inspect.getsource has no linecache entry).
        self._sim_func_sources = self._build_function_sources(notebook_cells)
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
        a downstream cell reading it would serve a pre-mutation value.
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
        loop_target_vars: set[str] = None,
        vars_with_stale_files: set[str] | None = None,
    ) -> None:
        """
        Simulate execution of a control structure as a single unit.

        The entire control structure is treated as one statement for
        simulation purposes: ``stmt_code = ast.unparse(node)``, one key, one
        lineage update for everything the structure writes.

        **This does NOT always match the runtime.** The simulator models every
        loop as one unit; ``ControlStructureProcessor`` only executes one as a
        unit when ``ForLoopHandler._should_execute_loop_as_single_unit`` says
        so (roughly ``n >= 125`` for a one-statement body). Below that the
        runtime decomposes per-iteration and writes per-iteration entries,
        while this method still models the whole loop.

        That divergence is deliberate and load-bearing, not an oversight to
        "fix" by decomposing here:

        * What the simulator owes its callers is the loop's **effect on
          lineage** -- which variables it writes and what they now depend on --
          so downstream consumers invalidate correctly. Modelling the whole
          loop gets that right for both dispatch modes.
        * Reproducing per-iteration keys would mean re-deriving each
          iteration's ``__iteration_context__`` discriminator without running
          the loop, which requires the iteration VALUES the simulator does not
          have.

        The practical consequence, worth knowing before reasoning about loop
        cache keys: for a decomposed loop the key computed here corresponds to
        no entry the runtime ever writes, so it simply misses. What keeps an
        unrelated upstream edit from re-planning such a loop (CAS-262) is the
        outcome the runtime recorded for it -- see
        ``TrackingState.control_outcomes`` in ``_simulate_one_control_unit``.
        """
        if vars_mutated_by_loops is None:
            vars_mutated_by_loops = set()
        if loop_target_vars is None:
            loop_target_vars = set()

        # A loop with a recorded split verdict is modelled as TWO statements.
        #
        # This is the mechanism, not a parity nicety: the re-execution planner
        # runs the statements simulated here, so splitting this model is what
        # actually makes the runtime execute a head and a tail. Splitting only
        # in the runtime leaves the planner re-running the whole loop against
        # entries written for halves -- a silent stale value, and the cause of
        # three reverted attempts. See ``notebook/loop_split.py``.
        split_k = self._loop_split_k(node)
        if split_k is not None:
            from ..loop_split import split_nodes
            try:
                halves = split_nodes(node, split_k)
            except ValueError:          # for/else -- not splittable
                halves = ()
            if halves:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] loop split at k=%d -> simulating "
                        "head and tail separately", split_k)
                for half in halves:
                    self._simulate_one_control_unit(
                        half, virtual_lineage, virtual_modules, simulation_trace,
                        stmt_lookup_times, vars_mutated_by_loops, loop_target_vars,
                        vars_with_stale_files)
                return

        self._simulate_one_control_unit(
            node, virtual_lineage, virtual_modules, simulation_trace,
            stmt_lookup_times, vars_mutated_by_loops, loop_target_vars,
            vars_with_stale_files)

    def _loop_split_k(self, node: ast.AST) -> int | None:
        """Persisted split point for *node*, or ``None`` if it is not split.

        Best-effort: any failure to resolve the store reads as "not split",
        which is the pre-split behaviour. A simulator that cannot find the
        store must never start guessing -- a split it invents would be one
        the runtime never recorded.
        """
        if not isinstance(node, ast.For):
            return None
        try:
            from ..loop_split import is_split_half, loop_source_hash, store_for_backend
            if is_split_half(node):
                return None            # never split a half; that recurses
            if self._split_store is None:
                backend = self.cash_instance.backend if self.cash_instance else None
                self._split_store = store_for_backend(backend)
                if self._split_store is None:
                    return None
            return self._split_store.get(loop_source_hash(node))
        except Exception:  # noqa: BLE001 - never let a lookup break simulation
            logger.debug("[UPSTREAM_DEBUG] loop split lookup failed", exc_info=True)
            return None

    def _simulate_one_control_unit(
        self,
        node: ast.AST,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        simulation_trace: list[tuple],
        stmt_lookup_times: dict[str, float],
        vars_mutated_by_loops: set[str],
        loop_target_vars: set[str],
        vars_with_stale_files: set[str] | None = None,
    ) -> None:
        """Simulate ONE control structure as a single statement.

        Split out of :meth:`_simulate_control_structure` so a split loop can
        run it twice -- the second half seeing the virtual lineage the first
        produced, exactly as two source-level statements would.
        """
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

        # The runtime ran this very structure with these very inputs: what it
        # left behind is the answer, not a formula it never used (see
        # TrackingState.control_outcomes).
        recorded = self._tracking_state.control_outcomes.get(
            hashlib.sha256(stmt_code.encode('utf-8')).hexdigest())
        if recorded is not None and recorded[0] == input_hashes:
            if compute_file_hash_component(recorded[2]) == recorded[3]:
                virtual_lineage.update(recorded[1])
                all_outputs = all_outputs | set(recorded[1])
            else:
                # Same inputs, but a file behind its outputs changed: the one
                # change the entry lineages cannot show. Say so, or the loop
                # trust keeps the stale value.
                files_stale = True
                if vars_with_stale_files is not None:
                    vars_with_stale_files.update(all_outputs | set(recorded[1]))

        if self.debug:
            cs_type = get_control_structure_type(node) if node else 'unknown'
            logger.debug("[UPSTREAM_DEBUG] Simulating %s as single unit: %s... Outputs: %s", cs_type, stmt_code[:60], all_outputs)

        if all_outputs:
            produced_lineages = {out: virtual_lineage[out] for out in all_outputs if out in virtual_lineage}
            simulation_trace.append(_TraceEntry(stmt_code, all_outputs, inputs, input_hashes, produced_lineages, files_stale))
            if lookup_time > 0:
                stmt_lookup_times[stmt_code] = lookup_time
        elif self._may_write_files(node, stmt_code):
            # ``if PACK.exists(): shutil.rmtree(PACK)`` binds nothing, so it had
            # no trace entry, and a replay after a restart re-ran the cell's
            # ``PACK.mkdir()`` without it (round 23, r23s2: FileExistsError).
            # The same rule simple statements follow in _simulate_one_node.
            simulation_trace.append(_TraceEntry(stmt_code, set(), inputs, input_hashes, {}, files_stale))

    @staticmethod
    def _may_write_files(node: ast.AST, stmt_code: str) -> bool:
        """A write in the text, or a call to something that might be a
        user function that writes (the planner decides which)."""
        from ..cacheability import statement_writes_files
        if statement_writes_files(stmt_code):
            return True
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and not hasattr(builtins, n.func.id)
                   for n in ast.walk(node))

    # -- Helpers for _update_virtual_lineage ----------------------------------

    @staticmethod
    def _validate_file_freshness(
        hist_files: dict[str, Any], debug: bool = False, memo_key: str | None = None,
    ) -> bool:
        """Return True if all historical file dependencies are still fresh.

        Each entry is ``{path: {'mtime': ..., 'size': ...}}``. When ``size``
        is recorded it is checked too — that catches rewrites within a
        single mtime tick on coarse-resolution filesystems (HFS+/APFS,
        some ext4 configs).

        *memo_key* -- the entry's cache key. A "fresh" verdict holds for the
        rest of the cell run: the simulation re-validated the same upstream
        entry for every statement of the cell, twice -- 5,222 files x 2 x 24
        statements of stats in r23s4. Within one run the upstream values are
        what a from-the-top run gives even if this cell later writes one of
        their files (upstream ran before the write), so re-checking can only
        repeat the answer.
        """
        from .. import file_dep_snapshot as _fds
        epoch = _fds._HASH_EPOCH
        memo = _FRESH_ENTRY_VERDICTS
        if memo_key is not None and epoch is not None:
            if memo.get("epoch") != epoch:
                memo.clear()
                memo["epoch"] = epoch
                memo["keys"] = set()
            if memo_key in memo["keys"]:
                return True
        full_hash_max = _fds._full_hash_max_bytes() if hist_files else None
        run = _file_state_this_run()
        fresh_this_run = run["fresh"] if run is not None else set()
        pending = {(fpath, _snapshot_token(stored)): (fpath, stored) for fpath, stored in hist_files.items()}
        pending = {k: v for k, v in pending.items() if k not in fresh_this_run}
        located = _locate_files([fpath for fpath, _ in pending.values()], run)
        for token, (fpath, stored) in pending.items():
            resolved, listed = located[fpath]
            if resolved is None:
                if debug:
                    logger.debug("[UPSTREAM] Forward prop failed: Miss file %s", fpath)
                return False
            # Content-authoritative freshness when the size matches.
            is_fresh, reason = file_dep_is_fresh(resolved, stored, full_hash_max, listed)
            if not is_fresh:
                if debug:
                    logger.debug("[UPSTREAM] Forward prop failed: Stale file (%s) %s", reason, resolved)
                return False
            if run is not None:
                fresh_this_run.add(token)
        if memo_key is not None and epoch is not None:
            memo["keys"].add(memo_key)
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

        # An untracked module contributes NOTHING, matching the cache-key read
        # path and the runtime lineage writer. ``virtual_modules`` was accepted
        # here but never consulted, so the simulation fell through to
        # ``compute_hash(module)`` -> ``sha256(str(id(module)))``, a per-session
        # memory address. Runtime and simulation must agree byte for
        # byte, so this guard has to exist on both sides.
        if is_cash_instrumentation(val) or is_module_like(inp, val, virtual_modules):
            return None

        try:
            if self.compute_hash_fn:
                return self.compute_hash_fn(val)
            return hashlib.sha256(str(val).encode('utf-8')).hexdigest()
        except (TypeError, ValueError):
            logger.debug("[UPSTREAM] Failed to compute hash for input '%s'", inp)
            return None

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
        """Stat each path in *hist_files* and return ``{path: mtime}`` for existing files.

        Once per path per cell run (``_stats_this_run``), and from a directory
        listing where many share a directory."""
        return {p: st.st_mtime for p, (_resolved, st) in _stats_this_run(hist_files).items()
                if st is not None}

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
        # live-alias derivatives. Same skip-inputs rule and
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
                if out not in self.variable_lineage:
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
        stmt_file_deps = self._stat_file_deps(hist_files)
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
                files_valid = not hist_files or self._validate_file_freshness(hist_files, self.debug, memo_key=cache_key)

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
        current = _stats_this_run(file_deps_to_check)
        for file_path in sorted(file_deps_to_check):
            resolved, stat = current[file_path]
            # Only a file that is where it was recorded, as before: the
            # relocation fallbacks would put a different path's state in a key.
            if stat is not None and resolved == file_path:
                file_components.append(f"{file_path}:{stat.st_mtime}:{stat.st_size}")
                stmt_file_deps[file_path] = stat.st_mtime
        if file_components:
            return ":" + hashlib.sha256(",".join(file_components).encode('utf-8')).hexdigest()
        return ""

    #: See ``_virtual_callables``.
    _VIRTUAL_CALLABLES_MAX = 4096

    def _register_virtual_callable(
        self, stmt_code: str, tree: ast.Module | None, virtual_lineage: dict[str, str],
    ) -> None:
        """Remember a simulated ``def`` under its lineage (see ``VirtualCallable``).

        *stmt_code* is ``ast.unparse`` of the def, which is also the text the
        runtime compiles it from and ``inspect.getsource`` returns for it, so
        its digest and code are the live function's. A decorated def is left
        out: the name is bound to whatever the decorator returns, whose source
        and code are not the def's.
        """
        if tree is None or len(tree.body) != 1:
            return
        node = tree.body[0]
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.decorator_list:
            return
        lineage = virtual_lineage.get(node.name)
        if not lineage or lineage in self._virtual_callables:
            return
        try:
            module = compile(stmt_code, '<cash-simulated>', 'exec', dont_inherit=True)
        except (SyntaxError, ValueError):
            return
        code = next((c for c in module.co_consts
                     if isinstance(c, types.CodeType) and c.co_name == node.name), None)
        if code is None:
            return
        if len(self._virtual_callables) >= self._VIRTUAL_CALLABLES_MAX:
            self._virtual_callables.clear()
        self._virtual_callables[lineage] = VirtualCallable(source_identity_digest(stmt_code), code)

    def _virtual_callee_lineages(
        self, inputs: set[str], virtual_lineage: dict[str, str], virtual_modules: set[str],
    ) -> dict[str, str] | None:
        """Lineages, here, of the globals read by callees that exist only as simulated defs."""
        user_ns = self.shell.user_ns
        if not self._virtual_callables or all(name in user_ns for name in inputs):
            return None
        virtual = virtual_namespace(self._virtual_callables, virtual_lineage,
                                    self.variable_lineage, virtual_modules)
        deps = called_function_dependencies(sorted(inputs), user_ns, self.variable_lineage, virtual)
        found = dict(dep.split(':', 1) for dep in deps)
        return {name: lin for name, lin in found.items() if lin != 'ABSENT'} or None

    def absent_callee_globals(
        self, inputs: set[str], virtual_lineage: dict[str, str], virtual_modules: set[str],
    ) -> set[str]:
        """Names the callees in *inputs* read that the kernel does not hold.

        A statement re-run to rebuild a value needs them bound, and its own
        inputs do not name them: after a restart ``summary = score(raw)`` was
        re-run with ``score``'s ``OFFSET`` never rebuilt -- a NameError, where
        the cell had run fine. Modules included: the def's cell imported them.
        """
        from ..cache_key import called_function_globals
        user_ns = self.shell.user_ns
        virtual = (virtual_namespace(self._virtual_callables, virtual_lineage,
                                     self.variable_lineage, virtual_modules)
                   if self._virtual_callables else None)
        names = called_function_globals(inputs, user_ns, virtual, keep_modules=True)
        return {name for name in names if name not in user_ns}

    def _virtual_callable_hashes(self, inputs: set[str], virtual_lineage: dict[str, str]) -> dict[str, str]:
        """``name -> source digest`` for inputs that are simulated defs, not live functions."""
        if not self._virtual_callables:
            return {}
        user_ns = self.shell.user_ns
        found: dict[str, str] = {}
        for name in inputs:
            if name in user_ns:
                continue
            virtual = self._virtual_callables.get(
                virtual_lineage.get(name) or self.variable_lineage.get(name) or '')
            if virtual is not None:
                found[name] = virtual.source_hash
        return found

    def _compute_virtual_output_lineages(
        self,
        source_hash: str,
        input_lineages_all: list[str],
        file_hash_component: str,
        inputs: set[str],
        outputs: set[str],
        stmt_code: str,
        tree: ast.Module | None = None,
        virtual_lineage: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """The lineage of each output of a simulated statement.

        Built by the runtime's own formula (``lineage_formula``), one output at
        a time as the runtime does: a module-source component belongs to the
        name that came from the module. This used to be a second copy that
        gave every output one hash and knew only ``import X`` -- so every name
        from ``from helpers import clean`` disagreed with the runtime, and so
        did everything computed from it (round 21).

        A callee that is only a simulated def contributes its digest as the
        live function would (``_virtual_callable_hashes``).
        """
        function_tracker = getattr(self, 'function_tracker', None)
        user_ns = self.shell.user_ns
        try:
            func_component = callable_source_component(function_tracker, inputs, user_ns)
            virtual = self._virtual_callable_hashes(inputs, virtual_lineage or {})
            if virtual and function_tracker is not None:
                hashes = function_tracker.get_callable_source_hashes(inputs, user_ns)
                hashes.update(virtual)
                func_component = ":" + ":".join(f"{k}:{v}" for k, v in sorted(hashes.items()))
        except (TypeError, ValueError, AttributeError):
            logger.debug("[UPSTREAM] Failed to compute function source hashes for capture")
            func_component = ""
        return {
            out: output_lineage(
                source_hash, input_lineages_all, file_hash_component, func_component,
                module_source_component(function_tracker, user_ns.get(out), out, stmt_code, tree),
            )
            for out in outputs
        }

    def _collect_session_file_deps(self, outputs: set[str]) -> set[str]:
        """Return file dependencies from the current session for *outputs*."""
        file_deps: set[str] = set()
        if hasattr(self, 'executed_file_deps') and self.executed_file_deps:
            for out in outputs:
                if out in self.executed_file_deps:
                    file_deps.update(self.executed_file_deps[out])
        return file_deps

    def _bound_modules(self, outputs: set[str], tree: ast.Module | None) -> set[str]:
        """The names an import statement binds to a MODULE.

        ``import x`` always binds one. ``from m import name`` usually binds a
        function or a constant, and the runtime's key builder decides by the
        value (``is_module_like``): a function goes in as an input with its
        source hash. Counting every imported name as a module gave
        ``df = clean(raw)`` a different key in the simulation, so the
        simulation never found that statement's entry (round 21). A name not
        bound yet keeps the old answer; the runtime has no key for it either.
        """
        if tree is None:
            return set(outputs)
        from_bound = {alias.asname or alias.name
                      for node in tree.body if isinstance(node, ast.ImportFrom)
                      for alias in node.names}
        user_ns = self.shell.user_ns
        return {out for out in outputs
                if out not in from_bound or out not in user_ns
                or isinstance(user_ns[out], types.ModuleType)}

    def _propagate_import_lineage(
        self,
        outputs: set[str],
        virtual_modules: set[str],
        lineage_by_out: dict[str, str],
    ) -> None:
        """Propagate module lineages to ``self.variable_lineage`` for import statements.

        Called after computing the lineage hash for an import so that
        ``compute_cache_key`` can find the module in ``variable_lineage`` and
        include it in the module component â€” preventing cache key mismatches.
        """
        # Every name the import binds, not only modules: an import the runtime
        # SKIPPED leaves its names without a lineage otherwise, and a statement
        # reading one is then not cached ("input variable missing lineage").
        for out in outputs:
            if out not in self.variable_lineage and out in lineage_by_out:
                self._restores.record_restore(var_name=out, lineage_hash=lineage_by_out[out])
                if self.debug:
                    logger.debug(
                        "[LINEAGE_DEBUG] Propagated module '%s' lineage to variable_lineage: %s...",
                        out, lineage_by_out[out][:12],
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
            # Mirrors the runtime's `_analyze_and_hash`: a global the CALLEE
            # mutates is declared as both an input and an output here too,
            # or the two engines disagree about what the statement reads and
            # writes and ADR-007's identical-key rule breaks (CAS-265).
            inputs, outputs = CodeAnalyzer.analyze_code_block(
                stmt_code, resolve_source=self._resolve_sim_function_source,
                user_ns=self.shell.user_ns)

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
                # above-the-del consumer's inputs. Only ``ast.Name``
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
                virtual_modules.update(self._bound_modules(outputs, mutation_tree))

            # ADR-018: RNG state is a hidden lineage variable. A draw READS it
            # (fold into the key + the output-lineage inputs, so a re-seed both
            # re-keys the draw and propagates to everything cached downstream); a
            # seed PRODUCES it. Kept out of the plain ``inputs`` set that feeds the
            # trace/cacheability. Mirrors the runtime seam byte-for-byte.
            hidden_reads = hidden_lineage_reads(stmt_code) | self._observed_rng_reads(stmt_code)
            hidden_writes = hidden_lineage_writes(stmt_code)

            # A bare ``seed()`` carries no output, so it would return below before
            # recording its hidden variable. Compute its key (a seed is not a
            # draw, so no hidden read) and write the variable first.
            if hidden_writes and not outputs:
                seed_key, _, _, _, _ = compute_cache_key(
                    stmt_code,
                    inputs,
                    ctx=CacheKeyContext(
                        variable_lineage=self.variable_lineage,
                        user_ns=self.shell.user_ns,
                        function_tracker=self.function_tracker if hasattr(self, 'function_tracker') else None,
                        virtual_lineage=virtual_lineage,
                        virtual_modules=virtual_modules,
                        compute_hash_fn=self.compute_hash_fn,
                        virtual_callables=self._virtual_callables,
                    ),
                    outputs=outputs,
                    occurrence_index=occurrence_index,
                )
                for var in hidden_writes:
                    virtual_lineage[var] = hidden_write_lineage(seed_key)

            # CAS-260: the globals a CALLEE writes join ``outputs`` so the
            # simulated lineage is bumped with the same source-based formula
            # the runtime uses. The runtime ALSO skip-caches such a statement;
            # that half is runtime-only, exactly like ``mut_pre_route``.
            outputs = outputs | self._callee_mutated_globals(stmt_code, mutation_tree)

            if not outputs:
                return set(), 0.0, False, {}

            source_hash = hashlib.sha256(stmt_code.encode('utf-8')).hexdigest()

            key_lineage_inputs = inputs | hidden_reads

            # Resolve input lineages (includes ALL inputs for output lineage computation)
            input_lineages_all = self._resolve_virtual_input_lineages(
                stmt_code, key_lineage_inputs, virtual_lineage, virtual_modules
            )

            # Compute cache key using the unified function
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                key_lineage_inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker if hasattr(self, 'function_tracker') else None,
                    virtual_lineage=virtual_lineage,
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                    debug=self.debug,
                    debug_print_fn=print,
                    virtual_callables=self._virtual_callables,
                ),
                outputs=outputs,
                occurrence_index=occurrence_index,
            )

            # Combined seed+draw statement (has an output): record its hidden
            # write AFTER its key, matching the runtime's ordering.
            for var in hidden_writes:
                virtual_lineage[var] = hidden_write_lineage(cache_key)

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
                self._register_virtual_callable(stmt_code, mutation_tree, virtual_lineage)
                return outputs, cache_lookup_time, False, stmt_file_deps

            _, cache_lookup_time, files_stale, stmt_file_deps, extra_file_deps = cache_result
            file_deps_to_check.update(extra_file_deps)

            # Build file hash component
            file_hash_component = self._build_file_hash_component(file_deps_to_check, stmt_file_deps)
            own_reads = self._tracking_state.statement_file_reads.get(cache_key)
            if own_reads is not None:
                # The runtime hashed the files THIS statement read -- not the
                # ones its outputs inherited -- with compute_file_hash_component.
                # Same files, same function: an unchanged file gives the
                # runtime's lineage, a changed one a different lineage.
                file_hash_component = compute_file_hash_component(*own_reads)

            # Compute output lineage hashes
            lineage_by_out = self._compute_virtual_output_lineages(
                source_hash, input_lineages_all, file_hash_component, inputs, outputs,
                stmt_code, mutation_tree, virtual_lineage,
            )

            if self.debug and ('sort' in stmt_code or 'VolAdj' in stmt_code or 'read_csv' in stmt_code or 'exists' in stmt_code):
                logger.debug("[LINEAGE_CALC] Statement: %s...", stmt_code[:40])
                logger.debug("[LINEAGE_CALC]   source_hash: %s...", source_hash[:16])
                logger.debug("[LINEAGE_CALC]   sorted(input_lineages_all): %s", [h[:12]+'...' for h in sorted(input_lineages_all)])
                logger.debug("[LINEAGE_CALC]   file_hash_component: %s...", file_hash_component[:20] if file_hash_component else '(empty)')
                logger.debug("[LINEAGE_CALC]   => lineages: %s", {v: h[:16] for v, h in lineage_by_out.items()})

            # Update virtual state
            virtual_lineage.update(lineage_by_out)
            self._register_virtual_callable(stmt_code, mutation_tree, virtual_lineage)

            # Mirror the runtime derivation-alias bump: when
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
                self._propagate_import_lineage(outputs, virtual_modules, lineage_by_out)

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
            # Content is authoritative when the size matches.
            is_fresh, reason = file_dep_is_fresh(resolved, stored)
            if not is_fresh:
                if self.debug:
                    print(f"[UPSTREAM] Restore failed: Stale file ({reason}) {resolved}")
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

    def _lineage_confirmed_vars(
        self,
        metadata: dict,
        file_deps: dict[str, float],
        expected_lineages: dict[str, str] | None,
    ) -> frozenset[str]:
        """Vars whose cached lineage was positively matched against the expected one.

        Only these may have an EMPTY cached value restored over a non-empty
        in-memory one. A confirmed lineage means the empty value is
        the correct current result — a filter that legitimately matched nothing
        — rather than a corrupt or truncated entry.

        The guard condition mirrors ``_check_lineage_consistency`` exactly: when
        that check does not run (file deps present, or no expected lineages),
        nothing is confirmed, so the conservative empty-guard below stays in
        force. Absence of evidence is not treated as evidence.
        """
        if file_deps or not expected_lineages or 'output_lineages' not in metadata:
            return frozenset()
        cached = metadata['output_lineages']
        return frozenset(
            var
            for var, expected_hash in expected_lineages.items()
            if cached.get(var) and cached.get(var) == expected_hash
        )

    def _restore_vars_from_cache(
        self,
        variables_to_restore: dict,
        metadata: dict,
        lineage_confirmed: frozenset[str] = frozenset(),
    ) -> set[str]:
        """Restore variables into shell namespace.  Returns the set of restored var names.

        ``lineage_confirmed`` names the variables whose cached lineage matched
        the expected one; it defaults to empty so any caller that cannot
        establish that keeps the conservative behaviour.
        """
        restored_vars: set[str] = set()
        for var, val in variables_to_restore.items():
            if var in self.shell.user_ns and var not in lineage_confirmed:
                # Refuse to let an empty cached value clobber live data UNLESS
                # its lineage was confirmed above. Without that confirmation an
                # empty value is indistinguishable from a corrupt entry, and
                # overwriting 1000 rows with 0 is the more expensive mistake.
                # With it, blocking the restore is what costs correctness: the
                # statement re-executes forever and a legitimately-empty result
                # can never be served from cache.
                existing = self.shell.user_ns[var]
                try:
                    if len(existing) > 0 and len(val) == 0:
                        if self.debug:
                            logger.debug(
                                "[UPSTREAM] Restore BLOCKED for '%s': cached value is empty "
                                "but in-memory has %d items, and its lineage is unconfirmed. "
                                "Keeping in-memory value.",
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
                        val._cash_lineage_src = "statement"
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
                    deps_valid = self._validate_file_freshness(file_deps, self.debug, memo_key=cache_key)

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
                    virtual_lineage=_key_lineages(input_hashes),
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                    debug=self.debug,
                    debug_print_fn=print,
                    virtual_callables=self._virtual_callables,
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
                restored_vars = self._restore_vars_from_cache(
                    variables_to_restore,
                    metadata,
                    self._lineage_confirmed_vars(metadata, file_deps, expected_lineages),
                )
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
             # and was wrongly discarded. [layer 1]
             projected = self._compute_virtual_output_lineages(
                 source_hash, input_lineages, "", inputs, outputs, code
             )

             return actual_lineage in projected.values()

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
        from ..cacheability import analyze_statement, selfref_reassignment_targets

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
                # Self-referential reassignment accumulators (``total = total + b``,
                # ``total += b``) leave no in-place-mutation trace, so
                # all_mutated_vars misses them and the loop is wrongly re-executed,
                # re-draining one-shot iterables. Trust them like append.
                mutated_vars.update(selfref_reassignment_targets(body_node))

        # Filter out built-ins and loop targets
        return mutated_vars - _BUILTIN_NAMES - loop_targets
