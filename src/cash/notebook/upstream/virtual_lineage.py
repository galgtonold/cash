"""Phase 1 of the notebook simulator: forward simulation + cache probing.

:meth:`VirtualLineage.simulate` replays the cells above the checked one into a
:class:`SimulationResult`, starting from the first cell changed since the
previous simulation (:class:`SimulationCache`). It also restores a statement
from the cache for the later phases (``try_virtual_restore``).
"""

from __future__ import annotations

import ast
import base64
import builtins
import hashlib
import importlib.util
import logging
import marshal
import os
import re
import sys
import time as time_module
import types
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from cash.control_markers import iteration_digest, strip_markers

from ..._paths import resolve_file_dep_path
from ...analysis.ast_util import called_names, parse_cached
from ...analysis.cacheability import statement_writes_files
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source
from ...analysis.mutation_effects import (
    StatementEffects,
    classify_receivers,
    control_structure_mutations,
    live_function_source,
    statement_effects,
)
from ...analysis.namespace_effects import bare_call_argument_names, bare_call_arguments
from ...source_norm import source_identity_digest
from ...tracking import file_dep_snapshot as _fds
from ...tracking.file_dep_snapshot import LISTING_MIN_FILES, FreshnessMemo, snapshot_is_fresh, stats_from_listings
from ...tracking.randomness import (
    hidden_lineage_writes,
    hidden_write_lineage,
)
from ...value_types import BUILTIN_NAMES
from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from ..cache_key import (
    CacheKeyContext,
    VirtualCallable,
    called_function_dependencies,
    called_function_globals,
    compute_cache_key,
    control_outcome_key,
    import_bindings_key,
    mutation_verdict_key,
    statement_source_hash,
    virtual_callable_key,
    virtual_namespace,
)
from ..cache_status import CacheStatus
from ..call_refs import resolve_call_refs
from ..control_structures import extract_target_names, get_control_structure_type, is_control_structure
from ..lineage_formula import (
    callable_source_component,
    input_lineage,
    key_hidden_reads,
    lineage_hidden_reads,
    module_source_component,
    output_lineage,
    statement_environment_component,
)
from ..loop_split import is_split_half, loop_source_hash, split_nodes, store_for_backend
from ..statement import is_control_body
from ..statement.derivation_edges import bump_derived_lineages
from ..statement.file_deps import compute_file_hash_component
from ._types import (
    IncrementalStartResult,
    RestoreCollector,
    SimulationCache,
    SimulationCacheEntry,
    SimulationResult,
    TraceEntry,
    apply_collected_mutations,
)

if TYPE_CHECKING:
    from ...tracking.function_tracker import FunctionTracker

__all__ = ["VirtualLineage"]

logger = logging.getLogger(__name__)


def normalize_stmt(s: str) -> str:
    """Strip iteration-context comments and whitespace for code comparison."""
    return strip_markers(s).strip()


# Stands in the namespace for a variable the forward probe found a current-cell
# cache hit for: a statement whose input is not in the namespace is never
# looked up (``cacheability_decision._has_missing_lineage``), so without it the
# hit that restores the variable could not happen. The restore replaces it.
_FORWARD_PROBE_PLACEHOLDER = object()


class _InputHashes(dict):
    """A trace entry's input lineages, plus what only its cache key reads.

    ``input_hashes`` names the statement's own inputs, and other code copies
    it as such (a restore records it as the variable's input lineages). Two
    more things belong in the key, at the statement's position, but nowhere
    else, so they ride alongside and are read back only when the key is
    rebuilt from the trace: the globals a simulated-only callee reads (see
    ``VirtualCallable``), and the hidden RNG variables the statement reads
    (``lineage_formula.key_hidden_reads``).
    """

    __slots__ = ("callee_lineages", "hidden_lineages")

    def __init__(
        self,
        own: dict[str, str],
        callee_lineages: dict[str, str] | None = None,
        hidden_lineages: dict[str, str | None] | None = None,
    ) -> None:
        super().__init__(own)
        self.callee_lineages = callee_lineages or {}
        self.hidden_lineages = hidden_lineages or {}


def key_lineages(input_hashes: dict[str, str]) -> dict[str, str]:
    """*input_hashes* plus the key-only lineages riding on it (``_InputHashes``)."""
    callee = getattr(input_hashes, "callee_lineages", None) or {}
    hidden = {k: v for k, v in (getattr(input_hashes, "hidden_lineages", None) or {}).items() if v is not None}
    return {**callee, **hidden, **input_hashes} if callee or hidden else input_hashes


def key_inputs(inputs: set[str], input_hashes: dict[str, str]) -> set[str]:
    """The names a trace entry's cache key reads: its inputs plus the hidden
    variables riding on *input_hashes*, as the runtime keys it."""
    hidden = getattr(input_hashes, "hidden_lineages", None)
    return set(inputs) | set(hidden) if hidden else set(inputs)


#: Cache keys whose file dependencies were found fresh in the current cell run
#: (see VirtualLineage._validate_file_freshness).
_FRESH_ENTRY_VERDICTS: dict = {}

#: Per cell run, per FILE: the freshness answer for each (path, recorded
#: snapshot) pair, and each path's resolution and mtime. The same run-long trust the entry verdicts
#: above already take, one level down: upstream entries share their files -- in
#: one notebook every entry depended on the same 5,222 documents, in two spellings --
#: and each entry checked all of them again, twice (freshness, then mtime):
#: 7-11 s before every cell of a notebook that runs in 30 s uncached.
_FILE_STATE_THIS_RUN: dict = {}


def _file_state_this_run() -> dict | None:
    """This cell run's per-file memo, or None outside a run."""

    epoch = _fds.HASH_EPOCH
    if epoch is None:
        return None
    if _FILE_STATE_THIS_RUN.get("epoch") != epoch:
        _FILE_STATE_THIS_RUN.clear()
        _FILE_STATE_THIS_RUN.update(epoch=epoch, memo=FreshnessMemo(), where={})
    return _FILE_STATE_THIS_RUN


def forget_file_state_this_run() -> None:
    """A statement of this cell run wrote files: answers taken before it are
    not answers for entries checked after it."""
    _FILE_STATE_THIS_RUN.clear()


def _locate_files(paths: Iterable[str], run: dict | None) -> dict[str, tuple[str | None, Any]]:
    """``{path: (resolved or None, stat or None)}`` for *paths*, this run's answers first.

    A crowded directory is read with one listing (``stats_from_listings``); a
    listed path is where it was recorded. The rest go through
    ``resolve_file_dep_path``'s relocation fallbacks, as before.
    """
    where = run["where"] if run is not None else {}
    paths = list(paths)
    todo = [p for p in paths if p not in where]
    listed = stats_from_listings(todo) if len(todo) >= LISTING_MIN_FILES else {}
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


def loop_derived_vars(vars_mutated_by_loops: set[str], simulation_trace: list[TraceEntry]) -> set[str]:
    """*vars_mutated_by_loops* plus every variable built from one, walking the
    trace forward. These are trusted in memory rather than replaced with stale
    cached values."""
    if not vars_mutated_by_loops:
        return set()
    vars_derived = set(vars_mutated_by_loops)
    for entry in simulation_trace:
        if entry.inputs & vars_derived:
            vars_derived.update(entry.outputs)
    return vars_derived


def lineage_conflict(
    metadata: dict[str, Any], file_deps: dict[str, Any], expected_lineages: dict[str, str] | None
) -> str | None:
    """An output whose cached lineage is not the one the simulation expects,
    or None.

    Not compared for an entry with file dependencies: its lineages fold the
    files' state, which the freshness check has already judged.
    """
    if file_deps or not expected_lineages or "output_lineages" not in metadata:
        return None
    cached = metadata["output_lineages"]
    for var, expected in expected_lineages.items():
        if cached.get(var) and cached[var] != expected:
            return var
    return None


def lineage_confirmed_vars(
    metadata: dict[str, Any], file_deps: dict[str, Any], expected_lineages: dict[str, str] | None
) -> frozenset[str]:
    """Outputs whose cached lineage was positively matched against the expected one.

    Only these may have an EMPTY cached value restored over a non-empty live
    one: a confirmed lineage makes the empty value the correct result (a
    filter that legitimately matched nothing), not a corrupt entry. Where
    :func:`lineage_conflict` compares nothing, nothing is confirmed.
    """
    if file_deps or not expected_lineages or "output_lineages" not in metadata:
        return frozenset()
    cached = metadata["output_lineages"]
    return frozenset(var for var, expected in expected_lineages.items() if cached.get(var) and cached[var] == expected)


class VirtualLineage:
    """Phase 1 of NotebookSimulator: forward simulation + cache probing.

    Writes to ``TrackingState`` are buffered in ``restores``.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol | None,
        tracking_state: TrackingState,
        compute_hash_fn: Callable[[Any], str] | None = None,
        function_tracker: FunctionTracker | None = None,
        cache: SimulationCache | None = None,
    ) -> None:
        self.shell = shell
        self.cash_instance = cash_instance
        self.compute_hash_fn = compute_hash_fn
        #: The runtime's tracker, so the simulation hashes called functions
        #: and modules exactly as the statement processor does.
        self.function_tracker = function_tracker

        #: The checker's. ``mutation_verdicts`` and
        #: ``observed_rng_statement_draws`` are read from it because the
        #: simulation must reproduce the runtime's key inputs exactly.
        self.tracking_state = tracking_state

        # Simulator-owned caches.
        # Resolved on first loop-split lookup; None means 'not yet
        # resolved', not 'no splits'. See ``_loop_split_k``.
        self._split_store = None
        self.cache = cache if cache is not None else SimulationCache()
        #: Simulated ``def``s by lineage (``VirtualCallable``). Content-
        #: addressed, so an entry never goes stale; the cap bounds memory.
        self._virtual_callables: dict[str, VirtualCallable] = {}
        #: Classes a simulated ``from X import C`` bound, by lineage -> the
        #: source digest the runtime folds into a lineage (see
        #: ``_register_imported_callables``). Not in the key: it skips classes.
        self._imported_classes: dict[str, str] = {}
        #: The lineage ``_propagate_import_lineage`` last gave each name, so a
        #: later import of that name can replace it -- but not one the runtime set.
        self.propagated_imports: dict[str, str] = {}

        # Buffered TrackingState mutations; orchestrator drains after the phase.
        self.restores = RestoreCollector()

        # Derivation-alias vars bumped during the most recent cache-hit
        # propagation; read back by _update_virtual_lineage.
        self._last_hit_bumped: set[str] = set()
        #: Names the forward probe bound to ``_FORWARD_PROBE_PLACEHOLDER``.
        self._probe_placeholders: set[str] = set()
        #: ``{name: source}`` of every top-level def in the notebook's cells,
        #: set by each pass 1 (see ``_resolve_sim_function_source``).
        self._sim_func_sources: dict[str, str] = {}
        #: ``TrackingState.module_generation`` the last pass 1 saw.
        self._simulated_module_generation = 0
        #: ``_import_bindings`` answers by statement; cleared with the caches.
        self._import_bindings_memo: dict[str, dict[str, dict]] = {}

    @staticmethod
    def _build_function_sources(notebook_cells: list[str]) -> dict[str, str]:
        """``{function_name: source}`` for every top-level ``def`` across cells.

        Resolves from cell SOURCE (not ``inspect.getsource``, which has no
        linecache entry under nbclient) so ``function_arg_mutations`` can analyse
        a called function's body during the headless simulation. Later same-name
        defs win (last definition), matching the runtime namespace. A cell's
        magics are stripped first, as the simulation reads every cell.
        """
        sources: dict[str, str] = {}
        for code in notebook_cells:
            tree = parse_cell_source(code)
            if tree is None:
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
        they come from the cell text stashed by pass 1. A function imported
        from a ``.py`` file is not in the cell text; it resolves as the runtime
        resolves it, so an imported helper that mutates its argument is seen by
        both engines.
        """
        source = self._sim_func_sources.get(name)
        if source is not None:
            return source
        return live_function_source(name, self.shell.user_ns)

    def _mutation_receivers(
        self,
        stmt_code: str,
        tree: ast.Module,
        virtual_modules: set[str] | None = None,
    ) -> set[str]:
        """Names *tree*'s calls change in place, decided as the runtime decides
        them (``classify_receivers``).

        The runtime's verdict for the statement is read from this session, else
        from the backend (an earlier kernel). *virtual_modules* names modules
        the simulation bound but the kernel does not hold yet: after a restart
        ``pd.set_option(...)`` must still read as a module call, not as an
        unknown method that bumps ``pd`` for every reader.
        """
        user_ns = self.shell.user_ns

        def load_verdict() -> set[str] | None:
            source_hash = statement_source_hash(stmt_code)
            verdict = self.tracking_state.mutation_verdicts.get(source_hash)
            return verdict if verdict is not None else self._persisted_mutation_verdict(source_hash)

        # Bare-call arguments: the live ones the runtime watches, and, after a
        # restart, the ones not live yet, whose recorded verdict is all there
        # is to go on (`heapq.heapify(xs)` must replay before `xs[0]`).
        arguments = bare_call_arguments(tree, user_ns) | {n for n in bare_call_argument_names(tree) if n not in user_ns}
        classes = classify_receivers(
            tree, user_ns, load_verdict, arguments=arguments, virtual_modules=virtual_modules or ()
        )
        # The simulation cannot watch the statement run: an undecided
        # receiver is assumed to change, an undecided argument not to.
        return set(classes.mutated | classes.unknown_receivers)

    def reset_caches(self) -> None:
        """Forget every cell snapshot of the previous simulation."""
        self.cache.reset()
        self._import_bindings_memo.clear()

    def _get_metadata_only(self, cache_key: str) -> dict | None:
        """Get only metadata for a cache key without deserializing the full value.

        Delegates to ``backend.get_metadata()``, which every
        :class:`cash.backends.CacheBackend` provides (the base supplies a
        ``get()``-discard-value fallback; ``FileBackend`` overrides for a
        cheaper metadata-only read path). Lets callers skip the expensive
        deserialization of large cached objects (e.g. DataFrames) when
        only metadata is needed.
        """
        backend = self.backend()
        if backend is None:
            return None
        return backend.get_metadata(cache_key)

    def backend(self):
        """The cache backend the simulation probes, or None without a Cash."""
        return self.cash_instance.backend if self.cash_instance else None

    def record_replayed_file_deps(self, rerecorded: set[str]) -> None:
        """Add the files behind the *rerecorded* variables to the snapshots of
        the cells that produce them.

        A cell's snapshot records the files the simulation knew of. Before a
        replay -- after a restart above all -- the simulation cannot find a
        statement's cache entry (nothing is live yet to key it with), so the
        snapshot has no file dependency for a file read inside a helper; the
        replay then restores the statement, and the snapshot stays blind. A
        re-delivered file was never looked at again and the old result was
        served. With the files the restore brought back
        (``executed_file_deps``) in the snapshot, a later change of one makes
        the next simulation redo the cell, as it always did without a restart.

        Only the file list changes. Re-simulating the replayed cells instead
        exposed lineages the simulation cannot rebuild (a view-of-view's bump
        of its root base), and the replay then rebuilt the base under a view
        that still pointed at the old one.
        """
        if not rerecorded:
            return
        for entry in self.cache.entries:
            for trace_entry in entry.trace_segment:
                for var in set(trace_entry.outputs) & rerecorded:
                    for path in self.tracking_state.executed_file_deps.get(var, ()):
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
                logger.debug(
                    "[UPSTREAM_DEBUG] File dependency changed: %s (cached mtime=%s, current=%s)",
                    resolved,
                    stored_mtime,
                    st.st_mtime,
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
        for idx in range(min(current_cell_idx, len(self.cache.entries))):
            cell_code = notebook_cells[idx].replace("\r\n", "\n")
            cell_hash = hashlib.sha256(cell_code.encode("utf-8")).hexdigest()
            cached = self.cache.entries[idx]
            if cached.cell_code_hash != cell_hash:
                cache_had_hash_mismatch = True
                logger.debug(
                    "[UPSTREAM_DEBUG] Hash mismatch in cell %d (cached=%s, current=%s). Re-simulating from here.",
                    idx,
                    cached.cell_code_hash[:12],
                    cell_hash[:12],
                )
                break
            if cached.cell_environment != self._cell_environment(cell_code):
                # An environment variable the cell reads has another value:
                # its outputs' lineages fold it, so re-simulate from here --
                # like a changed file, not like edited code.
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
        cache_range_end = min(current_cell_idx, len(self.cache.entries)) if self.cache.entries else 0
        for idx in range(cache_range_end, current_cell_idx):
            if idx not in self.cache.cell_hashes:
                continue
            cell_code = notebook_cells[idx].replace("\r\n", "\n")
            cell_hash = hashlib.sha256(cell_code.encode("utf-8")).hexdigest()
            if self.cache.cell_hashes[idx] != cell_hash:
                logger.debug(
                    "[UPSTREAM_DEBUG] Hash mismatch in cell %d "
                    "(detected via lightweight hash cache, main cache truncated)",
                    idx,
                )
                return True
        return False

    def _restore_cached_state(self, first_changed_cell: int) -> SimulationResult:
        """The simulation state after the cached cells before *first_changed_cell*."""
        cached_entry = self.cache.entries[first_changed_cell - 1]
        sim = SimulationResult(
            virtual_lineage=dict(cached_entry.virtual_lineage),
            virtual_modules=set(cached_entry.virtual_modules),
        )
        for ci in range(first_changed_cell):
            sim.trace.extend(self.cache.entries[ci].trace_segment)
            sim.vars_mutated_by_loops.update(self.cache.entries[ci].vars_mutated_by_loops)
            sim.vars_with_stale_files.update(self.cache.entries[ci].vars_with_stale_files)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[UPSTREAM_DEBUG] Incremental simulation: reusing cache for cells 0-%d, simulating from cell %d",
                first_changed_cell - 1,
                first_changed_cell,
            )
        return sim

    def find_incremental_start(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> IncrementalStartResult:
        """Find the first upstream cell that changed since last simulation.

        Compares cached simulation hashes with current notebook cells and checks
        file dependency mtimes. Returns the index to start re-simulation from,
        along with restored cached state (virtual lineage, modules, trace, etc.).
        """
        sim = SimulationResult()
        first_changed_cell = 0
        had_prior_cache = bool(self.cache.entries)
        cache_had_hash_mismatch = False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[UPSTREAM_DEBUG] simulate_upstream: current_cell_idx=%d, "
                "had_prior_cache=%s, cache_size=%d, cell_hashes_size=%d",
                current_cell_idx,
                had_prior_cache,
                len(self.cache.entries) if self.cache.entries else 0,
                len(self.cache.cell_hashes),
            )

        if self.cache.entries:
            first_changed_cell, cache_had_hash_mismatch = self._scan_main_cache_for_changes(
                current_cell_idx, notebook_cells
            )

        # A reloaded helper module changes what cells compute without changing
        # their text or their files, which is all that scan compares, so it
        # replayed the pre-edit simulation and a cell below the helper's caller
        # printed the pre-edit value. Re-simulate from the first cell
        # that READS the module -- see TrackingState.reloaded_names for why not
        # from the import. Like a file change, this is not flagged as an
        # upstream CODE modification, which would withdraw trust from every
        # loop in the notebook.
        state = self.tracking_state
        generation = state.module_generation
        reloaded: set[str] = set()
        if generation != self._simulated_module_generation:
            self._simulated_module_generation = generation
            reloaded = set(state.reloaded_names)
            state.reloaded_names = set()
            reader = _first_cell_reading(notebook_cells, current_cell_idx, reloaded)
            if reader is not None and reader < first_changed_cell:
                first_changed_cell = reader
                logger.debug(
                    "[UPSTREAM_DEBUG] A tracked module was reloaded; re-simulating from cell %d, its first reader.",
                    reader,
                )

        # Check the lightweight hash cache for cells beyond the main cache range.
        if not cache_had_hash_mismatch and self.cache.cell_hashes:
            if self._check_lightweight_hash_cache(current_cell_idx, notebook_cells):
                cache_had_hash_mismatch = True

        # Restore cached state for cells before the first change, regardless of
        # what type of change was detected (code hash OR file dep staleness).
        # Without this, stale file deps would cause ALL cached state to be lost,
        # even for cells before the stale cell.
        if first_changed_cell > 0 and self.cache.entries and first_changed_cell <= len(self.cache.entries):
            sim = self._restore_cached_state(first_changed_cell)
        # The cells kept from the cache include the import, so a reloaded
        # module's name still carries its PRE-edit lineage there. A loop that
        # read it recorded its outcome against that lineage and found it
        # matching -- a regional table was adopted stale. The invalidator
        # has already given the name its new lineage; use that one. Module
        # names only: a from-imported name's lineage is cleared on purpose.
        for name in reloaded:
            live = self.tracking_state.variable_lineage.get(name)
            if live and isinstance(self.shell.user_ns.get(name), types.ModuleType):
                sim.virtual_lineage[name] = live

        new_cache_entries = list(self.cache.entries[:first_changed_cell]) if self.cache.entries else []

        return IncrementalStartResult(
            first_changed_cell=first_changed_cell,
            had_prior_cache=had_prior_cache,
            cache_had_hash_mismatch=cache_had_hash_mismatch,
            new_cache_entries=new_cache_entries,
            simulation=sim,
        )

    def simulate(self, current_cell_idx: int, notebook_cells: list[str]) -> SimulationResult:
        """Pass 1: simulate every cell above *current_cell_idx*, starting from
        the first one changed since the previous simulation."""
        start = self.find_incremental_start(current_cell_idx, notebook_cells)
        sim = start.simulation
        self.simulate_cells_pass1(
            sim, start.first_changed_cell, current_cell_idx, notebook_cells, start.new_cache_entries
        )
        # True only for an EDIT to a cell the previous simulation saw: a cell
        # merely new to the cache (first run of cell 3 after cell 1) is not one.
        sim.upstream_has_modifications = start.had_prior_cache and start.cache_had_hash_mismatch
        logger.debug(
            "[UPSTREAM_DEBUG] upstream_has_modifications=%s "
            "(had_prior_cache=%s, cache_had_hash_mismatch=%s, first_changed_cell=%s)",
            sim.upstream_has_modifications,
            start.had_prior_cache,
            start.cache_had_hash_mismatch,
            start.first_changed_cell,
        )
        return sim

    def _collect_notebook_statements(self, notebook_cells: list[str]) -> set[str]:
        """Collect all normalized statement codes from notebook cells.

        Used to distinguish downstream statements from unsaved extensions.
        """
        all_notebook_stmts: set[str] = set()
        for cell_code in notebook_cells:
            try:
                if clean_cell_source(cell_code).strip():
                    tree = parse_cell_source(cell_code)
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

    def reapply_unsaved_extensions(
        self,
        broken_vars: set[str],
        vars_updated_by_trace: set[str],
        simulation_trace: list,
        notebook_cells: list[str],
        statements_to_reexecute: list[str],
    ) -> None:
        """Re-apply unsaved extension code for broken variables.

        If a broken variable's producing code is NOT in the notebook (unsaved
        extension), schedule it for re-execution — unless the trace already
        updated that variable.
        """
        all_notebook_stmts = self._collect_notebook_statements(notebook_cells)

        for var_name in broken_vars:
            if var_name in vars_updated_by_trace:
                continue

            if var_name in self.tracking_state.executed_cell_codes:
                mem_code = self.tracking_state.executed_cell_codes[var_name]

                is_in_trace = False
                for entry in simulation_trace:
                    if entry.stmt_code.strip() == mem_code.strip():
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
        "for ",
        "while ",
        "async for ",
        "if ",
        "elif ",
        "else:",
        "with ",
        "async with ",
        "try:",
        "except ",
        "except:",
        "finally:",
    )

    def loop_accumulators_with_external_init(
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
                stmt_code, outputs, inputs = entry.stmt_code, entry.outputs, entry.inputs
                if acc not in outputs or acc in inputs:
                    continue  # not a producer, or self-referential (loop body)
                if stmt_code.lstrip().startswith(self._CTRL_PREFIXES):
                    continue  # loop/control wrapper; iterable feeds via the loop
                for inp in inputs:
                    if inp in loop_target_vars or self._unbound_builtin(inp):
                        continue
                    val = self.shell.user_ns.get(inp)
                    if val is not None and isinstance(val, types.ModuleType):
                        continue
                    tainted.add(acc)
                    break
                if acc in tainted:
                    break
        return tainted

    def loops_reading_changed_data(
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
        served from the old data.

        So compare each data input of a loop producing an accumulator, as the
        simulation has it at that point, with the lineage it had when the
        loop last ran (``TrackingState.control_outcomes``, recorded on entry).
        Inputs that are themselves loop-built or loop targets are skipped:
        their lineages disagree by construction.
        """
        changed: set[str] = set()
        outcomes = self.tracking_state.control_outcomes
        for entry in simulation_trace:
            stmt_code, outputs, inputs, input_hashes = (
                entry.stmt_code,
                entry.outputs,
                entry.inputs,
                entry.input_hashes or {},
            )
            accs = outputs & vars_mutated_by_loops
            if not accs or not stmt_code.lstrip().startswith(self._CTRL_PREFIXES):
                continue
            recorded = outcomes.get(hashlib.sha256(stmt_code.encode("utf-8")).hexdigest())
            if recorded is None:
                continue
            for inp in inputs:
                if (
                    inp in outputs
                    or inp in loop_target_vars
                    or inp in vars_derived_from_loops
                    or self._unbound_builtin(inp, input_hashes)
                ):
                    continue
                if isinstance(self.shell.user_ns.get(inp), types.ModuleType):
                    continue
                now, then = input_hashes.get(inp), recorded[0].get(inp)
                if now is not None and then is not None and now != then:
                    changed |= accs
                    break
        return changed

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
        logger.debug("[UPSTREAM] Checking skipped stmt [%d]: %.30s...", i, stmt_code)
        try:
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                key_inputs(inputs, input_hashes),
                ctx=CacheKeyContext(
                    variable_lineage=self.tracking_state.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker,
                    virtual_lineage=key_lineages(input_hashes),
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                    virtual_callables=self._virtual_callables,
                ),
                outputs=outputs,
            )
            metadata = self._get_metadata_only(cache_key)
            if metadata:
                saved_time = metadata.get("execution_time", 0.0)
                is_metadata_only = metadata.get("metadata_only", False)
                logger.debug(
                    "[UPSTREAM] Skipped stmt [%d] hit cache. Saved: %ss (metadata_only=%s)",
                    i,
                    saved_time,
                    is_metadata_only,
                )
                entry: dict = {
                    "code": stmt_code,
                    "status": CacheStatus.SKIPPED,
                    "saved_time": saved_time,
                    "is_upstream": True,
                    "source": "Skipped",
                    "position": i,
                    "has_cache": True,
                }
                if "storage" in metadata:
                    entry["storage"] = metadata["storage"]
                return entry
            logger.debug("[UPSTREAM] Skipped stmt [%d] miss cache: %s. Key: %s", i, stmt_code[:60], cache_key)
            return {
                "code": stmt_code,
                "status": CacheStatus.SKIPPED,
                "saved_time": 0.0,
                "is_upstream": True,
                "source": "Skipped",
                "position": i,
                "has_cache": False,
            }
        except (KeyError, TypeError, OSError, ValueError) as e:
            logger.debug("[UPSTREAM] Error checking skipped stmt: %s", e)
            return None

    def collect_skipped_statement_metrics(
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
            if "position" in info:
                restored_indices.add(info["position"])
                if "restored_vars" in info:
                    restored_outputs.update(info["restored_vars"])

        dependency_chain: set[int] = set()
        if restored_outputs:
            needed = set(restored_outputs)
            for i in range(len(simulation_trace) - 1, -1, -1):
                entry = simulation_trace[i]
                if entry.outputs & needed:
                    dependency_chain.add(i)
                    needed.update(entry.inputs)

        skipped_metrics: list[dict] = []
        for i, entry in enumerate(simulation_trace):
            if i in executed_indices or i in restored_indices or i not in dependency_chain:
                continue
            metric = self._skipped_stmt_metric(
                i, entry.stmt_code, entry.outputs, entry.inputs, entry.input_hashes, virtual_modules
            )
            if metric is not None:
                skipped_metrics.append(metric)
        return skipped_metrics

    def _is_reinit_to_skip(
        self,
        idx: int,
        simulation_trace: list,
        scheduled_iteration_outputs: dict[str, list],
        vars_mutated_by_loops: set[str],
        fully_rerun_mutated: set[str],
    ) -> bool:
        """Return True if the statement at *idx* is an accumulator init that should be skipped.

        Skips when the statement initialises to an empty container (e.g. ``x = {}``)
        but the accumulator already has data in memory, to avoid wiping state.
        """
        stmt_code, outputs = simulation_trace[idx].stmt_code, simulation_trace[idx].outputs
        if iteration_digest(stmt_code) is not None:
            return False
        if len(outputs) != 1:
            return False
        out_var = list(outputs)[0]
        if out_var in fully_rerun_mutated:
            # When the loop that mutates out_var is itself fully re-executed, the
            # init must run alongside it, else the accumulation doubles; skipping
            # is only safe for pure incremental extension of a cached loop.
            return False
        is_loop_updated = out_var in scheduled_iteration_outputs or out_var in vars_mutated_by_loops
        if not is_loop_updated:
            return False
        stripped = stmt_code.strip()
        empty_init_pattern = re.compile(
            rf"^{re.escape(out_var)}\s*=\s*(\{{\}}|\[\]|set\(\)|dict\(\)|list\(\)|frozenset\(\))$"
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
            logger.debug(
                "[UPSTREAM] Skipping accumulator init '%.40s' - already has %d items in memory",
                stmt_code,
                len(existing_val),
            )
            return True
        return False

    def _loop_vars_fully_rescheduled(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        vars_mutated_by_loops: set[str],
    ) -> set[str]:
        """Loop-mutated vars whose mutation is scheduled OUTSIDE a cached iteration-context body (=> full re-run)."""
        if not vars_mutated_by_loops:
            return set()
        patterns = {
            mv: re.compile(rf"\b{re.escape(mv)}\s*(?:\.\s*\w+\s*\(|\[[^\]]*\]\s*=(?!=))")
            for mv in vars_mutated_by_loops
        }
        fully_rerun_mutated: set[str] = set()
        for idx in stmts_to_run_indices:
            stmt_code, outputs = simulation_trace[idx].stmt_code, simulation_trace[idx].outputs
            if iteration_digest(stmt_code) is not None:
                continue
            for mv, pat in patterns.items():
                # Only a statement that WRITES it: `def draw_roc` iterating
                # `results.items()` matched the text, so the init was scheduled
                # for a loop that was not, and `results` was re-run empty.
                if mv not in fully_rerun_mutated and mv in outputs and pat.search(stmt_code):
                    fully_rerun_mutated.add(mv)
        return fully_rerun_mutated

    def filter_accumulator_reinits(
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
        scheduled_iteration_outputs: dict[str, list] = {}
        for idx in stmts_to_run_indices:
            stmt_code, outputs = simulation_trace[idx].stmt_code, simulation_trace[idx].outputs
            if iteration_digest(stmt_code) is not None:
                for out in outputs:
                    scheduled_iteration_outputs.setdefault(out, []).append(idx)

        fully_rerun_mutated = self._loop_vars_fully_rescheduled(
            stmts_to_run_indices,
            simulation_trace,
            vars_mutated_by_loops,
        )

        # A fully re-run loop replays its in-place mutations (.append / [k]=)
        # onto whatever the accumulator currently holds.  If the empty-container
        # init was never scheduled (the backward scan often schedules only the
        # loop body, treating the accumulator output as already satisfied), the
        # replay doubles the accumulated value.  Schedule the missing init so it
        # runs alongside the loop.  (Pure incremental extension keeps the init
        # unscheduled and is handled by the removal pass below.)
        stmts_to_run_indices = self._schedule_missing_accumulator_inits(
            stmts_to_run_indices,
            simulation_trace,
            fully_rerun_mutated,
        )

        indices_to_remove: set[int] = set()
        for idx in stmts_to_run_indices:
            if self._is_reinit_to_skip(
                idx,
                simulation_trace,
                scheduled_iteration_outputs,
                vars_mutated_by_loops,
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
            mv: re.compile(rf"^{re.escape(mv)}\s*=\s*(\{{\}}|\[\]|set\(\)|dict\(\)|list\(\)|frozenset\(\))$")
            for mv in fully_rerun_mutated
        }
        additional: list[int] = []
        for idx, entry in enumerate(simulation_trace):
            if idx in scheduled:
                continue
            stmt_code, outputs = entry.stmt_code, entry.outputs
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

    def check_loop_derived_trust_override(
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
            if mv not in self.tracking_state.executed_cell_codes:
                continue
            exec_code = strip_markers(self.tracking_state.executed_cell_codes[mv]).strip()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[UPSTREAM_DEBUG] Checking loop trust for '%s': exec_code=%s", mv, repr(exec_code[:60]))
                matching = [sc for sc in simulation_trace_codes if exec_code in sc or sc in exec_code]
                logger.debug(
                    "[UPSTREAM_DEBUG]   Partial matches in simulation_trace_codes: %s", [repr(m[:60]) for m in matching]
                )
            if exec_code and exec_code not in simulation_trace_codes:
                logger.debug(
                    "[UPSTREAM_DEBUG] Loop-mutated var '%s' was produced by code "
                    "not found on disk (unsaved edit or stale execution). Distrusting ALL loop-derived vars.",
                    mv,
                )
                return True
        return False

    def build_loop_var_input_lineages(
        self,
        simulation_trace: list,
        vars_derived_from_loops: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> dict[str, dict[str, str]]:
        """Return a mapping of loop-derived variable → its data-input virtual lineages.

        Used to detect when loop inputs change (e.g., N=10→20) even when the
        producing code is unchanged on disk.
        """
        loop_var_input_lineages: dict[str, dict[str, str]] = {}
        for entry in simulation_trace:
            for out in entry.outputs:
                if out in vars_derived_from_loops:
                    data_input_lineages: dict[str, str] = {}
                    for inp in entry.inputs:
                        if inp in virtual_modules:
                            continue
                        if inp in virtual_lineage:
                            data_input_lineages[inp] = virtual_lineage[inp]
                    loop_var_input_lineages[out] = data_input_lineages
        return loop_var_input_lineages

    def build_simulation_trace_codes(self, simulation_trace: list) -> set[str]:
        """Return the set of normalised statement codes present in *simulation_trace*.

        Includes body-level statements from control structures so that
        per-iteration cache entries (which record body statements rather than
        the whole for-loop) are matched correctly.
        """
        simulation_trace_codes: set[str] = set()
        for entry in simulation_trace:
            normalized = strip_markers(entry.stmt_code).strip()
            simulation_trace_codes.add(normalized)
            try:
                tree = parse_cached(normalized)
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

    def simulate_one_node(
        self,
        sim: SimulationResult,
        i: int,
        node: ast.AST,
        cell_stmt_occurrence_counts: dict[str, int],
        cell_file_deps: dict[str, float],
        raw_cell: str | None = None,
    ) -> None:
        """Simulate one top-level statement of cell *i* into *sim*.

        A control structure is simulated as one unit
        (``_simulate_control_structure``).

        *raw_cell* is the text *node* was parsed from. The runtime keys an
        expression followed by ``;`` WITH the ``;`` (IPython's display
        suppression), which ``ast.unparse`` drops -- so ``ax.bar(...);
        ax.set_xlabel(...)`` on one line got another key and lineage here, and
        every chart drawn that way disagreed.
        """
        try:
            if is_control_structure(node):
                self._simulate_control_structure(node, sim)
                return

            stmt_code = ast.unparse(node)
            if raw_cell is not None:
                # Local: import cycle upstream.virtual_lineage -> ipython.cell_executor -> ... -> upstream.virtual_lineage.
                from ..ipython.cell_executor import CellExecutor

                if CellExecutor.expr_has_trailing_semicolon(raw_cell, node):
                    stmt_code += ";"
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("[UPSTREAM] Error processing node in cell %d: %s", i, e)
            raise

        occ = cell_stmt_occurrence_counts.get(stmt_code, 0)
        cell_stmt_occurrence_counts[stmt_code] = occ + 1
        occurrence_index = occ  # 0-based

        virtual_lineage = sim.virtual_lineage
        virtual_modules = sim.virtual_modules
        inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        input_hashes: dict[str, str] = {}
        for inp in inputs:
            if inp in virtual_lineage:
                input_hashes[inp] = virtual_lineage[inp]
            elif inp in self.tracking_state.variable_lineage:
                input_hashes[inp] = self.tracking_state.variable_lineage[inp]
        callee_lineages = self._virtual_callee_lineages(inputs, virtual_lineage, virtual_modules)
        hidden_lineages = {
            var: virtual_lineage.get(var, self.tracking_state.variable_lineage.get(var))
            for var in key_hidden_reads(stmt_code, self.tracking_state)
        }
        if callee_lineages or hidden_lineages:
            input_hashes = _InputHashes(input_hashes, callee_lineages, hidden_lineages)

        outputs, lookup_time, files_stale, stmt_file_deps = self._update_virtual_lineage(
            stmt_code,
            virtual_lineage,
            virtual_modules,
            occurrence_index=occurrence_index,
        )

        if stmt_file_deps:
            cell_file_deps.update(stmt_file_deps)

        self._update_stale_file_deps(inputs, outputs, files_stale, sim.vars_with_stale_files)

        if outputs:
            produced_lineages = {out: virtual_lineage[out] for out in outputs if out in virtual_lineage}
            sim.trace.append(TraceEntry(stmt_code, outputs, inputs, input_hashes, produced_lineages, files_stale))
            if lookup_time > 0:
                sim.stmt_lookup_times[stmt_code] = lookup_time
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
            # a figure was rebuilt without it.
            if statement_writes_files(stmt_code) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
                sim.trace.append(TraceEntry(stmt_code, outputs, inputs, input_hashes, {}, files_stale))

    def simulate_one_cell(
        self,
        sim: SimulationResult,
        i: int,
        cell_code: str,
        new_cache_entries: list[SimulationCacheEntry] | None = None,
    ) -> None:
        """Simulate cell *i* into *sim*, and append its snapshot to
        *new_cache_entries* when given."""
        if new_cache_entries is None:
            new_cache_entries = []
        cell_hash = hashlib.sha256(cell_code.encode("utf-8")).hexdigest()
        simulation_trace = sim.trace
        virtual_lineage = sim.virtual_lineage
        virtual_modules = sim.virtual_modules
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
        for line in cell_code.split("\n"):
            if line.strip().startswith("%reset"):
                virtual_lineage.clear()
                virtual_modules.clear()

        try:
            clean_cell_code = clean_cell_source(cell_code)
            if not clean_cell_code.strip():
                new_cache_entries.append(
                    SimulationCacheEntry(
                        cell_code_hash=cell_hash,
                        virtual_lineage=dict(virtual_lineage),
                        virtual_modules=set(virtual_modules),
                        trace_segment=[],
                        vars_mutated_by_loops=set(),
                        vars_with_stale_files=set(),
                        cell_file_deps={},
                    )
                )
                return

            tree = parse_cell_source(cell_code)
            if tree is None:
                ast.parse(clean_cell_code)  # will raise SyntaxError

            cell_stmt_occurrence_counts: dict = {}

            for node in tree.body:
                # A top-level ``raise`` unconditionally aborts the cell — every
                # statement after it is dead code that never runs in a real
                # from-start execution. Stop here so the simulation does not
                # register a post-raise assignment (``z = 1; raise; z = 2``) as
                # the variable's producer and later reconstruct that dead value
                # .
                if isinstance(node, ast.Raise):
                    break
                self.simulate_one_node(
                    sim, i, node, cell_stmt_occurrence_counts, cell_file_deps, raw_cell=clean_cell_code
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
            entry.cell = i
        cell_trace_segment = simulation_trace[trace_start:]
        new_cache_entries.append(
            SimulationCacheEntry(
                cell_code_hash=cell_hash,
                virtual_lineage=dict(virtual_lineage),
                virtual_modules=set(virtual_modules),
                trace_segment=cell_trace_segment,
                vars_mutated_by_loops=set(sim.vars_mutated_by_loops),
                vars_with_stale_files=set(sim.vars_with_stale_files),
                cell_file_deps=dict(cell_file_deps),
                cell_environment=self._cell_environment(cell_code),
            )
        )

    def _cell_environment(self, cell_code: str) -> str:
        """What the environment reads written in *cell_code* return now
        (``statement_environment_component``), for telling a cell simulated
        under another value from one that may be reused."""
        return statement_environment_component(clean_cell_source(cell_code), self.shell.user_ns)

    def simulate_cells_pass1(
        self,
        sim: SimulationResult,
        first_changed_cell: int,
        current_cell_idx: int,
        notebook_cells: list[str],
        new_cache_entries: list[SimulationCacheEntry],
    ) -> None:
        """Run pass-1 simulation for cells *first_changed_cell*..*current_cell_idx* and update caches."""
        # Source of every top-level function across all cells, so
        # ``_mutation_receivers`` can decide which bare ``proc(d)`` calls mutate
        # their argument (headless: inspect.getsource has no linecache entry).
        self._sim_func_sources = self._build_function_sources(notebook_cells)
        for i in range(first_changed_cell, current_cell_idx):
            cell_code = notebook_cells[i].replace("\r\n", "\n")
            self.simulate_one_cell(sim, i, cell_code, new_cache_entries)

        # Update simulation cache for future incremental simulation.
        # NOTE: We only store entries for cells 0..(current_cell_idx-1).
        # Entries beyond that are discarded to avoid stale lineage data.
        # For hash change detection across intermediate cell runs, we use
        # cache.cell_hashes (a separate lightweight structure).
        self.cache.entries = new_cache_entries

        # This persists across intermediate cell runs so that a later cell can
        # detect code changes in cells that were truncated from the main cache.
        for idx, entry in enumerate(new_cache_entries):
            self.cache.cell_hashes[idx] = entry.cell_code_hash

        # Also record the CURRENT cell's hash so that a later cell (e.g., cell 3
        # running after cell 2 in a run_all()) sees the up-to-date hash and
        # doesn't falsely detect a modification from a stale hash left over
        # from a previous run_all().
        if current_cell_idx < len(notebook_cells):
            current_cell_code = notebook_cells[current_cell_idx].replace("\r\n", "\n")
            self.cache.cell_hashes[current_cell_idx] = hashlib.sha256(current_cell_code.encode("utf-8")).hexdigest()

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
        if isinstance(node, ast.For):
            loop_target_vars.update(extract_target_names(node.target))
        if not isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try)):
            return set()
        mutated_vars = control_structure_mutations(node, self._unbound_builtin)
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
        source_hash = statement_source_hash(stmt_code)
        for mv in mutated_vars:
            if mv not in outputs and mv in inputs:
                new_lineage = output_lineage(source_hash, input_hashes.values())
                virtual_lineage[mv] = new_lineage
                extra_outputs.add(mv)
                logger.debug(
                    "[UPSTREAM_DEBUG] Loop-mutated var '%s' virtual lineage updated to %s...",
                    mv,
                    new_lineage[:12],
                )
        return extra_outputs

    def _simulate_control_structure(self, node: ast.AST, sim: SimulationResult) -> None:
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
        unrelated upstream edit from re-planning such a loop is the
        outcome the runtime recorded for it -- see
        ``TrackingState.control_outcomes`` in ``_simulate_one_control_unit``.
        """
        # A loop with a recorded split verdict is modelled as TWO statements.
        #
        # This is the mechanism, not a parity nicety: the re-execution planner
        # runs the statements simulated here, so splitting this model is what
        # actually makes the runtime execute a head and a tail. Splitting only
        # in the runtime leaves the planner re-running the whole loop against
        # entries written for halves -- a silent stale value, and the cause of
        # three reverted attempts. See ``notebook/loop_split.py``.
        #
        # Unless the loop's last run left an outcome that still holds. A
        # verdict applies from the NEXT run, so a loop that learned it ran
        # whole, and so did one a direct re-run split (the runtime records
        # its outcome under the whole loop's source either way). Its halves
        # have no outcome of their own until the planner runs them as two
        # statements, so modelled as halves after a restart, what the loop
        # built got lineages no entry was written with, and a restart plus
        # one run of the last cell replayed the loop and everything above
        # it -- whenever the first run's timing had recorded a verdict. The
        # record holding means the loop's inputs, files and callees are what
        # they were, so it has nothing to re-run and its outputs are the
        # recorded ones; any change falls through to the split.
        split_k = self._loop_split_k(node)
        if split_k is not None and not self._recorded_outcome_holds(node, sim):
            try:
                halves = split_nodes(node, split_k)
            except ValueError:  # for/else -- not splittable
                halves = ()
            if halves:
                logger.debug("[UPSTREAM_DEBUG] loop split at k=%d -> simulating head and tail separately", split_k)
                for half in halves:
                    self._simulate_one_control_unit(half, sim)
                return

        self._simulate_one_control_unit(node, sim)

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
            if is_split_half(node):
                return None  # never split a half; that recurses
            if self._split_store is None:
                backend = self.cash_instance.backend if self.cash_instance else None
                self._split_store = store_for_backend(backend)
                if self._split_store is None:
                    return None
            return self._split_store.get(loop_source_hash(node))
        except Exception:  # noqa: BLE001 - never let a lookup break simulation
            logger.debug("[UPSTREAM_DEBUG] loop split lookup failed", exc_info=True)
            return None

    def _simulate_one_control_unit(self, node: ast.AST, sim: SimulationResult) -> None:
        """Simulate ONE control structure as a single statement.

        Split out of :meth:`_simulate_control_structure` so a split loop can
        run it twice -- the second half seeing the virtual lineage the first
        produced, exactly as two source-level statements would.
        """
        stmt_code = ast.unparse(node)
        virtual_lineage = sim.virtual_lineage

        inputs, input_hashes = self._control_input_hashes(stmt_code, virtual_lineage)

        outputs, lookup_time, files_stale, _ = self._update_virtual_lineage(
            stmt_code, virtual_lineage, sim.virtual_modules
        )

        mutated_vars = self._collect_loop_mutation_info(node, sim.loop_target_vars, sim.vars_mutated_by_loops)

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

        # What the runtime left behind when it last ran this very structure
        # (see TrackingState.control_outcomes): the files it read, and the
        # lineages it produced.
        recorded = self._recorded_control_outcome(stmt_code, virtual_lineage)
        if recorded is not None:
            if compute_file_hash_component(recorded[2]) != recorded[3]:
                # A file behind its outputs moved: the one change the entry
                # lineages cannot show, because a `Path` does not change when
                # the file it names does. Say so, or the loop trust keeps the
                # stale value.
                #
                # Checked BEFORE the input-lineage comparison and regardless of
                # how it comes out, because the two answer different questions.
                # "Have the files this structure read changed?" is decided by
                # the files alone; the entry lineages have nothing to say about
                # it either way. Requiring a match first made the check
                # unreachable whenever a name the loop reads is bound in the
                # same cell as `%cash_on`:
                #
                #     import cash
                #     %cash_on
                #     DATA = Path(...)
                #
                # Such a name has no runtime lineage (cash was not yet
                # listening when that cell started) while the simulation, which
                # reads that cell from the file, has one -- so `entry` lacked a
                # key `input_hashes` carried, equality was false forever, and
                # neither branch ran. (The gap itself
                # is closed separately, in `_entry_lineages`; this check no
                # longer depends on it either way.)
                #
                # Marking the outputs stale can only cause a re-run, never a
                # restore, so running it on a mismatch is the safe direction of
                # the one it was already taking on a match.
                files_stale = True
                sim.vars_with_stale_files.update(all_outputs | set(recorded[1]))
            elif recorded[0] == input_hashes:
                # The runtime ran this very structure with these very inputs
                # and the files it read are where it left them: what it left
                # behind is the answer, not a formula it never used.
                virtual_lineage.update(recorded[1])
                all_outputs = all_outputs | set(recorded[1])

        if logger.isEnabledFor(logging.DEBUG):
            cs_type = get_control_structure_type(node) if node else "unknown"
            logger.debug(
                "[UPSTREAM_DEBUG] Simulating %s as single unit: %s... Outputs: %s", cs_type, stmt_code[:60], all_outputs
            )

        if all_outputs:
            produced_lineages = {out: virtual_lineage[out] for out in all_outputs if out in virtual_lineage}
            sim.trace.append(TraceEntry(stmt_code, all_outputs, inputs, input_hashes, produced_lineages, files_stale))
            if lookup_time > 0:
                sim.stmt_lookup_times[stmt_code] = lookup_time
        elif self._may_write_files(node, stmt_code):
            # ``if PACK.exists(): shutil.rmtree(PACK)`` binds nothing, so it had
            # no trace entry, and a replay after a restart re-ran the cell's
            # ``PACK.mkdir()`` without it.
            # The same rule simple statements follow in simulate_one_node.
            sim.trace.append(TraceEntry(stmt_code, set(), inputs, input_hashes, {}, files_stale))

    def _control_input_hashes(self, stmt_code: str, virtual_lineage: dict[str, str]) -> tuple[set[str], dict[str, str]]:
        """What *stmt_code* reads, and the lineage each of those names has here."""
        inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        input_hashes = {}
        for inp in inputs:
            if inp in virtual_lineage:
                input_hashes[inp] = virtual_lineage[inp]
            elif inp in self.tracking_state.variable_lineage:
                input_hashes[inp] = self.tracking_state.variable_lineage[inp]
        return inputs, input_hashes

    def _recorded_control_outcome(
        self, stmt_code: str, virtual_lineage: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, str], frozenset[str], str] | None:
        """The outcome the runtime recorded for *stmt_code*: this session's,
        else an earlier kernel's that may still be trusted."""
        recorded = self.tracking_state.control_outcomes.get(hashlib.sha256(stmt_code.encode("utf-8")).hexdigest())
        if recorded is None:
            recorded = self._persisted_control_outcome(stmt_code, virtual_lineage)
        return recorded

    def _recorded_outcome_holds(self, node: ast.AST, sim: SimulationResult) -> bool:
        """Whether *node*'s recorded outcome would be taken as its result here:
        read with these input lineages, and every file behind it unchanged --
        the two checks ``_simulate_one_control_unit`` makes before trusting it."""
        stmt_code = ast.unparse(node)
        try:
            _, input_hashes = self._control_input_hashes(stmt_code, sim.virtual_lineage)
            recorded = self._recorded_control_outcome(stmt_code, sim.virtual_lineage)
            return (
                recorded is not None
                and recorded[0] == input_hashes
                and compute_file_hash_component(recorded[2]) == recorded[3]
            )
        except Exception:  # noqa: BLE001 - any doubt keeps the split, as before
            logger.debug("[UPSTREAM_DEBUG] could not check a loop's recorded outcome", exc_info=True)
            return False

    def _persisted_mutation_verdict(self, source_hash: str) -> set[str] | None:
        """The runtime's verdict on a bare method call, from an earlier kernel.

        See ``mutation_verdict_key``. Kept in ``mutation_verdicts`` once read,
        where the runtime overwrites it when the statement runs again.
        """
        backend = self.backend()
        if backend is None:
            return None

        try:
            record = backend.get_metadata(mutation_verdict_key(source_hash))
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        if not record or not record.get("mutation_verdict"):
            return None
        verdict = set(record.get("receivers") or ())
        self.tracking_state.mutation_verdicts.setdefault(source_hash, verdict)
        return verdict

    def _persisted_control_outcome(
        self,
        stmt_code: str,
        virtual_lineage: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str], frozenset[str], str] | None:
        """A loop's outcome from an earlier kernel, when it may still be trusted.

        See ``control_outcome_key``. Written only for a loop whose outcome was
        all it did; trusted only when every global its callees read has, here,
        the lineage it had then -- the entry lineages and file state are
        checked by the caller, exactly as for the session's own record. Any
        doubt returns None, and the loop is replayed.
        """
        backend = self.backend()
        if backend is None:
            return None

        try:
            record = backend.get_metadata(control_outcome_key(stmt_code))
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        if not record or not record.get("control_outcome") or record.get("code") != stmt_code:
            return None
        try:
            for name, then in (record.get("callees") or {}).items():
                now = virtual_lineage.get(name) or self.tracking_state.variable_lineage.get(name) or "ABSENT"
                if now != then:
                    return None
            return (
                dict(record["entry"]),
                dict(record["left"]),
                frozenset(record["files"]),
                str(record["file_component"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _may_write_files(node: ast.AST, stmt_code: str) -> bool:
        """A write in the text, or a call to something that might be a
        user function that writes (the planner decides which)."""

        if statement_writes_files(stmt_code):
            return True
        return any(not hasattr(builtins, name) for name in called_names(node))

    # -- Helpers for _update_virtual_lineage ----------------------------------

    @staticmethod
    def _validate_file_freshness(
        hist_files: dict[str, Any],
        memo_key: str | None = None,
    ) -> bool:
        """Return True if all historical file dependencies are still fresh.

        Each entry is ``{path: {'mtime': ..., 'size': ...}}``. When ``size``
        is recorded it is checked too — that catches rewrites within a
        single mtime tick on coarse-resolution filesystems (HFS+/APFS,
        some ext4 configs).

        *memo_key* -- the entry's cache key. A "fresh" verdict holds for the
        rest of the cell run: the simulation re-validated the same upstream
        entry for every statement of the cell, twice -- 5,222 files x 2 x 24
        statements of one notebook. Within one run the upstream values are
        what a from-the-top run gives even if this cell later writes one of
        their files (upstream ran before the write), so re-checking can only
        repeat the answer.
        """

        epoch = _fds.HASH_EPOCH
        memo = _FRESH_ENTRY_VERDICTS
        if memo_key is not None and epoch is not None:
            if memo.get("epoch") != epoch:
                memo.clear()
                memo["epoch"] = epoch
                memo["keys"] = set()
            if memo_key in memo["keys"]:
                return True
        run = _file_state_this_run()
        fresh, stale = snapshot_is_fresh(hist_files, run["memo"] if run is not None else None)
        if not fresh:
            logger.debug("[UPSTREAM] Forward prop failed: stale file dependency (%s)", stale)
            return False
        if memo_key is not None and epoch is not None:
            memo["keys"].add(memo_key)
        return True

    def _resolve_virtual_input_lineages(
        self, stmt_code: str, inputs: set[str], virtual_lineage: dict[str, str], virtual_modules: set[str]
    ) -> list[str]:
        """Each input's lineage (``lineage_formula.input_lineage``), with the
        simulation's own lineages in front of the recorded ones."""
        input_lineages_all = []
        function_tracker = self.function_tracker
        for inp in sorted(inputs):
            if inp in {"get_ipython", "__builtins__"}:
                continue
            lineage = input_lineage(
                inp,
                self.shell.user_ns,
                (virtual_lineage, self.tracking_state.variable_lineage),
                compute_hash=self.compute_hash_fn,
                function_tracker=function_tracker,
                code=stmt_code,
                virtual_modules=virtual_modules,
            )
            if lineage:
                input_lineages_all.append(lineage)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[LINEAGE_DEBUG] %s... inputs %s -> %s",
                stmt_code[:50],
                sorted(inputs),
                [ln[:12] + "..." for ln in input_lineages_all],
            )
        return input_lineages_all

    @staticmethod
    def _stat_file_deps(hist_files: dict[str, float]) -> dict[str, float]:
        """Stat each path in *hist_files* and return ``{path: mtime}`` for existing files.

        Once per path per cell run (``_stats_this_run``), and from a directory
        listing where many share a directory."""
        return {p: st.st_mtime for p, (_resolved, st) in _stats_this_run(hist_files).items() if st is not None}

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

        Updates *virtual_lineage* (and optionally *self.tracking_state.variable_lineage* for imports)
        in place.  Returns ``('hit', 0.0, stmt_file_deps)`` where the caller
        should substitute the real ``cache_lookup_time``.
        """
        logger.debug("[UPSTREAM] Forward propagating cached lineages for %s...", stmt_code[:30])
        for var, h in output_lineages.items():
            virtual_lineage[var] = h
        # Even on a cache hit, replay the derivation-alias bump so a mutation of
        # a base/frame (its own lineage restored from cache here) still bumps its
        # live-alias derivatives. Same skip-inputs rule and
        # deterministic formula as the runtime and the miss path. Bumped vars are
        # threaded back so the caller can union them into ``outputs``.
        self._last_hit_bumped = bump_derived_lineages(
            self.tracking_state.derivation_edges,
            virtual_lineage,
            outputs,
            inputs,
            record=lambda t, h: virtual_lineage.__setitem__(t, h),
            present=lambda t: True,
        )
        if is_import:
            for out in outputs:
                if out not in self.tracking_state.variable_lineage:
                    lineage_val = output_lineages.get(out)
                    if lineage_val:
                        self.restores.record_restore(var_name=out, lineage_hash=lineage_val)
                        logger.debug(
                            "[LINEAGE_DEBUG] Propagated module '%s' lineage (from cache): %s...",
                            out,
                            lineage_val[:12],
                        )
        # Mid-simulation drain: same reasoning as in _propagate_import_lineage.
        apply_collected_mutations(self.restores, self.tracking_state)
        stmt_file_deps = self._stat_file_deps(hist_files)
        return ("hit", 0.0, stmt_file_deps)

    def _collect_historical_file_deps(
        self,
        hist_files: dict[str, float],
    ) -> tuple[set[str], dict[str, float]]:
        """Collect file dependency sets when cache propagation is aborted.

        Returns ``(file_deps_to_check, stmt_file_deps)``.
        """
        file_deps_to_check: set[str] = set(hist_files.keys())
        stmt_file_deps = self._stat_file_deps(hist_files)
        if logger.isEnabledFor(logging.DEBUG):
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
        on cache miss or failed validation, or None-wrapped early-return tuple isn't used—
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
            return ("miss", cache_lookup_time, files_stale, stmt_file_deps, file_deps_to_check)

        try:
            logger.debug("[UPSTREAM] Virtual lookup Key: %s", cache_key)

            t_lookup = time_module.time()
            metadata = self._get_metadata_only(cache_key)
            cache_lookup_time = time_module.time() - t_lookup

            if metadata:
                hist_files = metadata.get("file_dependencies", {})
                output_lineages = metadata.get("output_lineages", {})
                files_valid = not hist_files or self._validate_file_freshness(hist_files, memo_key=cache_key)

                if files_valid and output_lineages:
                    self._last_hit_bumped = set()
                    _sentinel, _, hit_file_deps = self._apply_cache_hit_propagation(
                        stmt_code,
                        cache_key,
                        outputs,
                        inputs,
                        virtual_lineage,
                        virtual_modules,
                        is_import,
                        metadata,
                        hist_files,
                        output_lineages,
                    )
                    return ("hit", cache_lookup_time, hit_file_deps, self._last_hit_bumped)

                if not files_valid:
                    files_stale = True

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[UPSTREAM] Forward prop aborted. files_valid=%s, output_lineages keys=%s",
                        files_valid,
                        list(output_lineages.keys()) if output_lineages else "None/Empty",
                    )

                if hist_files:
                    extra_fdeps, extra_stmt_deps = self._collect_historical_file_deps(hist_files)
                    file_deps_to_check.update(extra_fdeps)
                    stmt_file_deps.update(extra_stmt_deps)
        except (KeyError, TypeError, OSError, ValueError) as e:
            logger.debug("[UPSTREAM] Virtual lookup failed: %s", e)

        return ("miss", cache_lookup_time, files_stale, stmt_file_deps, file_deps_to_check)

    def _build_file_hash_component(self, file_deps_to_check: set[str], stmt_file_deps: dict[str, float]) -> str:
        """The file component of a statement's lineage when the runtime's own
        record of what it read is not available: the files its outputs depend
        on, valued by the runtime's formula (``compute_file_hash_component``).

        Also updates stmt_file_deps with current mtimes for tracked files.
        """
        if not file_deps_to_check:
            return ""

        present: set[str] = set()
        current = _stats_this_run(file_deps_to_check)
        for file_path in file_deps_to_check:
            resolved, stat = current[file_path]
            # Only a file that is where it was recorded, as before: the
            # relocation fallbacks would put a different path's state in a key.
            if stat is not None and resolved == file_path:
                present.add(file_path)
                stmt_file_deps[file_path] = stat.st_mtime
        return compute_file_hash_component(present) if present else ""

    #: See ``_virtual_callables``.
    _VIRTUAL_CALLABLES_MAX = 4096

    def _register_virtual_callable(
        self,
        stmt_code: str,
        tree: ast.Module | None,
        virtual_lineage: dict[str, str],
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
        if not lineage or virtual_callable_key(lineage, node.name) in self._virtual_callables:
            return
        try:
            module = compile(stmt_code, "<cash-simulated>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            return
        code = next((c for c in module.co_consts if isinstance(c, types.CodeType) and c.co_name == node.name), None)
        if code is None:
            return
        if len(self._virtual_callables) >= self._VIRTUAL_CALLABLES_MAX:
            self._virtual_callables.clear()
        self._virtual_callables[virtual_callable_key(lineage, node.name)] = VirtualCallable(
            source_identity_digest(stmt_code), code
        )

    def _register_imported_callables(
        self,
        tree: ast.Module | None,
        virtual_lineage: dict[str, str],
        stmt_code: str = "",
    ) -> None:
        """Remember what a simulated ``from X import Y`` binds, as the live object would count.

        The runtime folds a source digest into the lineage of every statement
        that reads a callable -- a class included -- and into the key for a
        function. After a restart ``EXPORTS = Path(...)`` was simulated before
        the import had run again: no ``Path`` in ``user_ns``, no digest, and a
        lineage the runtime never gave ``EXPORTS`` -- so nothing built from it
        restored. The digest comes from the module object
        when it is already imported (``pathlib`` always is), else from what the
        statement bound when it last ran (``_import_bindings``): the simulation
        never imports anything itself.
        """
        if tree is None:
            return
        tracker = self.function_tracker
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            module = sys.modules.get(node.module)
            for alias in node.names:
                name = alias.asname or alias.name
                lineage = virtual_lineage.get(name)
                if not lineage or name in self.shell.user_ns:
                    continue
                obj = getattr(module, alias.name, None) if module is not None else None
                if obj is not None and tracker is not None:
                    if not callable(obj):
                        continue
                    try:
                        digest = tracker.get_function_source_hash(obj)
                    except Exception:  # noqa: BLE001 - a digest we cannot take is one we do not claim
                        continue
                    code = getattr(obj, "__code__", None)
                    is_class = isinstance(obj, type)
                else:
                    entry = self._import_bindings(stmt_code).get(name) if stmt_code else None
                    if not entry or entry.get("module"):
                        continue
                    digest, is_class, code = entry.get("digest"), bool(entry.get("is_class")), None
                    if entry.get("code"):
                        try:
                            code = marshal.loads(base64.b64decode(entry["code"]))
                        except (ValueError, EOFError, TypeError):
                            code = None
                if digest is None:
                    continue
                if is_class:
                    self._imported_classes[virtual_callable_key(lineage, name)] = digest
                elif isinstance(code, types.CodeType):
                    self._virtual_callables.setdefault(
                        virtual_callable_key(lineage, name), VirtualCallable(digest, code)
                    )

    def _virtual_callee_lineages(
        self,
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> dict[str, str] | None:
        """Lineages, here, of the globals read by callees that exist only as simulated defs."""
        user_ns = self.shell.user_ns
        if not self._virtual_callables or all(name in user_ns for name in inputs):
            return None
        virtual = virtual_namespace(
            self._virtual_callables, virtual_lineage, self.tracking_state.variable_lineage, virtual_modules
        )
        deps = called_function_dependencies(sorted(inputs), user_ns, self.tracking_state.variable_lineage, virtual)
        found = dict(dep.split(":", 1) for dep in deps)
        return {name: lin for name, lin in found.items() if lin != "ABSENT"} or None

    def absent_callee_globals(
        self,
        inputs: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> set[str]:
        """Names the callees in *inputs* read that the kernel does not hold.

        A statement re-run to rebuild a value needs them bound, and its own
        inputs do not name them: after a restart ``summary = score(raw)`` was
        re-run with ``score``'s ``OFFSET`` never rebuilt -- a NameError, where
        the cell had run fine. Modules included: the def's cell imported them.
        """

        user_ns = self.shell.user_ns
        virtual = (
            virtual_namespace(
                self._virtual_callables, virtual_lineage, self.tracking_state.variable_lineage, virtual_modules
            )
            if self._virtual_callables
            else None
        )
        names = called_function_globals(inputs, user_ns, virtual, keep_modules=True)
        return {name for name in names if name not in user_ns}

    def _virtual_callable_hashes(self, inputs: set[str], virtual_lineage: dict[str, str]) -> dict[str, str]:
        """``name -> source digest`` for callable inputs the kernel does not hold yet:
        simulated defs, and what a simulated ``from X import Y`` bound."""
        if not self._virtual_callables and not self._imported_classes:
            return {}
        user_ns = self.shell.user_ns
        found: dict[str, str] = {}
        for name in inputs:
            if name in user_ns:
                continue
            lineage = virtual_lineage.get(name) or self.tracking_state.variable_lineage.get(name) or ""
            key = virtual_callable_key(lineage, name)
            virtual = self._virtual_callables.get(key)
            if virtual is not None:
                found[name] = virtual.source_hash
            elif key in self._imported_classes:
                found[name] = self._imported_classes[key]
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
        did everything computed from it.

        A callee that is only a simulated def contributes its digest as the
        live function would (``_virtual_callable_hashes``).
        """
        function_tracker = self.function_tracker
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
        environment = statement_environment_component(stmt_code, user_ns)
        return {
            out: output_lineage(
                source_hash,
                input_lineages_all,
                file_hash_component,
                func_component,
                module_source_component(function_tracker, user_ns.get(out), out, stmt_code, tree),
                environment,
            )
            for out in outputs
        }

    def _collect_session_file_deps(self, outputs: set[str]) -> set[str]:
        """Return file dependencies from the current session for *outputs*."""
        file_deps: set[str] = set()
        executed_file_deps = self.tracking_state.executed_file_deps
        for out in outputs:
            file_deps.update(executed_file_deps.get(out, ()))
        return file_deps

    def _bound_modules(self, outputs: set[str], tree: ast.Module | None, stmt_code: str = "") -> set[str]:
        """The names an import statement binds to a MODULE.

        ``import x`` always binds one. ``from m import name`` usually binds a
        function or a constant, and the runtime's key builder decides by the
        value (``is_module_like``): a function goes in as an input with its
        source hash. Counting every imported name as a module gave
        ``df = clean(raw)`` a different key in the simulation, so the
        simulation never found that statement's entry.

        A name not bound yet is answered by what the statement bound when it
        last ran (``_import_bindings``), and without that record keeps the old
        answer: a module.
        """
        if tree is None:
            return set(outputs)
        from_bound = {
            alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        user_ns = self.shell.user_ns
        recorded = self._import_bindings(stmt_code) if stmt_code and (from_bound - set(user_ns)) else {}

        def is_module(out: str) -> bool:
            if out in user_ns:
                return isinstance(user_ns[out], types.ModuleType)
            if out in recorded:
                return bool(recorded[out].get("module"))
            return True

        return {out for out in outputs if out not in from_bound or is_module(out)}

    def _import_bindings(self, stmt_code: str) -> dict[str, dict]:
        """What *stmt_code* (a ``from`` import) bound when it last ran -- see
        ``import_bindings_key`` -- or ``{}``. Memoized; cleared with the caches."""
        memo = self._import_bindings_memo
        if stmt_code in memo:
            return memo[stmt_code]
        found: dict[str, dict] = {}
        backend = self.backend()
        if backend is not None:
            try:
                record = backend.get_metadata(import_bindings_key(stmt_code))
            except (OSError, TypeError, ValueError, AttributeError):
                record = None
            if record and record.get("import_bindings") and record.get("code") == stmt_code:
                found = dict(record.get("bindings") or {})
                found = {k: v for k, v in found.items() if isinstance(v, dict)}
                if record.get("magic") != importlib.util.MAGIC_NUMBER.hex():
                    for entry in found.values():
                        entry.pop("code", None)  # another interpreter's bytecode
        memo[stmt_code] = found
        return found

    def _propagate_import_lineage(
        self,
        outputs: set[str],
        virtual_modules: set[str],
        lineage_by_out: dict[str, str],
    ) -> None:
        """Propagate module lineages to ``self.tracking_state.variable_lineage`` for import statements.

        Called after computing the lineage hash for an import so that
        ``compute_cache_key`` can find the module in ``variable_lineage`` and
        include it in the module component — preventing cache key mismatches.
        """
        # Every name the import binds, not only modules: an import the runtime
        # SKIPPED leaves its names without a lineage otherwise, and a statement
        # reading one is then not cached ("input variable missing lineage").
        # A lineage put there for an EARLIER import of the name is replaced:
        # `import os, sys` in the cell that turns cash on (it runs uncached) and
        # `import sys` in the next. After a restart the second never runs --
        # `sys` is bound -- and `sys` kept the first's lineage while the session
        # before had keyed everything with the second's. Every helper reading
        # `sys.__stderr__` that ran again got a new lineage, and a 235 s
        # sweep missed.
        for out in outputs:
            if out not in lineage_by_out:
                continue
            held = self.tracking_state.variable_lineage.get(out)
            if held is None or held == self.propagated_imports.get(out):
                self.restores.record_restore(var_name=out, lineage_hash=lineage_by_out[out])
                self.propagated_imports[out] = lineage_by_out[out]
                logger.debug(
                    "[LINEAGE_DEBUG] Propagated module '%s' lineage to variable_lineage: %s...",
                    out,
                    lineage_by_out[out][:12],
                )
        # Mid-simulation drain: subsequent statements' compute_cache_key reads
        # variable_lineage to include module components, so the write must be
        # visible before the next _update_virtual_lineage call.
        apply_collected_mutations(self.restores, self.tracking_state)

    def _statement_reads_writes(
        self,
        stmt_code: str,
        tree: ast.Module | None,
        virtual_modules: set[str],
    ) -> tuple[StatementEffects, set[str], set[str]]:
        """*stmt_code*'s effects, and what it reads and writes, as its key sees them.

        The same analysis the runtime's ``_analyze_and_hash`` runs, with the
        notebook's cell text as the source of called functions: the two
        engines must agree on what a statement reads and writes, or they mint
        different keys. A bare method call (``lst.append(x)``) or a bare call
        to a helper that mutates its argument has no Store target, so the
        receivers the runtime treats as mutated are writes too. The globals a
        callee writes (``effects.callee_globals``) are left to the caller.
        """
        effects = statement_effects(
            stmt_code,
            tree,
            namespace=self.shell.user_ns,
            resolve_source=self._resolve_sim_function_source,
            control_body=is_control_body(stmt_code),
            virtual_modules=virtual_modules,
        )
        inputs, outputs = set(effects.inputs), set(effects.outputs)
        if tree is not None:
            outputs |= self._mutation_receivers(stmt_code, tree, virtual_modules)
            outputs |= effects.arg_mutations
        return effects, inputs, outputs

    def _update_virtual_lineage(
        self,
        stmt_code: str,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str] = None,
        occurrence_index: int = 0,
    ) -> tuple[set[str], float, bool, dict[str, float]]:
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

            mutation_tree = parse_cached(stmt_code)
            effects, inputs, outputs = self._statement_reads_writes(stmt_code, mutation_tree, virtual_modules)

            if mutation_tree is not None:
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
            is_import = stripped.startswith(("import ", "from "))
            if is_import:
                virtual_modules.update(self._bound_modules(outputs, mutation_tree, stmt_code))

            # RNG state is a hidden lineage variable (ADR-018): a draw reads it,
            # a seed produces it. Kept out of the plain ``inputs``.
            hidden_reads = key_hidden_reads(stmt_code, self.tracking_state)
            hidden_writes = hidden_lineage_writes(stmt_code)

            # A bare ``seed()`` carries no output, so it would return below before
            # recording its hidden variable. Compute its key (a seed is not a
            # draw, so no hidden read) and write the variable first.
            if hidden_writes and not outputs:
                seed_key, _, _, _, _ = compute_cache_key(
                    stmt_code,
                    inputs,
                    ctx=CacheKeyContext(
                        variable_lineage=self.tracking_state.variable_lineage,
                        user_ns=self.shell.user_ns,
                        function_tracker=self.function_tracker,
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

            # The globals a CALLEE writes join ``outputs`` so the
            # simulated lineage is bumped with the same source-based formula
            # the runtime uses. The runtime ALSO skip-caches such a statement;
            # that half is runtime-only, exactly like ``mut_pre_route``.
            outputs = outputs | effects.callee_globals

            if not outputs:
                return set(), 0.0, False, {}

            source_hash = statement_source_hash(stmt_code)

            key_lineage_inputs = inputs | hidden_reads

            input_lineages_all = self._resolve_virtual_input_lineages(
                stmt_code, inputs | lineage_hidden_reads(stmt_code), virtual_lineage, virtual_modules
            )

            # Compute cache key using the unified function
            cache_key, _, _, _, _ = compute_cache_key(
                stmt_code,
                key_lineage_inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self.tracking_state.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker,
                    virtual_lineage=virtual_lineage,
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
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

            if cache_result[0] == "hit":
                _, cache_lookup_time, stmt_file_deps, hit_bumped = cache_result
                # Union derivation-bumped vars so this cached mutation statement
                # is still recorded as a producer of the aliased base.
                outputs = outputs | hit_bumped
                self._register_virtual_callable(stmt_code, mutation_tree, virtual_lineage)
                if is_import:
                    self._register_imported_callables(mutation_tree, virtual_lineage, stmt_code)
                return outputs, cache_lookup_time, False, stmt_file_deps

            _, cache_lookup_time, files_stale, stmt_file_deps, extra_file_deps = cache_result
            file_deps_to_check.update(extra_file_deps)

            # Build file hash component
            file_hash_component = self._build_file_hash_component(file_deps_to_check, stmt_file_deps)
            own_reads = self.tracking_state.statement_file_reads.get(cache_key)
            if own_reads is not None:
                # The runtime hashed the files THIS statement read -- not the
                # ones its outputs inherited -- with compute_file_hash_component.
                # Same files, same function: an unchanged file gives the
                # runtime's lineage, a changed one a different lineage.
                file_hash_component = compute_file_hash_component(*own_reads)

            # Compute output lineage hashes
            lineage_by_out = self._compute_virtual_output_lineages(
                source_hash,
                input_lineages_all,
                file_hash_component,
                inputs,
                outputs,
                stmt_code,
                mutation_tree,
                virtual_lineage,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[LINEAGE_CALC] Statement: %s...", stmt_code[:40])
                logger.debug("[LINEAGE_CALC]   source_hash: %s...", source_hash[:16])
                logger.debug(
                    "[LINEAGE_CALC]   sorted(input_lineages_all): %s",
                    [h[:12] + "..." for h in sorted(input_lineages_all)],
                )
                logger.debug(
                    "[LINEAGE_CALC]   file_hash_component: %s...",
                    file_hash_component[:20] if file_hash_component else "(empty)",
                )
                logger.debug("[LINEAGE_CALC]   => lineages: %s", {v: h[:16] for v, h in lineage_by_out.items()})

            # Update virtual state
            virtual_lineage.update(lineage_by_out)
            self._register_virtual_callable(stmt_code, mutation_tree, virtual_lineage)
            if is_import:
                self._register_imported_callables(mutation_tree, virtual_lineage, stmt_code)

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
                self.tracking_state.derivation_edges,
                virtual_lineage,
                outputs,
                inputs,
                record=lambda t, h: virtual_lineage.__setitem__(t, h),
                present=lambda t: True,
            )
            outputs = outputs | bumped

            # CRITICAL: Propagate module lineages to self.tracking_state.variable_lineage immediately.
            # Without this, compute_cache_key won't find the module in variable_lineage
            # and will exclude it from module_component, causing key mismatches.
            if is_import:
                self._propagate_import_lineage(outputs, virtual_modules, lineage_by_out)

            return outputs, cache_lookup_time, files_stale, stmt_file_deps

        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.error("[UPSTREAM] Error simulating statement '%s...': %s", stmt_code[:20], e)
            raise

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
                        logger.debug(
                            "[UPSTREAM] Restore BLOCKED for '%s': cached value is empty "
                            "but in-memory has %d items, and its lineage is unconfirmed. "
                            "Keeping in-memory value.",
                            var,
                            len(existing),
                        )
                        continue
                except (TypeError, AttributeError):
                    pass
            self.shell.user_ns[var] = val
            restored_vars.add(var)
            if "output_lineages" in metadata:
                new_lineage = metadata["output_lineages"].get(var)
                if var in self.tracking_state.lineage and new_lineage is not None:
                    # Buffer a value-coupled restore so apply_collected_mutations
                    # routes through lineage.record, attaching _cash_lineage_hash
                    # to the live object. Drain immediately so the attribute is
                    # visible before _update_tracking_after_restore runs.
                    self.restores.record_restore(
                        var_name=var,
                        lineage_hash=new_lineage,
                        value=val,
                    )
                    apply_collected_mutations(self.restores, self.tracking_state)
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
        output_lineages = metadata.get("output_lineages", {}) if "output_lineages" in metadata else {}
        stored_code = metadata.get("code")
        stored_hash = metadata.get("source_hash")

        # Resolve file deps once.
        resolved_paths: set[str] = set()
        file_deps_meta = metadata.get("file_dependencies", {})
        if file_deps_meta:
            for stored_path in file_deps_meta:
                resolved = resolve_file_dep_path(stored_path)
                if resolved is not None:
                    resolved_paths.add(resolved)

        for var in restored_vars:
            lin = output_lineages.get(var) if output_lineages else None
            self.restores.record_restore(
                var_name=var,
                lineage_hash=lin,  # may be None — apply step skips lineage write if so
                code=stored_code if stored_code else None,
                code_hash=stored_hash if stored_hash else None,
                input_lineages=dict(input_hashes) if input_hashes else None,
                file_deps=set(resolved_paths) if resolved_paths else None,
            )

    def drop_probe_placeholders(self) -> None:
        """Unbind the names the forward probe held that no restore filled.

        The probe binds a placeholder for the cell it checked; its restore
        replaces it as the cell runs. One still bound afterwards (the restore
        failed) is not a value: left in place it would read as present to the
        next check and to the user. It goes, with the lineage recorded for it.
        """
        for var in self._probe_placeholders:
            if self.shell.user_ns.get(var) is _FORWARD_PROBE_PLACEHOLDER:
                del self.shell.user_ns[var]
                self.tracking_state.lineage.discard(var)
        self._probe_placeholders.clear()

    def eliminate_broken_vars_via_current_cell_probe(
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
        produce the broken variable — the cache restore will provide it.

        This avoids expensive upstream re-execution for scenarios like
        kernel restarts where heavy current-cell statements are on disk.
        """
        if not self.cash_instance or not broken_vars:
            return

        try:
            raw_cell = notebook_cells[current_cell_idx]
        except IndexError:
            return
        tree = parse_cell_source(raw_cell)
        if tree is None:
            return
        clean_cell = clean_cell_source(raw_cell)
        # Local: import cycle upstream.virtual_lineage -> ipython.cell_executor -> ... -> upstream.virtual_lineage.
        from ..ipython.cell_executor import CellExecutor

        # Track which broken vars are resolved by forward cache hits.
        # We simulate forward through the current cell's statements:
        # if a statement (a) would cache-hit and (b) its outputs overlap
        # with broken_vars, those outputs become available in memory.
        resolved_by_cache = set()
        # A broken var a statement that misses reads before any hit restores it
        # is needed from upstream: a later hit would come too late.
        needed_first: set[str] = set()
        occurrences: dict[str, int] = {}

        for node in tree.body:
            if is_control_structure(node):
                # Too complex to probe: what it reads is needed as it runs.
                try:
                    reads, _ = CodeAnalyzer.analyze_code_block(ast.unparse(node))
                except (SyntaxError, ValueError, TypeError):
                    reads = set(broken_vars)
                needed_first |= reads & broken_vars
                continue

            # The statement's key, built as ``_update_virtual_lineage`` builds
            # it (and so as the runtime does): its text with an expression's
            # trailing ``;``, its occurrence in the cell, its reads with the
            # hidden ones (an RNG a draw reads), and its writes with the
            # globals its callees write.
            try:
                stmt_code = ast.unparse(node)
            except (ValueError, TypeError):
                continue
            if CellExecutor.expr_has_trailing_semicolon(clean_cell, node):
                stmt_code += ";"
            occurrence_index = occurrences.get(stmt_code, 0)
            occurrences[stmt_code] = occurrence_index + 1

            effects, inputs, outputs = self._statement_reads_writes(stmt_code, parse_cached(stmt_code), virtual_modules)
            outputs |= effects.callee_globals
            unresolved = inputs & (broken_vars - resolved_by_cache - needed_first)
            if not unresolved:
                continue

            # Probe the cache's metadata only: nothing is restored here.
            try:
                cache_key, _, _, _, _ = compute_cache_key(
                    stmt_code,
                    inputs | key_hidden_reads(stmt_code, self.tracking_state),
                    ctx=CacheKeyContext(
                        variable_lineage=self.tracking_state.variable_lineage,
                        user_ns=self.shell.user_ns,
                        function_tracker=self.function_tracker,
                        virtual_lineage=virtual_lineage,
                        virtual_modules=virtual_modules,
                        compute_hash_fn=self.compute_hash_fn,
                        virtual_callables=self._virtual_callables,
                    ),
                    outputs=outputs,
                    occurrence_index=occurrence_index,
                )
                metadata = self._get_metadata_only(cache_key)
            except (KeyError, TypeError, ValueError, OSError):
                metadata = None
            # A metadata-only record (the value stayed in RAM, or was too large
            # to write) restores nothing; file deps must still be valid (mtime +
            # size, both forms -- see _validate_file_freshness for rationale).
            if (
                not metadata
                or metadata.get("metadata_only")
                or not self._validate_file_freshness(metadata.get("file_dependencies", {}), memo_key=cache_key)
            ):
                needed_first |= unresolved
                continue

            # Cache hit! Its restore puts back what its entry recorded,
            # including any broken vars among them.
            produced = outputs & unresolved & set(metadata.get("output_lineages") or ())
            if not produced:
                continue
            resolved_by_cache.update(produced)
            # The runtime keys the statement with these lineages, as the
            # simulation did: record them, so the restore finds the entry, and
            # hold each name's place until the restore fills it.
            for var in produced:
                if var in virtual_lineage:
                    self.restores.record_restore(var_name=var, lineage_hash=virtual_lineage[var])
                if var not in self.shell.user_ns:
                    self.shell.user_ns[var] = _FORWARD_PROBE_PLACEHOLDER
                    self._probe_placeholders.add(var)
            # Drain so subsequent statements probing the cache see the lineage.
            apply_collected_mutations(self.restores, self.tracking_state)
            logger.debug(
                "[UPSTREAM] Forward probe: cache hit for '%s' resolves broken vars: %s",
                stmt_code[:50],
                produced,
            )

        if resolved_by_cache:
            broken_vars -= resolved_by_cache
            logger.debug(
                "[UPSTREAM] Forward probe eliminated %d broken vars: %s. Remaining: %s",
                len(resolved_by_cache),
                resolved_by_cache,
                broken_vars,
            )

    def try_virtual_restore(
        self,
        stmt_code: str,
        outputs: set[str],
        inputs: set[str],
        input_hashes: dict[str, str],
        virtual_modules: set[str] | None = None,
        expected_lineages: dict[str, str] | None = None,
    ) -> tuple[set[str], float, float]:
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
                key_inputs(inputs, input_hashes),
                ctx=CacheKeyContext(
                    variable_lineage=self.tracking_state.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker,
                    virtual_lineage=key_lineages(input_hashes),
                    virtual_modules=virtual_modules,
                    compute_hash_fn=self.compute_hash_fn,
                    virtual_callables=self._virtual_callables,
                ),
                outputs=outputs,
            )

            logger.debug("[UPSTREAM] Attempting virtual restore Key: %s", cache_key)

            # 2. Query Memory Backend first (fastest) - Or just generic backend
            metadata, cached_data = self.cash_instance.backend.get(cache_key)
            if cached_data is not None:
                # Call results the entry refers to rather than copies (call_refs).
                cached_data = resolve_call_refs(cached_data, self.cash_instance.backend)

            # Extract saved execution time
            saved_time = metadata.get("execution_time", 0.0) if metadata else 0.0

            if metadata and cached_data is not None:
                # 3. Check file dependencies (Critical!)
                file_deps = metadata.get("file_dependencies", {})
                fresh, stale = snapshot_is_fresh(file_deps)
                if not fresh:
                    logger.debug("[UPSTREAM] Restore failed: stale file dependency (%s)", stale)
                    return set(), time_module.time() - start_time, 0.0

                conflict = lineage_conflict(metadata, file_deps, expected_lineages)
                if conflict is not None:
                    logger.debug("[UPSTREAM] Restore failed: lineage mismatch for %s", conflict)
                    return set(), time_module.time() - start_time, 0.0

                # 4. Success! Restore into shell.
                # Cache stores variables under 'variables' key (see _store_in_cache)
                variables_to_restore = cached_data.get("variables", {})
                restored_vars = self._restore_vars_from_cache(
                    variables_to_restore,
                    metadata,
                    lineage_confirmed_vars(metadata, file_deps, expected_lineages),
                )
                self._update_tracking_after_restore(restored_vars, metadata, input_hashes)
                return restored_vars, time_module.time() - start_time, saved_time

        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.debug("[UPSTREAM] Virtual restore error: %s", e)

        return set(), time_module.time() - start_time, 0.0

    def code_exists_in_notebook(self, mem_code: str, notebook_cells: list[str]) -> bool:
        """Return True if the normalized form of *mem_code* appears as a top-level
        statement in any notebook cell.

        Used when upstream modifications are detected to verify that an
        extension-validated variable's producing code has not been deleted or
        replaced.  On any parse/IO failure, returns False (conservative).
        """
        try:
            normalized_mem_code = normalize_stmt(mem_code)
            for cell_code in notebook_cells:
                if not clean_cell_source(cell_code).strip():
                    continue
                try:
                    cell_tree = parse_cell_source(cell_code)
                    if cell_tree is None:
                        continue
                    for node in cell_tree.body:
                        try:
                            node_code = normalize_stmt(ast.unparse(node))
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
            if vname not in self.tracking_state.variable_lineage:
                continue
            if virtual_lineage[vname] == self.tracking_state.variable_lineage[vname]:
                continue
            producing_code = self.tracking_state.executed_cell_codes.get(vname)
            if producing_code is None:
                continue
            normalized_prod = strip_markers(producing_code).strip()
            if normalized_prod in simulation_trace_codes:
                directly_mismatched.add(vname)
            elif vname in virtual_lineage:
                # Written last by this cell or one below it: ahead of the cells
                # above, not an edit to them. This cell counts: re-running one
                # that adds a column to a frame from above
                # (``docs['topic'] = ...``) found the frame ahead by its own
                # earlier write and rebuilt everything derived from it.
                is_downstream = any(
                    normalized_prod in notebook_cells[di] for di in range(current_cell_idx, len(notebook_cells))
                )
                if not is_downstream:
                    directly_mismatched.add(vname)
        return directly_mismatched

    def compute_tainted_vars_from_unsaved_edits(
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
        returned set — they are handled by the backward scan which has full
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
        for entry in simulation_trace:
            if entry.inputs & propagation_sources:
                vars_tainted.update(entry.outputs - directly_mismatched_vars)
                propagation_sources.update(entry.outputs)

        if logger.isEnabledFor(logging.DEBUG) and vars_tainted:
            logger.debug(
                "[UPSTREAM_DEBUG] Transitive mismatch propagation (unsaved edit): "
                "root mismatches = %s, tainted dependents = %s",
                directly_mismatched_vars,
                vars_tainted,
            )
        return vars_tainted

    def is_valid_extension(
        self, code: str, actual_lineage: str, virtual_lineage: dict[str, str], required_dependency: str | None = None
    ) -> bool:
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
                if self._unbound_builtin(inp, virtual_lineage):
                    continue

                if inp in virtual_lineage:
                    input_lineages.append(virtual_lineage[inp])
                elif inp in self.tracking_state.variable_lineage:
                    # Fallback to memory if virtual missing (external var not in notebook)
                    input_lineages.append(self.tracking_state.variable_lineage[inp])
                else:
                    # Input missing entirely. Cannot verify.
                    return False

            source_hash = statement_source_hash(code)
            # Route through the shared func-inclusive projection (matches the
            # recorder in statement/lineage.py) so an unsaved edit that calls a
            # user-defined function is not spuriously rejected for lacking the
            # function-source component. A hand-rolled sha256(code)+input_lineages
            # omitted it, so any function-routed edit always projected != recorded
            # and was wrongly discarded. [layer 1]
            projected = self._compute_virtual_output_lineages(source_hash, input_lineages, "", inputs, outputs, code)

            return actual_lineage in projected.values()

        except (KeyError, TypeError, ValueError, SyntaxError):
            return False

    @staticmethod
    def _iter_body_nodes(node: ast.AST):
        """Yield all body statements of a control structure (recursively)."""
        for attr in ("body", "orelse", "finalbody"):
            for child in getattr(node, attr, []) or []:
                yield child
                if is_control_structure(child):
                    yield from VirtualLineage._iter_body_nodes(child)
        # ast.Try handlers
        for handler in getattr(node, "handlers", []) or []:
            for child in handler.body:
                yield child
                if is_control_structure(child):
                    yield from VirtualLineage._iter_body_nodes(child)

    def _unbound_builtin(self, name: str, bound: Mapping[str, str] | None = None) -> bool:
        """Is *name* a builtin here: one of `BUILTIN_NAMES` that neither the
        kernel (``variable_lineage``) nor the simulation so far (*bound*) has
        bound? A user's ``max = ...`` or ``id = ...`` is an input like any other,
        as the runtime treats it."""
        return (
            name in BUILTIN_NAMES
            and name not in self.tracking_state.variable_lineage
            and (bound is None or name not in bound)
        )


def _first_cell_reading(notebook_cells: list[str], limit: int, names: set[str]) -> int | None:
    """Index of the first cell before *limit* that loads any of *names*, or
    binds one by a ``from ... import``.

    The binding cell too: kept from the cache, it carried the name's
    pre-reload lineage into every reader below, while a fresh kernel's import
    gives it the reloaded file's. A helper whose function reads the whole
    module (a clock read) was keyed apart in the session from the next
    morning, and nothing it built restored. Replaying
    the import is safe since imports are never restored (b4f2539).
    """
    if not names:
        return None
    for idx in range(min(limit, len(notebook_cells))):
        try:
            tree = parse_cell_source(notebook_cells[idx])
        except (ValueError, TypeError):
            tree = None
        if tree is None:
            return idx  # cannot tell: assume it reads them
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
                return idx
            if isinstance(node, ast.ImportFrom) and any((alias.asname or alias.name) in names for alias in node.names):
                return idx
    return None
