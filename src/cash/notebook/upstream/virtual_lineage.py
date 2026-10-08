"""Phase 1 of the notebook simulator: forward simulation of the cells above.

:meth:`VirtualLineage.simulate` replays the cells above the checked one into a
:class:`SimulationResult`, starting from the first cell changed since the
previous simulation (:class:`SimulationCache`). One statement's key and
lineages come from ``statement_lineage.py``, a control structure's from
``control_simulation.py``.
"""

from __future__ import annotations

import ast
import functools
import logging
import os
import re
import types
from collections import ChainMap
from typing import TYPE_CHECKING

from cash.control_markers import strip_markers

from ..._paths import resolve_file_dep_path
from ...analysis.ast_util import parse_cached
from ...analysis.cacheability import statement_writes_files
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source, statement_code
from ...source_norm import exact_source_digest
from .._protocols import ShellProtocol
from ..control_structures import is_control_structure
from ..lineage_formula import (
    key_hidden_reads,
    statement_environment_component,
)
from ..callee_reach import reached_user_code
from ..magic_effects import (
    is_magic_statement,
    is_rerun_magic,
    magic_base,
    magic_effects,
    magic_output_lineage,
    magic_rng_advances,
    simulation_cell,
)
from ..recorded_reads import outside_changes, watched_module_data
from ..run_memo import stats_this_run
from ..tracking_state import TrackingState
from ._types import (
    IncrementalStartResult,
    InputHashes,
    SimulationCache,
    SimulationCacheEntry,
    SimulationResult,
    TraceEntry,
)
from .control_simulation import ControlSimulation
from .simulated_callables import SimulatedCallables
from .statement_lineage import StatementLineage

if TYPE_CHECKING:
    pass

__all__ = ["VirtualLineage"]

logger = logging.getLogger(__name__)


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


#: ``reset_magic_deletes`` result for a reset that empties the user namespace.
RESET_ALL = re.compile("")


#: ``get_ipython().run_line_magic("name", "arg")`` on a line of its own.
_RUN_LINE_MAGIC = re.compile(
    r"""\s*get_ipython\(\)\.run_line_magic\(\s*(['"])(?P<magic>\w+)\1\s*,\s*(['"])(?P<arg>[^'"]*)\3\s*\)\s*$"""
)


def reset_magic_deletes(line: str) -> re.Pattern[str] | None:
    """Which user variables the IPython magic on *line* deletes.

    ``None`` when the line is not a reset that deletes variables, else a
    pattern the deleted names match (``RESET_ALL`` for all of them).

    - ``%reset`` with no target, or with ``-s``, empties the namespace.
    - ``%reset in|out|dhist`` flush only IPython's history caches.
    - ``%reset array`` deletes the names that hold numpy arrays, which the
      simulation cannot tell from the source. Treating it as a full wipe would
      make every other name fall back to its live lineage and hide an edit
      above, so it deletes nothing here.
    - ``%reset_selective regex`` deletes the names ``re.search`` matches.
    - ``%xdel name`` deletes *name*.

    A hand-written ``get_ipython().run_line_magic("xdel", "name")`` counts as
    the magic it runs.
    """
    call = _RUN_LINE_MAGIC.match(line)
    if call is not None:
        line = f"%{call['magic']} {call['arg']}"
    parts = line.split()
    if parts and parts[0] == "%xdel":
        names = [p for p in parts[1:] if not p.startswith("-")]
        return re.compile(rf"\A{re.escape(names[0])}\Z") if len(names) == 1 else None
    if not parts or parts[0] not in ("%reset", "%reset_selective"):
        return None
    flags = [p for p in parts[1:] if p.startswith("-")]
    args = [p for p in parts[1:] if not p.startswith("-")]
    if parts[0] == "%reset":
        soft = any(not f.startswith("--") and "s" in f for f in flags)
        return RESET_ALL if soft or not args else None
    if not args:
        return None
    try:
        return re.compile(" ".join(args))
    except re.error:
        return None


class VirtualLineage:
    """Phase 1 of NotebookSimulator: forward simulation of the cells above.

    Walks each cell's statements into a :class:`SimulationResult`, handing a
    simple statement to :class:`StatementLineage` and a control structure to
    :class:`ControlSimulation`, and keeps the per-cell snapshots
    (:class:`SimulationCache`) the next simulation starts from.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        *,
        statements: StatementLineage,
        controls: ControlSimulation,
        callables: SimulatedCallables,
        cache: SimulationCache,
    ) -> None:
        self.shell = shell
        #: The checker's, shared with the statement processor.
        self.tracking_state = tracking_state
        #: The key and output lineages of one statement.
        self.statements = statements
        #: Control structures, each simulated as one unit.
        self.controls = controls
        #: Defs and imported callables the kernel does not hold yet.
        self.callables = callables
        #: The previous simulation's per-cell snapshots, where the next one starts.
        self.cache = cache
        #: ``TrackingState.module_generation`` the last pass 1 saw.
        self._simulated_module_generation = 0

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

        One stat per file per cell run (``run_memo.stats_this_run``)."""
        current = stats_this_run(cached_file_deps)
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

    def _reaches_watched_module_data(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None,
        cell_code: str | None,
    ) -> bool:
        """Whether the cell about to run can be reached by watched module
        data: it reads some itself, or a statement it derives an input from
        does (the cached trace, read backwards from *required_inputs*, as
        ``read_scope`` scopes file reads). Unknown inputs or cell: yes.

        Checking that data for changes made outside the notebook hashes all
        of it, before every cell: once ``v = helpers.score(3)`` had read a
        128 MB table, ``x = 1`` cost 350 ms. A cell no reader reaches cannot
        be told anything by it, and leaves it to the next cell that can.
        """
        labels = watched_module_data(self.tracking_state.reads)
        if not labels:
            return False
        if required_inputs is None:
            return True
        if cell_code is None and current_cell_idx < len(notebook_cells):
            cell_code = notebook_cells[current_cell_idx]
        if cell_code is None:
            return True
        namespace = self.shell.user_ns

        def reads(code: str) -> bool:
            # A magic (``%run``, ``%aimport``) can read anything: yes.
            if "get_ipython()" in code:
                return True
            return any(label in labels for label, _value in reached_user_code(code, namespace).data)

        cell_code = cell_code.replace("\r\n", "\n")
        if parse_cached(cell_code) is None or reads(clean_cell_source(cell_code)):
            return True
        needed = set(required_inputs)
        for cached in reversed(self.cache.entries[: min(current_cell_idx, len(self.cache.entries))]):
            for entry in reversed(cached.trace_segment):
                if entry.outputs & needed:
                    if reads(entry.stmt_code):
                        return True
                    needed |= entry.inputs
        return False

    def _scan_main_cache_for_changes(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        cell_code: str | None = None,
    ) -> tuple[int, bool]:
        """Scan the main simulation cache to find the first changed cell.

        Returns ``(first_changed_cell, cache_had_hash_mismatch)``.
        """
        first_changed_cell = 0
        cache_had_hash_mismatch = False
        def in_notebook(code: str) -> bool:
            self.statements.set_notebook_functions(notebook_cells)
            return self.statements.in_notebook(code)

        module_data = self._reaches_watched_module_data(current_cell_idx, notebook_cells, required_inputs, cell_code)
        if outside_changes(self.tracking_state.reads, in_notebook, module_data=module_data):
            # The environment or a module's data, read by a statement, was
            # changed outside the notebook's cells: no cell's code says so, so
            # simulate them all again, as for a changed file.
            return first_changed_cell, cache_had_hash_mismatch
        for idx in range(min(current_cell_idx, len(self.cache.entries))):
            cell_code = notebook_cells[idx].replace("\r\n", "\n")
            cell_hash = exact_source_digest(cell_code)
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
            if cached.magic_generation not in (None, self.tracking_state.magic_generation):
                # A magic has run since: what the cell's magics left changed.
                break
            if cached.stopped_at != self._stop_index(cell_code):
                # The cell has run (or failed) since: what of it ran changed,
                # not its code.
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
            cell_hash = exact_source_digest(cell_code)
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
        required_inputs: set[str] | None = None,
        cell_code: str | None = None,
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
                current_cell_idx, notebook_cells, required_inputs, cell_code
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

    def simulate(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        cell_code: str | None = None,
    ) -> SimulationResult:
        """Pass 1: simulate every cell above *current_cell_idx*, starting from
        the first one changed since the previous simulation. *required_inputs*
        and *cell_code* are the cell's (None: not known), see
        `_reaches_watched_module_data`."""
        start = self.find_incremental_start(current_cell_idx, notebook_cells, required_inputs, cell_code)
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

    def build_simulation_trace_codes(self, simulation_trace: list) -> set[str]:
        """Return the set of normalised statement codes present in *simulation_trace*.

        Includes body-level statements from control structures so that
        per-iteration cache entries (which record body statements rather than
        the whole for-loop) are matched correctly.
        """
        simulation_trace_codes: set[str] = set()
        for entry in simulation_trace:
            simulation_trace_codes.update(_trace_codes(entry.stmt_code))
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

    def _simulate_magic(self, sim: SimulationResult, node: ast.stmt, stmt_code: str) -> None:
        """Give each name the magic statement *node* binds or changes the
        lineage its last run left (``StatementProcessor.record_magic``).

        From the lineages it reads here and the digest of the value it left
        when it last ran with those: when what it reads differs from that
        run, no digest is found, and the lineage differs from the live one.
        A ``%time``/``%timeit``/``%prun`` line (``is_rerun_magic``) leaves a
        trace entry, so a rebuild runs it again; any other magic is never
        rebuilt and leaves none.
        """
        virtual_lineage = sim.virtual_lineage
        live = self.tracking_state.variable_lineage
        changed, read = magic_effects(node, sim.virtual_modules.__contains__)
        reads = {name: virtual_lineage.get(name, live.get(name)) for name in read}
        base = magic_base(stmt_code, reads)
        digests = self.tracking_state.magic_values.get(base, {})
        rerun = is_rerun_magic(node)
        for name in changed:
            lineage = magic_output_lineage(base, digests.get(name, "not run"))
            if not rerun:
                self.tracking_state.magic_lineages.add(lineage)
            virtual_lineage[name] = lineage
        virtual_lineage.update(magic_rng_advances(node, stmt_code, ChainMap(virtual_lineage, live)))
        if rerun and changed:
            # Python a rebuild runs again, as any statement: the planner
            # schedules it where what it binds or changes is needed.
            produced = {name: virtual_lineage[name] for name in changed}
            input_hashes = {name: lineage for name, lineage in reads.items() if lineage is not None}
            sim.trace.append(TraceEntry(stmt_code, set(changed), set(read), input_hashes, produced, False))

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
        (``ControlSimulation.simulate``).

        *raw_cell* is the text *node* was parsed from. The runtime keys an
        expression followed by ``;`` WITH the ``;`` (IPython's display
        suppression), which ``ast.unparse`` drops -- so ``ax.bar(...);
        ax.set_xlabel(...)`` on one line got another key and lineage here, and
        every chart drawn that way disagreed.
        """
        try:
            if is_control_structure(node):
                self.controls.simulate(node, sim)
                return

            stmt_code = statement_code(node, raw_cell)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("[UPSTREAM] Error processing node in cell %d: %s", i, e)
            raise

        if is_magic_statement(node):
            self._simulate_magic(sim, node, stmt_code)
            return

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
        callee_lineages = self.callables.callee_lineages(inputs, virtual_lineage, virtual_modules)
        hidden_lineages = {
            var: virtual_lineage.get(var, self.tracking_state.variable_lineage.get(var))
            for var in key_hidden_reads(stmt_code, self.tracking_state)
        }
        if callee_lineages or hidden_lineages:
            input_hashes = InputHashes(input_hashes, callee_lineages, hidden_lineages)

        outputs, lookup_time, files_stale, stmt_file_deps = self.statements.apply(
            stmt_code,
            virtual_lineage,
            virtual_modules,
            occurrence_index=occurrence_index,
        )

        if stmt_file_deps:
            cell_file_deps.update(stmt_file_deps)

        self._update_stale_file_deps(inputs, outputs, files_stale, sim.vars_with_stale_files)

        if outputs:
            # A statement that sets state on a local module changes it in
            # place, so it reads it, as `items.append(x)` reads `items`, even
            # where its text does not name it (`set_k(5)` imported from it):
            # a rebuild runs the import before it.
            inputs = set(inputs) | (outputs & self.statements.module_state_outputs(stmt_code))

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
            # (``carrier_fills.fills_carrier``). The runtime's recorded mutation verdict
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
        cell_hash = exact_source_digest(cell_code)
        stopped_at = self.tracking_state.failed_cells.get(cell_hash)
        simulation_trace = sim.trace
        virtual_lineage = sim.virtual_lineage
        virtual_modules = sim.virtual_modules
        trace_start = len(simulation_trace)
        cell_file_deps: dict = {}

        # Model the ``%reset`` and ``%xdel`` magics BEFORE the strip_magics empty-cell
        # short-circuit below (a reset cell strips to empty). Like ``del`` they
        # clear ``user_ns`` but not ``variable_lineage``; position-scoping (the
        # simulator only replays cells 0..current) means a reset ABOVE the
        # target drops the names it deletes from the virtual state, so the
        # liveness gate does not resurrect them as phantom restores, while a
        # reset BELOW is never simulated.
        for line in cell_code.split("\n"):
            dropped = reset_magic_deletes(line)
            if dropped is None:
                continue
            if dropped is RESET_ALL:
                virtual_lineage.clear()
                virtual_modules.clear()
                continue
            for name in [n for n in virtual_lineage if dropped.search(n)]:
                del virtual_lineage[name]
            virtual_modules.difference_update({m for m in virtual_modules if dropped.search(m)})

        # A cell holding magics is read as the runtime runs it, magic lines
        # included (``simulation_cell``): what they bind is theirs.
        ipython = simulation_cell(cell_code)
        magic_generation = self.tracking_state.magic_generation if ipython is not None else None
        try:
            clean_cell_code = clean_cell_source(cell_code)
            if ipython is None and not clean_cell_code.strip():
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

            if ipython is not None:
                clean_cell_code, tree = ipython
            else:
                tree = parse_cell_source(cell_code)
            if tree is None:
                ast.parse(clean_cell_code)  # will raise SyntaxError

            cell_stmt_occurrence_counts: dict = {}

            for index, node in enumerate(tree.body):
                # The statement this cell's last run raised at, and every one
                # after it, never ran: the kernel holds what came before.
                if index == stopped_at:
                    break
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
            # ``NotebookVetter._warn_broken_upstream_cells``; here we only keep
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
                stopped_at=stopped_at,
                magic_generation=magic_generation,
            )
        )

    def _stop_index(self, cell_code: str) -> int | None:
        """Where the simulation of *cell_code* stops (``TrackingState.failed_cells``)."""
        return self.tracking_state.failed_cells.get(exact_source_digest(cell_code))

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
        self.statements.set_notebook_functions(notebook_cells)
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
            self.cache.cell_hashes[current_cell_idx] = exact_source_digest(current_cell_code)

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


@functools.lru_cache(maxsize=8192)
def _trace_codes(stmt_code: str) -> tuple[str, ...]:
    """The normalised codes one trace statement stands for: itself, and the
    body statements of a control structure. The text decides them, and every
    cell asks for every statement above it: parsing and unparsing them again
    was 0.25 s of the last 20 cells of a 400-cell notebook."""
    normalized = strip_markers(stmt_code).strip()
    codes = [normalized]
    try:
        tree = parse_cached(normalized)
        if tree and len(tree.body) == 1 and is_control_structure(tree.body[0]):
            for body_node in VirtualLineage._iter_body_nodes(tree.body[0]):
                try:
                    codes.append(ast.unparse(body_node).strip())
                except (ValueError, TypeError):
                    logger.debug("[UPSTREAM] Failed to unparse body node in simulation trace")
    except (SyntaxError, ValueError):
        logger.debug("[UPSTREAM] Failed to parse control structure for simulation trace codes")
    return tuple(codes)


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
