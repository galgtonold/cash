from __future__ import annotations

import ast
import functools
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from ..._memo import NOTEBOOK_CELLS, NOTEBOOK_VERSIONS
from ...analysis.analysis_budget import AnalysisBudget, analysis_budget, analysis_budget_seconds
from ...analysis.ast_util import parse_cached, walk_tree
from ...analysis.code_analyzer import CodeAnalyzer, parse_cell_source
from ...analysis.mutation_effects import CellEffects, NotebookSources, cell_effects
from ...control_markers import strip_markers
from ...diagnostics import warn_diagnostic
from ...exceptions import AmbiguousCellError, CashWarning, UpstreamStateError
from ...value_types import BUILTIN_NAMES
from .._protocols import CashInstanceProtocol, ShellProtocol
from .._trace import trace_event
from ..cache_key import called_function_globals, statement_source_hash
from ..server_discovery import (
    get_notebook_cells,
    get_notebook_cells_with_ids,
    get_notebook_path,
    invalidate_notebook_path_cache,
    warn_no_live_reader_once,
    warn_notebook_not_found_once,
)
from ..shared_objects import hold_repair_snapshot, release_repair_snapshot
from ..staleness import StalenessTracker, normalise_source
from ..statement.carrier_advances import reachable_generators
from ..tracking_state import TrackingState
from .changed_below import warn_changed_below
from .notebook_vetting import NotebookVetter
from .replay import StatementReplay, UpstreamRepairGap, statement_cells
from .rng_rewind import RngRewind
from .simulator import NotebookSimulator

if TYPE_CHECKING:
    from ...tracking.function_tracker import FunctionTracker
    from ..statement import ProcessResult

__all__ = ["UpstreamChecker", "UpstreamResult"]


class UpstreamResult(NamedTuple):
    """Result of upstream checking and re-execution."""

    metrics: list[ProcessResult]
    restore_time: float
    execution_time: float


logger = logging.getLogger(__name__)

# A name the namespace did not hold (``None`` is a value).
_UNBOUND = object()


@functools.lru_cache(maxsize=NOTEBOOK_CELLS)
def _statement_texts(cell_code: str) -> frozenset[str]:
    """Every statement of *cell_code*, at any depth, as cash records the code
    that produced a name (``TrackingState.executed_cell_codes``)."""
    tree = parse_cell_source(cell_code)
    if tree is None:
        return frozenset()
    return frozenset(ast.unparse(node).strip() for node in walk_tree(tree) if isinstance(node, ast.stmt))


@functools.lru_cache(maxsize=NOTEBOOK_VERSIONS)
def _notebook_statement_texts(notebook_cells: tuple[str, ...]) -> frozenset[str]:
    """Every statement of the notebook: asked on every cell run while an
    unsaved run is on record, so once per notebook version."""
    return frozenset().union(*map(_statement_texts, notebook_cells))


def _mtime(path: str | None) -> float | None:
    """When the notebook file was last written; None when unknown."""
    if not path:
        return None
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


class UpstreamChecker:
    """Brings the state a cell reads up to date with the notebook above it.

    Before a cell runs, :meth:`check_and_reexecute` finds the cell in the
    saved notebook, vets the notebook (:class:`NotebookVetter`), simulates
    the cells above it (:class:`NotebookSimulator`) and re-runs or restores
    what the simulation found stale (:class:`StatementReplay`), with the
    random stream put where a top-to-bottom run leaves it
    (:class:`RngRewind`). Every part shares one ``TrackingState``.
    """

    def _cell_analysis_budget(self, cell_code: str) -> AnalysisBudget:
        """The analysis budget of *cell_code*'s cell-wide reading
        (`analysis_budget_seconds`): from what its statements last took
        together (`expected_run_time`); the floor when none was measured."""
        total = 0.0
        try:
            tree = parse_cached(cell_code)
            for node in tree.body:
                segment = ast.get_source_segment(cell_code, node)
                took = self.expected_run_time(segment.strip()) if segment else None
                total += took or 0.0
        except Exception:  # noqa: BLE001 - unmeasured: the floor applies
            total = 0.0
        return AnalysisBudget(analysis_budget_seconds(total or None))

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol | None = None,
        compute_hash_fn: Callable[[Any], str] | None = None,
        tracking_state: TrackingState | None = None,
        function_tracker: FunctionTracker | None = None,
    ) -> None:
        self.shell: ShellProtocol = shell
        self.cash_instance: CashInstanceProtocol | None = cash_instance
        self.compute_hash_fn: Callable[[Any], str] | None = compute_hash_fn
        #: The index of the cell checked last, for a run whose own index is unknown.
        self.last_cell_index: int | None = None
        #: The source of the cell checked last and its index in the notebook,
        #: when it was found: what a warning about that cell calls it.
        self.last_found_cell: tuple[str, int] | None = None
        #: Whether the cell being checked does not come right after the one checked before it.
        self._reran = False
        #: What a statement's text last took to run, or None (set by the
        #: magics, as the statement processor's): the cell's analysis budget.
        self.expected_run_time: Callable[[str], float | None] = lambda code: None
        #: Told the cells above the cell being checked, before the simulation
        #: (set by the magics, as the statement processor's): a global stream
        #: none of them seeds is unseeded for the cell, as after a restart.
        self.streams_above: Callable[[list[str]], None] = lambda cells_above: None

        # Proven-stale verdict for the saved .ipynb, held for the session.
        # Populated at the cell-ID match site below, where both the running
        # code and the file's copy of that cell are already in hand.
        self.staleness = StalenessTracker()
        self._notebook_path_for_staleness: str | None = None
        #: Names the cell's inputs gained from what the functions it calls
        #: read (`_callee_globals`), for this cell check.
        self._callee_inputs: set[str] = set()
        #: Cell id -> (the file's source, the source that ran) for a cell cash
        #: ran with an edit the saved notebook does not have yet. The kernel
        #: holds what ran, so the cells above a later cell are read as run
        #: until the file changes (``_as_run``).
        self._ran_unsaved: dict[str, tuple[str, str]] = {}
        #: The statements of each cell cash ran that the saved notebook does
        #: not hold and no cell id places, with the file's mtime then
        #: (``_unsaved_bindings``).
        self._unplaced_runs: list[tuple[frozenset[str], float | None]] = []
        #: The cells (as ``normalise_source`` reads them) already warned
        #: NOTEBOOK-CELL-UNPLACED for, so a cell says it once per text.
        self._warned_unplaced: set[str] = set()
        #: Whether the last check found the cell among the notebook's and
        #: simulated the cells above it.
        self._simulated = False

        #: Shared with the statement processor and the simulator: every
        #: tracking dict is read and written through it.
        self.tracking_state = tracking_state or TrackingState()

        # The simulation runs no user code: it replays the notebook from its
        # text and the cache, and this class runs what it schedules.
        self.simulator = NotebookSimulator(
            shell=shell,
            cash_instance=cash_instance,
            tracking_state=self.tracking_state,
            compute_hash_fn=compute_hash_fn,
            function_tracker=function_tracker,
        )
        self.vetter = NotebookVetter(shell, self.tracking_state)
        self.rng = RngRewind(shell, self.tracking_state)
        self.replay = StatementReplay(self.tracking_state)

    @property
    def function_tracker(self) -> FunctionTracker | None:
        """The runtime's function tracker the simulation keys with."""
        return self.simulator.function_tracker

    def plan_cell_run(
        self,
        nodes: list,
        raw_cell: str,
        occurrence_counts: dict[str, int],
    ) -> dict[int, dict] | None:
        """Which of a run of assignments in the cell being run need not run --
        see :meth:`NotebookSimulator.plan_cell_run`."""
        return self.simulator.plan_cell_run(nodes, raw_cell, occurrence_counts)

    def cell_touches_rng(self, src: str) -> bool:
        """True if *src* seeds or draws -- see :meth:`RngRewind.cell_touches_rng`."""
        return self.rng.cell_touches_rng(src)

    def reset_caches(self) -> None:
        """Forget the previous simulation.

        Should be called when switching notebooks (e.g., on %cash_on)
        to prevent stale simulation data from a previous notebook
        from interfering with the current one.
        """
        self.simulator.reset_caches()
        # Re-arm the broken-upstream-cell warning for the new notebook:
        # its cell indices/hashes are meaningless across a notebook switch.
        self.vetter.forget_warnings()
        self._ran_unsaved.clear()
        self._unplaced_runs.clear()
        self._warned_unplaced.clear()
        # A staleness verdict is proof about notebook A's file; carrying it
        # into notebook B (or a fresh %cash_on on the same one) would show a
        # warning about a file this session no longer even reads from, until
        # the first ID-matched run in the new notebook happens to reset it.
        self.staleness.reset()

    def _find_current_cell_index(
        self,
        cell_code: str,
        notebook_cells: list[str],
        cell_id: str | None = None,
        cells_with_ids: list[tuple[str, str]] = None,
    ) -> int | None:
        """Find the index of the current cell in the notebook.

        Uses a chain of matching strategies: ID match → exact content →
        normalized newlines → stripped whitespace → the text IPython runs.
        Returns the first unambiguous match, or raises ``AmbiguousCellError``
        when multiple cells share the same content and no cell ID is available.

        The last step compares the cells as ``normalise_source`` reads them:
        IPython empties every whitespace-only line before it runs a cell,
        while the saved file keeps the indentation an editor leaves on a
        blank line in a block. Both texts run as the same code, so this
        finds the cell without guessing between texts that differ.
        """
        # Strategy 1: Exact cell-ID match (available since IPython 8.3)
        if cell_id and cells_with_ids:
            for i, (nb_cell_id, nb_cell_src) in enumerate(cells_with_ids):
                if nb_cell_id == cell_id:
                    # Both strings are here: what IPython is running, and what
                    # the FILE thinks this same cell says. If they differ the
                    # file is out of date -- and so is every other cell we read
                    # from it. Detection only; the match result is unchanged.
                    self.staleness.observe(
                        running_code=cell_code,
                        file_code=nb_cell_src,
                        notebook_path=self._notebook_path_for_staleness,
                    )
                    logger.debug("[UPSTREAM_DEBUG] Found cell by ID match at index %s", i)
                    return i

            logger.debug("[UPSTREAM_DEBUG] Cell ID %s not found in notebook, falling back to content match", cell_id)

        # Strategies 2-4: content matching with progressive normalization
        content_matchers = [
            lambda cell: cell == cell_code,
            lambda cell: cell.replace("\r\n", "\n") == cell_code.replace("\r\n", "\n"),
            lambda cell: cell.strip() == cell_code.strip(),
            lambda cell, ran=normalise_source(cell_code): normalise_source(cell) == ran,
        ]
        for matcher in content_matchers:
            matches = [i for i, cell in enumerate(notebook_cells) if matcher(cell)]
            if matches:
                break
        else:
            return None

        if len(matches) == 1:
            return matches[0]

        # Multiple matches with no resolvable cell ID — ambiguous
        logger.debug(
            "[UPSTREAM_DEBUG] Ambiguous cell content (matches=%s). Unable to safely determine upstream context.",
            matches,
        )

        where = ", ".join(str(i + 1) for i in matches)
        raise AmbiguousCellError(
            f"Ambiguous cell execution: this cell's text is also in another cell (code cells {where}), "
            "and the editor sent no cell id, so cash cannot tell which one is running. Which one "
            "decides what the cell must show (the state at that point of a run from the top), so "
            "cash does not guess. Fix: make the cells differ (a comment is enough), or update "
            "IPython to 8.3 or newer and save the notebook, so each cell carries its id."
        )

    def _callee_globals(self, inputs: set[str]) -> set[str]:
        """The notebook names the functions in *inputs* read when called
        (`called_function_globals`), among those a cell bound."""
        lineage = self.tracking_state.variable_lineage
        try:
            names = called_function_globals(inputs, self.shell.user_ns, variable_lineage=lineage)
        except Exception:  # advisory widening: a walk fault keeps the cell's own inputs
            logger.debug("[UPSTREAM_DEBUG] callee globals walk failed", exc_info=True)
            return set()
        return {name for name in names if name in lineage}

    def check_and_reexecute(
        self,
        cell_code: str,
        required_inputs: set[str],
        process_statement_callback: Callable[..., ProcessResult],
        global_ttl: int | None = None,
        cell_id: str | None = None,
        progress_callback: Callable[..., None] | None = None,
        control_structure_callback: Callable[..., Any] | None = None,
    ) -> UpstreamResult:
        """
        Check if any upstream statements have changed and re-execute them if needed.

        Args:
            cell_code: The code content of the current cell
            required_inputs: Set of variable names this cell requires as inputs
            process_statement_callback: Callback to process/execute statements
            global_ttl: Optional time-to-live for cache entries
            cell_id: Optional Jupyter cell ID (available since IPython 8.3)
                     Used to correctly identify duplicate cells
            progress_callback: Optional callback(metrics_so_far, current_stmt_code)
                     Called after each upstream statement for progress reporting
            control_structure_callback: Optional callback(ast_node, ttl, silent)
                     for executing control structures with per-iteration caching.
                     When provided, for-loops and other control structures are
                     delegated to this callback instead of process_statement_callback.

        Returns:
            Tuple of (upstream_metrics, total_restore_time, total_execution_time)
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[UPSTREAM_DEBUG] check_and_reexecute called")
            logger.debug("[UPSTREAM_DEBUG]   cell_code: %s...", cell_code[:50])
            logger.debug("[UPSTREAM_DEBUG]   required_inputs: %s", required_inputs)
            logger.debug(
                "[UPSTREAM_DEBUG]   current variable_lineage keys: %s",
                list(self.tracking_state.variable_lineage.keys()),
            )
            if cell_id:
                logger.debug("[UPSTREAM_DEBUG]   cell_id: %s", cell_id)

        # The cell's statements are about to run: a memo of a cell holding
        # one simulated it on the records they leave behind.
        self.simulator.virtual_lineage.forget_cells_running(cell_code)
        # A name the previous cell's forward probe held and no restore filled.
        self.simulator.restorer.drop_probe_placeholders()
        self.tracking_state.probe_held.clear()
        # Unplaced until the check has found the cell and rebuilt what is above it.
        self.tracking_state.running_cell = None
        self.tracking_state.running_cell_rerun = False
        self._reran = False

        # Resolve the notebook path ONCE for the whole cell check and
        # thread it through the analysis helpers + Phase 2, instead of each site
        # re-running discovery. The negative cache in server_discovery bounds a
        # failed probe, and this collapses the success case to a single resolve.
        notebook_path = self._resolve_notebook_path()
        # The ID-match site below needs this to stat the file. Reuse the single
        # resolve rather than probing again -- discovery is deliberately done
        # once per cell check.
        self._notebook_path_for_staleness = notebook_path

        # What the cell writes, by the channel an isolated re-run resets it
        # through. A global it changes without naming it joins its inputs, so
        # the reset below restores that global's producer too.
        with analysis_budget(self._cell_analysis_budget(cell_code)):
            effects = cell_effects(cell_code, self._notebook_sources(cell_code, notebook_path), self.shell.user_ns)
        required_inputs = required_inputs | effects.hidden_inputs
        # A generator a called function draws from (`def boot(x): rng...`)
        # moves without being named, as a generator named in the cell does: an
        # isolated re-run first puts it where a top-to-bottom run has it.
        required_inputs = required_inputs | reachable_generators(
            required_inputs, self.shell.user_ns, self.tracking_state.variable_lineage
        )
        # What the functions the cell calls read when called, transitively:
        # ``def a(n): return b(n) * 2`` above ``def b`` named no ``b`` the
        # cell reads, so an edit of ``b``'s cell was never checked and the
        # old ``b`` answered.
        # Not for the forward-reference vetting: a body may name what a cell
        # below binds, as Python resolves it only when called.
        self._callee_inputs = self._callee_globals(required_inputs) - required_inputs
        required_inputs = required_inputs | self._callee_inputs
        # A reloaded module whose state cells set is rebuilt now, whatever the
        # cell reads (``ModuleInvalidator._schedule_state_rebuild``).
        rebuilds = {name for name in self.tracking_state.module_state_rebuilds if name in self.shell.user_ns}
        self.tracking_state.module_state_rebuilds.clear()
        required_inputs = required_inputs | rebuilds

        # Simulate the notebook statement by statement and compare the virtual
        # lineage with the in-memory state to find changed code.
        self._simulated = False
        all_metrics, total_restore_time, total_execution_time = self._bring_up_to_date(
            cell_code,
            required_inputs,
            process_statement_callback,
            global_ttl,
            effects,
            notebook_path=notebook_path,
            progress_callback=progress_callback,
            control_structure_callback=control_structure_callback,
            cell_id=cell_id,
        )
        if rebuilds and not self._simulated:
            _warn_state_not_rebuilt(rebuilds, self.shell.user_ns)
        if self.tracking_state.unheld_module_state:
            _warn_unheld_state(self.tracking_state)

        return UpstreamResult(all_metrics, total_restore_time, total_execution_time)

    def _resolve_notebook_path(self) -> str | None:
        """Resolve the current notebook path once per cell's upstream check.

        Single choke point for path discovery: the per-statement /
        per-upstream-cell analysis helpers used to each call ``get_notebook_path``
        independently, so one cell re-probed discovery 5-15 times — catastrophic
        when a stale Jupyter runtime makes each probe block on a network timeout.
        Resolving once and threading the result collapses that to a single probe.

        Also emits the once-per-session "upstream tracking disabled" advisory when
        discovery fails, so a user knows cash's headline feature is off rather
        than inferring it from silently-stale results.
        """

        path = get_notebook_path()
        if path is None:
            warn_notebook_not_found_once()
        return path

    def _notebook_cells_for(self, notebook_path: str | None) -> list[str]:
        """Read the notebook's on-disk cell sources for a pre-resolved path.

        ``get_notebook_cells`` is memoized by the file's (mtime, size), so the
        several analysis helpers that each need the cells share a single parse.
        Returns ``[]`` when no path resolved or the read fails.
        """
        if not notebook_path:
            return []
        try:
            return get_notebook_cells(notebook_path) or []
        except (OSError, ValueError, RuntimeError):
            return []

    def _notebook_sources(self, cell_code: str, notebook_path: str | None) -> NotebookSources:
        """Top-level definitions across the notebook's cells plus *cell_code*,
        read from their text; the notebook is only read if one is asked for."""
        return NotebookSources(lambda: self._notebook_cells_for(notebook_path), cell_code)

    def _resolve_fallback_cache_idx(self, cell_id: str | None) -> int | None:
        """Return the simulation cache index to use for the downstream advancement fallback.

        Returns ``None`` when no suitable cache entry can be found.
        """
        known_cell_idx: int | None = None
        if cell_id is not None:
            known_cell_idx = self.simulator.cache.last_index_by_cell_id.get(cell_id)
        if known_cell_idx is None and self.last_cell_index is not None:
            known_cell_idx = self.last_cell_index

        if known_cell_idx is not None and known_cell_idx > 0:
            target_idx = known_cell_idx - 1
            if target_idx < len(self.simulator.cache):
                return target_idx
        return None

    def _reset_advanced_lineages(
        self,
        overlap_vars: set[str],
        cached_virtual_lineage: dict[str, str],
        cache_idx: int,
    ) -> None:
        """Reset in-memory lineages that are "ahead" of the cached virtual lineage."""
        logger.debug(
            "[UPSTREAM_DEBUG]   Downstream advancement fallback: overlap_vars=%s, cache_idx=%s, last_cell_index=%s",
            overlap_vars,
            cache_idx,
            self.last_cell_index,
        )
        for var_name in overlap_vars:
            if var_name not in cached_virtual_lineage or var_name not in self.tracking_state.variable_lineage:
                continue
            virtual_hash = cached_virtual_lineage[var_name]
            actual_hash = self.tracking_state.variable_lineage[var_name]
            if actual_hash != virtual_hash:
                logger.debug(
                    "[UPSTREAM_DEBUG]   -> Downstream advancement fallback: "
                    "resetting '%s' lineage from %s to virtual %s",
                    var_name,
                    actual_hash[:8],
                    virtual_hash[:8],
                )
                self.tracking_state.lineage.reset_to(var_name, virtual_hash)

    def _handle_downstream_advancement_fallback(
        self,
        cell_id: str | None,
        required_inputs: set[str],
        current_cell_outputs: set[str] | None,
    ) -> None:
        """Reset lineage for variables that are both inputs and outputs of the current cell.

        Called when the current cell cannot be found on disk (unsaved edit).
        Uses the simulation cache's pre-cell virtual lineage to reset any
        "ahead" lineage caused by a prior downstream execution.
        """
        if not (required_inputs and current_cell_outputs and len(self.simulator.cache)):
            return

        overlap_vars = required_inputs & current_cell_outputs
        if not overlap_vars:
            return

        cache_idx = self._resolve_fallback_cache_idx(cell_id)
        if cache_idx is None:
            return

        entry = self.simulator.cache.entry(cache_idx)
        if entry is None:
            return
        self._reset_advanced_lineages(overlap_vars, entry.virtual_lineage, cache_idx)

    def _handle_unsaved_cell(
        self,
        cell_code: str,
        cell_id: str | None,
        required_inputs: set[str],
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
    ) -> int | None:
        """Determine how to proceed when the current cell is not found on disk.

        Returns a new ``current_cell_idx`` (len(notebook_cells) to treat all
        saved cells as upstream) when missing inputs require simulation, or
        ``None`` when no further action is needed (caller should return early).
        Performs side-effects (lineage reset, cache invalidation) as needed.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[UPSTREAM_DEBUG] Current cell not found in notebook")
            logger.debug("[UPSTREAM_DEBUG]   Looking for: %s...", cell_code.strip()[:60])
            for i, c in enumerate(notebook_cells[:5]):
                logger.debug("[UPSTREAM_DEBUG]   Cell %d: %s...", i, c.strip()[:60])

        self._unplaced_runs.append((_statement_texts(cell_code), _mtime(self._notebook_path_for_staleness)))

        # UNSAVED CELL UPSTREAM RESOLUTION
        missing_inputs: set[str] = set()
        if required_inputs:
            for inp in required_inputs:
                if inp in BUILTIN_NAMES or inp.startswith("_"):
                    continue
                if inp not in self.shell.user_ns:
                    missing_inputs.add(inp)

        if missing_inputs and notebook_cells:
            logger.debug("[UPSTREAM_DEBUG]   Unsaved cell has missing inputs: %s", missing_inputs)
            logger.debug(
                "[UPSTREAM_DEBUG]   Treating all %d saved cells as upstream",
                len(notebook_cells),
            )
            return len(notebook_cells)

        self._warn_unplaced(cell_code, cell_id, required_inputs)

        # DOWNSTREAM ADVANCEMENT FALLBACK (unsaved cell, no missing inputs)
        self._handle_downstream_advancement_fallback(cell_id, required_inputs, current_cell_outputs)

        invalidate_notebook_path_cache()
        return None

    def _warn_unplaced(self, cell_code: str, cell_id: str | None, required_inputs: set[str]) -> None:
        """Say NOTEBOOK-CELL-UNPLACED: the cell is not in the notebook cash
        read, so the cells above it were not checked.

        Only for a cell that reads a name of the notebook: one that reads
        none has nothing above it to bring up to date. Once per cell text,
        so a cell run again unchanged does not repeat it.
        """
        reads = sorted(n for n in required_inputs if n not in BUILTIN_NAMES and not n.startswith("_"))
        if not reads:
            return
        text = normalise_source(cell_code)
        if text in self._warned_unplaced:
            return
        self._warned_unplaced.add(text)
        why = (
            "the cell id the editor sent is not in it"
            if cell_id
            else "the editor sent no cell id and no saved cell has this text"
        )
        warn_diagnostic(
            CashWarning,
            "NOTEBOOK-CELL-UNPLACED",
            f"cash could not find this cell in the saved notebook ({why}), so it did not check the "
            f"cells above it: an edit above that has not run since does not reach "
            f"{', '.join(reads[:3])}{', ...' if len(reads) > 3 else ''} here, as it would in a run "
            "from the top.",
            "save the notebook (Ctrl+S) and run the cell again.",
            location=("<cash>", 0),
        )

    def _load_notebook_and_find_cell(
        self,
        cell_code: str,
        required_inputs: set[str],
        current_cell_outputs: set[str] | None,
        notebook_path: str | None,
        cell_id: str | None = None,
    ) -> tuple[list[str] | None, int | None]:
        """Load the notebook and resolve the current cell index.

        ``notebook_path`` is the path resolved once at the start of the cell's
        upstream check and threaded here, so Phase 2 does not re-run
        discovery.  Returns ``(notebook_cells, current_cell_idx)``.  If the
        notebook cannot be found or the cell index cannot be resolved,
        ``current_cell_idx`` may be ``None``, which the caller should treat as
        "return early with empty".  ``notebook_cells`` is ``None`` when no
        notebook file exists.
        """
        # Pass the (possibly None) resolved path straight through: when it is
        # None, get_notebook_cells re-resolves internally, but that lookup is
        # bounded by the negative cache, so it stays cheap while
        # preserving the get_notebook_cells seam that callers/tests patch.
        notebook_cells = get_notebook_cells(notebook_path)

        if not notebook_cells:
            logger.debug("[UPSTREAM_DEBUG] No notebook file found, skipping notebook check")
            return None, None

        logger.debug("[UPSTREAM_DEBUG] Found %d notebook cells", len(notebook_cells))

        cells_with_ids = get_notebook_cells_with_ids(notebook_path)

        current_cell_idx = self._resolve_current_cell_idx(
            cell_code, notebook_cells, cell_id, cells_with_ids, required_inputs, current_cell_outputs
        )
        self._note_how_it_ran(cell_code, cell_id, cells_with_ids, current_cell_idx)
        notebook_cells = self._as_run(notebook_cells, cells_with_ids)
        if current_cell_idx is not None:
            self.simulator.virtual_lineage.forget_cells_running("", current_cell_idx)
            # Not right after the cell above it: run on its own, not in a pass down the notebook.
            self._reran = current_cell_idx > 0 and self.last_cell_index != current_cell_idx - 1
            self.last_cell_index = current_cell_idx
            self.last_found_cell = (cell_code, current_cell_idx)
            if cell_id:
                self.simulator.cache.last_index_by_cell_id[cell_id] = current_cell_idx
            if current_cell_idx > 0:
                warn_no_live_reader_once(notebook_path)

        return notebook_cells, current_cell_idx

    def _note_how_it_ran(
        self, cell_code: str, cell_id: str | None, cells_with_ids: list | None, current_cell_idx: int | None
    ) -> None:
        """Remember the cell, found by its id, as it runs when the file says otherwise."""
        if not cell_id or not cells_with_ids or current_cell_idx is None or current_cell_idx >= len(cells_with_ids):
            return
        found_id, file_code = cells_with_ids[current_cell_idx]
        if found_id != cell_id:
            return
        if normalise_source(cell_code) == normalise_source(file_code):
            self._ran_unsaved.pop(cell_id, None)
        else:
            self._ran_unsaved[cell_id] = (file_code, cell_code)

    def _as_run(self, notebook_cells: list[str], cells_with_ids: list | None) -> list[str]:
        """*notebook_cells* with each cell cash ran unsaved read as it ran.

        Its saved source is what the kernel no longer holds: simulating it
        made the next cell re-run the old code over what the user had just
        run, silently undoing the edit. A cell whose file source has changed
        since (saved, or edited again and saved) is read from the file.
        """
        if not self._ran_unsaved or not cells_with_ids or len(cells_with_ids) != len(notebook_cells):
            return notebook_cells
        cells = list(notebook_cells)
        for i, (cell_id, file_code) in enumerate(cells_with_ids):
            ran = self._ran_unsaved.get(cell_id) if cell_id else None
            if ran is not None and ran[0] == file_code:
                cells[i] = ran[1]
        return cells

    def _unsaved_bindings(self, notebook_cells: list[str], notebook_path: str | None) -> dict[str, str]:
        """Names holding what a cell cash ran unsaved bound, with that code.

        A cell run with an edit not yet saved, which no cell id ties to its
        saved cell, binds names the notebook does not explain. Its record
        lasts until the file is written again.
        """
        stamp = _mtime(notebook_path)
        self._unplaced_runs = [run for run in self._unplaced_runs if run[1] == stamp]
        if not self._unplaced_runs:
            return {}
        unplaced = frozenset().union(*(run[0] for run in self._unplaced_runs))
        saved: frozenset[str] | None = None
        user_ns = self.shell.user_ns
        bindings: dict[str, str] = {}
        for name, code in self.tracking_state.executed_cell_codes.items():
            code = strip_markers(code).strip()
            if name not in user_ns or code not in unplaced:
                continue
            if saved is None:
                saved = _notebook_statement_texts(tuple(notebook_cells))
            if code not in saved:
                bindings[name] = code
        return bindings

    @staticmethod
    def _refuse_to_undo(bindings: dict[str, str], names: set[str]) -> None:
        """Stop before *names* are rebuilt over what an unsaved run bound.

        The check would rebuild them from the saved code above and silently
        undo the run (``x = 2`` run, ``x = 1`` saved: the next cell re-ran
        ``x = 1``). cash cannot tell where that code sits, so it says so
        instead of guessing, until the notebook is saved. An unsaved edit the
        simulation keeps (an extension of the saved code) is never rebuilt,
        so it is not refused.
        """
        for name in sorted(names & bindings.keys()):
            raise UpstreamStateError(
                f"'{name}' holds the result of `{bindings[name].splitlines()[0][:80]}`, which ran in this "
                "kernel but is not in the saved notebook, so cash cannot tell which cell it belongs to and "
                "re-running the saved code would undo it. Save the notebook (Ctrl+S), then run this "
                "cell again."
            )

    def _put_back(
        self, bindings: dict[str, Any], lineages: dict[str, str], codes: dict[str, str] | None = None
    ) -> None:
        """Bind every name the repair restored or re-ran to what it held
        before, with the lineage it had then.

        A repair that cannot finish leaves some names rebuilt and others not
        (``data = []`` re-run, the loop that fills it failing): a cell run on
        that mix reads values neither a plain run nor a run from the top
        gives. Put back, the cell runs on what plain Jupyter has. A change the
        repair made inside an object it did not rebind stays.

        With *codes*, each name put back also gets back the statement that
        made it (``executed_cell_codes``): the one the repair re-ran is not.
        """
        user_ns = self.shell.user_ns
        lineage = self.tracking_state.lineage
        for name in set(user_ns) | set(bindings):
            before = bindings.get(name, _UNBOUND)
            if user_ns.get(name, _UNBOUND) is before:
                continue
            if before is _UNBOUND:
                user_ns.pop(name, None)
            else:
                user_ns[name] = before
            if name in lineages:
                lineage.reset_to(name, lineages[name])
            else:
                lineage.discard(name)
            if codes is not None:
                made_by = self.tracking_state.executed_cell_codes
                if name in codes:
                    made_by[name] = codes[name]
                else:
                    made_by.pop(name, None)

    @staticmethod
    def _repair_cells(
        statements: list[str], restored_info: list[ProcessResult], notebook_cells: list[str]
    ) -> dict[str, int]:
        """``{name: the index of the last cell}`` a statement the repair
        re-ran or restored that touches the name stands in: one it restored
        binding it, or one it re-ran reading or binding it (``y.append(2)``
        binds nothing by its text, yet changes ``y``). A statement not found
        in the notebook leaves its names out."""
        cells = statement_cells(tuple(notebook_cells))
        found: dict[str, int] = {}

        def note(code: str, names: Any) -> None:
            at = cells.get(strip_markers(code).strip())
            if at is None:
                return
            for name in names:
                found[name] = max(at, found.get(name, at))

        for stmt in statements:
            try:
                reads, binds = CodeAnalyzer.analyze_code_block(stmt)
            except SyntaxError:
                continue
            note(stmt, reads | binds)
        for metrics in restored_info:
            note(metrics.get("code") or "", metrics.get("restored_vars") or ())
        return found

    def _keep_live_bindings(
        self,
        rebound: dict[str, int],
        bindings: dict[str, Any],
        lineages: dict[str, str],
        codes: dict[str, str],
        cells: dict[str, int | None],
        sessions: dict[str, str],
        current_cell_idx: int,
        required_inputs: set[str],
    ) -> None:
        """Put back the live binding of each name the repair rebound from a
        cell above the one whose run bound it last.

        Restoring or re-running a statement binds every name it binds:
        ``X, y = make_moons(...)``, run again on the way to a ``voting_clf``,
        bound ``X`` and ``y`` to the moons data where a cell below had bound
        other arrays to them, and the cell ran on that, where plain Jupyter
        and a run from the top both have what the cell below bound. *rebound*
        is the cell of the last repair statement binding each name
        (`_repair_cells`); *cells*, the cell whose run bound it before the
        repair (``bound_in_cell``). Put back with its lineage, its
        statement and its session digest.

        A cell above the cell must still bind it as it did: the live lineage
        is the one a run from the top has where the cell starts
        (``simulated_lineage``).

        Not a name the cell reads that the cell itself or a cell below it
        bound: a run from the top gives the cell what the repair gave it
        (``b = b * 10`` run again reads ``b`` as the cell above made it).
        """
        user_ns = self.shell.user_ns
        state = self.tracking_state
        for name, repaired_at in sorted(rebound.items()):
            at = cells.get(name)
            if at is None or at <= repaired_at:
                continue
            if at >= current_cell_idx and name in required_inputs:
                continue
            if at < current_cell_idx and state.simulated_lineage.get(name) != lineages.get(name):
                # The statement that bound it is no longer what the cell
                # there says (``data = data + [4]`` commented out).
                continue
            before = bindings.get(name, _UNBOUND)
            if before is _UNBOUND or name not in user_ns or user_ns[name] is before or name not in lineages:
                continue
            user_ns[name] = before
            state.lineage.reset_to(name, lineages[name])
            if name in codes:
                state.executed_cell_codes[name] = codes[name]
            state.bound_in_cell[name] = at
            if name in sessions:
                state.current_session_hashes[name] = sessions[name]
            trace_event("kept_live_binding", name=name, bound_in=at, repaired_in=repaired_at)

    def _warn_rebuilt_changed_below(
        self,
        rebuilt: set[str],
        bindings_before: dict[str, Any],
        codes_before: dict[str, str],
        notebook_cells: list[str],
        current_cell_idx: int,
    ) -> None:
        """Say NOTEBOOK-CHANGED-BELOW for a name the repair rebuilt on the
        way to the cell's inputs that a cell below had changed in place.

        The classifier says so for an input of the cell. A name rebuilt to
        make one goes the same way unsaid: an edited helper rebuilt ``hist``
        from ``played``, and ``played`` lost the column a cell below had
        added to it. What the cell gets is what a run from the top gives
        it; the warning tells the user ``played`` no longer holds the change.
        """
        warned = self.simulator.classifier._warned_changed_below
        for name in sorted(rebuilt):
            before = bindings_before.get(name, _UNBOUND)
            if before is _UNBOUND or self.shell.user_ns.get(name, _UNBOUND) is before:
                continue
            warn_changed_below(name, codes_before.get(name), notebook_cells, current_cell_idx, warned, along=True)

    def _rebuilt_names(self, statements: list[str], restored_info: list[ProcessResult]) -> set[str]:
        """What the planned re-runs bind, and what the simulation restored."""
        names: set[str] = set()
        for stmt in statements:
            try:
                names |= CodeAnalyzer.analyze_code_block(stmt)[1]
            except SyntaxError:
                continue
        for metrics in restored_info:
            names.update(metrics.get("restored_vars") or ())
        return names

    def _resolve_current_cell_idx(
        self,
        cell_code: str,
        notebook_cells: list[str],
        cell_id: str | None,
        cells_with_ids: list,
        required_inputs: set[str],
        current_cell_outputs: set[str] | None,
    ) -> int | None:
        """Find the current cell index, handling the unsaved-cell fallback.

        Returns the index, or ``None`` when the caller should return early with
        an empty result.
        """
        current_cell_idx = self._find_current_cell_index(
            cell_code, notebook_cells, cell_id=cell_id, cells_with_ids=cells_with_ids
        )
        if current_cell_idx is None:
            return self._handle_unsaved_cell(cell_code, cell_id, required_inputs, current_cell_outputs, notebook_cells)
        return current_cell_idx

    def _bring_up_to_date(
        self,
        cell_code: str,
        required_inputs: set[str],
        process_statement_callback: Callable[..., ProcessResult],
        global_ttl: int | None,
        effects: CellEffects | None = None,
        notebook_path: str | None = None,
        progress_callback: Callable[..., None] | None = None,
        control_structure_callback: Callable[..., Any] | None = None,
        cell_id: str | None = None,
    ) -> UpstreamResult:
        """Bring the state the cell reads up to date with the notebook above it.

        Find the cell, vet the notebook, simulate, re-run or restore what the
        simulation found stale, then resync the simulation with what ran.
        """
        bindings_before: dict[str, Any] | None = None
        try:
            notebook_cells, current_cell_idx = self._load_notebook_and_find_cell(
                cell_code,
                required_inputs,
                set(effects.outputs) if effects is not None else None,
                notebook_path,
                cell_id=cell_id,
            )
            if notebook_cells is None or current_cell_idx is None:
                return UpstreamResult([], 0.0, 0.0)
            self.vetter.vet(notebook_cells, cell_code, current_cell_idx, required_inputs - self._callee_inputs)
            self.streams_above(notebook_cells[:current_cell_idx])
            self._simulated = True

            unsaved = self._unsaved_bindings(notebook_cells, notebook_path)
            self.simulator.restorer.guard = (lambda names: self._refuse_to_undo(unsaved, names)) if unsaved else None
            self.simulator.restorer.lookup_ttl = global_ttl
            records_before = self.simulator.lineage_records()
            # What the namespace held before anything was restored or re-run,
            # to put back if the repair turns out to have a gap.
            bindings_before = dict(self.shell.user_ns)
            # Cash's own references, not a holder of the user's: a statement
            # the repair restores is not refused for the objects they keep.
            hold_repair_snapshot(bindings_before)
            lineages_before = dict(self.tracking_state.variable_lineage)
            # Where each name's live binding came from, and its session
            # digest, for the names a repair rebinds on the way
            # (`_keep_live_bindings`).
            cells_before = dict(self.tracking_state.bound_in_cell)
            session_before = dict(self.tracking_state.current_session_hashes)
            # The statement that last changed each name, before a restore
            # records the one it restored (`_warn_rebuilt_changed_below`).
            codes_before = dict(self.tracking_state.executed_cell_codes)
            # What the simulation last saw of the cells: an interrupted repair
            # puts it back with the names, so the next check still sees the edit.
            cache = self.simulator.cache
            seen_before = (list(cache.entries), dict(cache.cell_hashes))
            # The cell's own source goes along: the classifier re-simulates it to
            # tell its own earlier run apart from an upstream edit.
            self.simulator.restorer.restored_rng_states.clear()
            try:
                statements_to_reexecute, restored_info, total_restore_time = self.simulator.simulate_upstream(
                    current_cell_idx,
                    notebook_cells,
                    required_inputs,
                    effects,
                    cell_code=cell_code,
                )
            except UpstreamRepairGap:
                # A plan that would run a statement on values a run from the
                # top does not give it: nothing re-runs, and what the backward
                # scan restored goes back too.
                self._put_back(bindings_before, lineages_before)
                raise
            logger.debug(
                "[UPSTREAM_DEBUG] Simulation result: %s stmts to re-execute, %s stmts restored from cache",
                len(statements_to_reexecute),
                len(restored_info),
            )
            statements_to_reexecute, rng_rerun = self.rng.with_rng_chain(
                cell_code,
                notebook_cells,
                current_cell_idx,
                statements_to_reexecute,
                trace=self.simulator.last_trace,
                restored_states=self.simulator.restorer.restored_rng_states,
            )
            rebuilt = self._rebuilt_names(statements_to_reexecute, restored_info)
            self._refuse_to_undo(unsaved, rebuilt)

            executed_metrics = []
            if statements_to_reexecute:
                try:
                    executed_metrics = self.replay.reexecute(
                        statements_to_reexecute,
                        process_statement_callback,
                        global_ttl,
                        progress_callback=progress_callback,
                        restored_info=restored_info,
                        control_structure_callback=control_structure_callback,
                        annotations=self.replay.statement_directives(notebook_cells),
                        exec_sources=self.replay.statement_exec_sources(notebook_cells),
                        notebook_cells=notebook_cells,
                        before_statement=self.rng.before_statement,
                    )
                except UpstreamRepairGap:
                    self._put_back(bindings_before, lineages_before)
                    raise
                except KeyboardInterrupt:
                    # Stopped part way: ``lines = f.readlines()`` re-ran,
                    # the loop splitting it did not, and the next check took
                    # the unsplit lines for what the cell makes, the
                    # half-filled ``lines2`` for the loop's (it trusts what a
                    # loop fills once the statement binding it matches). The
                    # names go back to what they held, and the simulation to
                    # what it saw of the cells (or the next check, taking the
                    # edit for one it already repaired, trusts the loop
                    # again), as if the repair never started: the next cell
                    # repairs them again.
                    self._put_back(bindings_before, lineages_before, codes_before)
                    cache.entries, cache.cell_hashes = seen_before
                    raise
            self._keep_live_bindings(
                self._repair_cells(statements_to_reexecute, restored_info, notebook_cells),
                bindings_before,
                lineages_before,
                codes_before,
                cells_before,
                session_before,
                current_cell_idx,
                required_inputs,
            )
            self.rng.label_rerun_metrics(executed_metrics, rng_rerun)
            total_execution_time = self.replay.sum_execution_times(executed_metrics)
            self._warn_rebuilt_changed_below(rebuilt, bindings_before, codes_before, notebook_cells, current_cell_idx)

            # Put the random streams the cell draws from where a run from the
            # top has them before it. After the re-runs above, so it is the
            # last thing to touch them before the cell.
            self.rng.restore_position_rng_state(cell_code, notebook_cells, current_cell_idx)

            self.simulator.resync_after_replay(records_before)
            self.simulator.record_consumable_bases(required_inputs, current_cell_idx, cell_code)
            # The same cell-entry baseline for a frame, an array or a large
            # collection the cell may change in place: the very object it
            # starts from, which its next run compares with.
            self.simulator.stale_values.note_cell_start(current_cell_idx, required_inputs)
            self.tracking_state.running_cell = current_cell_idx
            self.tracking_state.running_cell_rerun = self._reran

            all_metrics = self.replay.in_notebook_order(restored_info + executed_metrics, notebook_cells)
            return UpstreamResult(all_metrics, total_restore_time, total_execution_time)

        except (RuntimeError, SyntaxError):
            raise
        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.debug("[UPSTREAM] Error in the upstream check: %s", e)
            raise UpstreamStateError(f"Failed to restore or simulate upstream state: {e}") from e
        finally:
            if bindings_before is not None:
                release_repair_snapshot(bindings_before)


def _warn_state_not_rebuilt(names: set[str], user_ns: dict[str, Any]) -> None:
    """Report the reloaded modules whose state no simulation rebuilt: the
    notebook's cells, or the cell's place among them, could not be found."""
    for module in sorted({user_ns[name].__name__ for name in names if name in user_ns}):
        warn_diagnostic(
            CashWarning,
            "NOTEBOOK-RELOAD-STATE",
            f"reloading the edited module {module!r} dropped the state cells set on it, and cash cannot "
            "rebuild it: it does not know the notebook's cells, or where this cell is among them",
            "run the cells that set it again.",
        )


def _warn_unheld_state(state: TrackingState) -> None:
    """Report each statement an earlier kernel ran that set state on a local
    module no name of the notebook sees, and that has not run in this one:
    nothing rebuilds that state, and a cell reading it would compute on the
    file's values (``TrackingState.unheld_module_state``)."""
    ran = {statement_source_hash(code) for codes in state.module_state_writers.values() for code in codes}
    for source_hash, (code, modules) in list(state.unheld_module_state.items()):
        del state.unheld_module_state[source_hash]
        if source_hash in ran:
            continue
        first = strip_markers(code).strip().splitlines()[0] if strip_markers(code).strip() else code
        warn_diagnostic(
            CashWarning,
            "NOTEBOOK-RELOAD-STATE",
            f"`{first}` set state on the module {', '.join(sorted(modules))!s} in an earlier kernel, and cash "
            "cannot rebuild it in this one: no name the notebook holds is the module or something taken from it",
            "run that cell again, or import the module itself in a cell above it.",
        )
