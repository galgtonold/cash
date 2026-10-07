from __future__ import annotations

import ast
import functools
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from ..._memo import NOTEBOOK_CELLS, NOTEBOOK_VERSIONS
from ...analysis.code_analyzer import CodeAnalyzer, parse_cell_source
from ...analysis.mutation_effects import CellEffects, NotebookSources, cell_effects
from ...control_markers import strip_markers
from ...diagnostics import warn_diagnostic
from ...exceptions import AmbiguousCellError, CashWarning, UpstreamStateError
from ...value_types import BUILTIN_NAMES
from .._protocols import CashInstanceProtocol, ShellProtocol
from ..cache_key import statement_source_hash
from ..server_discovery import (
    get_notebook_cells,
    get_notebook_cells_with_ids,
    get_notebook_path,
    invalidate_notebook_path_cache,
    warn_notebook_not_found_once,
)
from ..staleness import StalenessTracker, normalise_source
from ..statement.carrier_advances import reachable_generators
from ..tracking_state import TrackingState
from .notebook_vetting import NotebookVetter
from .replay import StatementReplay
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


@functools.lru_cache(maxsize=NOTEBOOK_CELLS)
def _statement_texts(cell_code: str) -> frozenset[str]:
    """Every statement of *cell_code*, at any depth, as cash records the code
    that produced a name (``TrackingState.executed_cell_codes``)."""
    tree = parse_cell_source(cell_code)
    if tree is None:
        return frozenset()
    return frozenset(ast.unparse(node).strip() for node in ast.walk(tree) if isinstance(node, ast.stmt))


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

        # Proven-stale verdict for the saved .ipynb, held for the session.
        # Populated at the cell-ID match site below, where both the running
        # code and the file's copy of that cell are already in hand.
        self.staleness = StalenessTracker()
        self._notebook_path_for_staleness: str | None = None
        #: Cell id -> (the file's source, the source that ran) for a cell cash
        #: ran with an edit the saved notebook does not have yet. The kernel
        #: holds what ran, so the cells above a later cell are read as run
        #: until the file changes (``_as_run``).
        self._ran_unsaved: dict[str, tuple[str, str]] = {}
        #: The statements of each cell cash ran that the saved notebook does
        #: not hold and no cell id places, with the file's mtime then
        #: (``_unsaved_bindings``).
        self._unplaced_runs: list[tuple[frozenset[str], float | None]] = []
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
        normalized newlines → stripped whitespace.  Returns the first
        unambiguous match, or raises ``AmbiguousCellError`` when multiple
        cells share the same content and no cell ID is available.
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

        raise AmbiguousCellError(
            f"Ambiguous cell execution! The current cell content appears {len(matches)} times in the notebook and no cell ID could be resolved. Please ensure cells are unique or save the notebook."
        )

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

        # A name the previous cell's forward probe held and no restore filled.
        self.simulator.restorer.drop_probe_placeholders()

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
        effects = cell_effects(cell_code, self._notebook_sources(cell_code, notebook_path), self.shell.user_ns)
        required_inputs = required_inputs | effects.hidden_inputs
        # A generator a called function draws from (`def boot(x): rng...`)
        # moves without being named, as a generator named in the cell does: an
        # isolated re-run first puts it where a top-to-bottom run has it.
        required_inputs = required_inputs | reachable_generators(required_inputs, self.shell.user_ns)
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

        # DOWNSTREAM ADVANCEMENT FALLBACK (unsaved cell, no missing inputs)
        self._handle_downstream_advancement_fallback(cell_id, required_inputs, current_cell_outputs)

        invalidate_notebook_path_cache()
        return None

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
            self.last_cell_index = current_cell_idx
            if cell_id:
                self.simulator.cache.last_index_by_cell_id[cell_id] = current_cell_idx

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
            self.vetter.vet(notebook_cells, cell_code, current_cell_idx, required_inputs)
            self._simulated = True

            unsaved = self._unsaved_bindings(notebook_cells, notebook_path)
            self.simulator.restorer.guard = (lambda names: self._refuse_to_undo(unsaved, names)) if unsaved else None
            records_before = self.simulator.lineage_records()
            # The cell's own source goes along: the classifier re-simulates it to
            # tell its own earlier run apart from an upstream edit.
            statements_to_reexecute, restored_info, total_restore_time = self.simulator.simulate_upstream(
                current_cell_idx,
                notebook_cells,
                required_inputs,
                effects,
                cell_code=cell_code,
            )
            logger.debug(
                "[UPSTREAM_DEBUG] Simulation result: %s stmts to re-execute, %s stmts restored from cache",
                len(statements_to_reexecute),
                len(restored_info),
            )
            statements_to_reexecute, rng_rerun = self.rng.with_rng_chain(
                cell_code, notebook_cells, current_cell_idx, statements_to_reexecute
            )
            self._refuse_to_undo(unsaved, self._rebuilt_names(statements_to_reexecute, restored_info))

            executed_metrics = []
            if statements_to_reexecute:
                executed_metrics = self.replay.reexecute(
                    statements_to_reexecute,
                    process_statement_callback,
                    global_ttl,
                    progress_callback=progress_callback,
                    restored_info=restored_info,
                    control_structure_callback=control_structure_callback,
                    annotations=self.replay.statement_directives(notebook_cells),
                    notebook_cells=notebook_cells,
                )
            self.rng.label_rerun_metrics(executed_metrics, rng_rerun)
            total_execution_time = self.replay.sum_execution_times(executed_metrics)

            # Restore the position-correct RNG state
            # right before the current draw runs, so a re-executed draw continues
            # from the stream position (and under the seed) it holds top-to-bottom
            # rather than wherever the live state was last left. Runs after any
            # upstream re-execution above, so it is the last thing to touch the
            # RNG before the cell.
            self.rng.restore_position_rng_state(cell_code, notebook_cells, current_cell_idx)

            self.simulator.resync_after_replay(records_before)
            self.simulator.record_consumable_bases(required_inputs, current_cell_idx, cell_code)
            # The same cell-entry baseline for a frame, an array or a large
            # collection the cell may change in place: the very object it
            # starts from, which its next run compares with.
            self.simulator.stale_values.note_cell_start(current_cell_idx, required_inputs)

            all_metrics = self.replay.in_notebook_order(restored_info + executed_metrics, notebook_cells)
            return UpstreamResult(all_metrics, total_restore_time, total_execution_time)

        except (RuntimeError, SyntaxError):
            raise
        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.debug("[UPSTREAM] Error in the upstream check: %s", e)
            raise UpstreamStateError(f"Failed to restore or simulate upstream state: {e}") from e


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
