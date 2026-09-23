from __future__ import annotations

import ast
import functools
import hashlib
import logging
import re
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from cash.control_markers import strip_markers

from ...analysis.annotations import get_statement_annotations, parse_annotation_line
from ...analysis.ast_util import called_names, parse_cached
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source
from ...analysis.mutation_effects import CellEffects, NotebookSources, cell_effects
from ...diagnostics import log_diagnostic, warn_diagnostic
from ...exceptions import AmbiguousCellError, CashUpstreamSyntaxWarning, ForwardReferenceError, UpstreamStateError
from ...tracking.randomness import (
    get_drawing_rng_modules,
    get_seeding_rng_modules,
    restore_rng_state,
    rng_lineage_fingerprint,
    seed_cells_not_yet_run,
)
from ...value_types import BUILTIN_NAMES
from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from ..cache_status import CacheStatus
from ..control_structures import is_control_structure
from ..server_discovery import (
    get_notebook_cells,
    get_notebook_cells_with_ids,
    get_notebook_path,
    invalidate_notebook_path_cache,
    warn_notebook_not_found_once,
)
from ..staleness import StalenessTracker
from ._types import ClassificationResult, SimulationResult
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


def _cell_writes(cell_code: str) -> set[str]:
    """The names a cell binds or changes (by its source); raises SyntaxError
    when it does not parse."""
    tree = parse_cell_source(cell_code)
    if tree is None:
        # ``await`` at the top of a cell parses here and not in the simulation.
        return CodeAnalyzer.analyze_code_block(cell_code)[1]
    return CodeAnalyzer.analyze_code_block(cell_code, tree=tree)[1]


@functools.lru_cache(maxsize=1024)
def _cell_reads(cell_code: str) -> frozenset[str]:
    """The names a cell reads that it does not bind first (by its source)."""
    try:
        clean = CodeAnalyzer.strip_magics(cell_code.replace("\r\n", "\n"))
        inputs, _ = CodeAnalyzer.analyze_code_block(clean)
    except (SyntaxError, ValueError, TypeError):
        return frozenset()
    return frozenset(inputs)


def _bound_by(fn: "ast.AST") -> set[str]:
    """Names a function or lambda binds itself: parameters and local targets.

    Only these can be subtracted safely. A name assigned in the body is bound
    there and never comes from an enclosing cell, so counting it would refuse
    on something no cell above could possibly provide.
    """
    bound: set[str] = set()
    args = getattr(fn, "args", None)
    if args is not None:
        for a in (*getattr(args, "posonlyargs", []), *args.args, *args.kwonlyargs):
            bound.add(a.arg)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                bound.add(extra.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
    return bound


class UpstreamChecker:
    """
    Manages detection and re-execution of changed upstream statements.

    Uses two complementary strategies:
    1. Lineage-based checking: Compares computed lineage hashes for already-executed variables
    2. Notebook-simulation checking: Simulates execution of notebook statements to detect code changes

    Attributes:
        shell: IPython shell instance
        executed_cell_codes: Maps variable names to the statement code that defined them
        executed_cell_hashes: Maps variable names to the SET of hashes of the statement code that defined them
        variable_lineage: Maps variable names to their lineage hash (includes input dependencies)
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

        # per-session ledger of already-warned broken upstream cells,
        # keyed by cell index -> cell source hash. Keeps the "cell N has a
        # syntax error" warning to once per distinct break (not once per
        # downstream cell run) while still re-warning when the break changes or
        # a fixed cell is broken again.
        self._warned_broken_cells: dict[int, str] = {}

        # Proven-stale verdict for the saved .ipynb, held for the session.
        # Populated at the cell-ID match site below, where both the running
        # code and the file's copy of that cell are already in hand.
        self.staleness = StalenessTracker()
        self._notebook_path_for_staleness: str | None = None

        ts = tracking_state or TrackingState()
        self._wire_state(ts)

        # Simulation lives behind a clear seam — see notebook_simulator.py.
        # UpstreamChecker is the orchestrator; the simulator does the AST +
        # cache-probing replay. Shared mutable state (tracking dicts) is
        # passed by reference so writes are visible to both.
        self.simulator = NotebookSimulator(
            shell=shell,
            cash_instance=cash_instance,
            tracking_state=ts,
            compute_hash_fn=compute_hash_fn,
            function_tracker=function_tracker,
        )

    @property
    def function_tracker(self) -> FunctionTracker | None:
        """The runtime's function tracker the simulation keys with."""
        return self.simulator.virtual_lineage.function_tracker

    def reset_caches(self) -> None:
        """Forget the previous simulation.

        Should be called when switching notebooks (e.g., on %cash_on)
        to prevent stale simulation data from a previous notebook
        from interfering with the current one.
        """
        self.simulator.reset_caches()
        # Re-arm the broken-upstream-cell warning for the new notebook:
        # its cell indices/hashes are meaningless across a notebook switch.
        self._warned_broken_cells.clear()
        # A staleness verdict is proof about notebook A's file; carrying it
        # into notebook B (or a fresh %cash_on on the same one) would show a
        # warning about a file this session no longer even reads from, until
        # the first ID-matched run in the new notebook happens to reset it.
        self.staleness.reset()

    def _wire_state(self, state: TrackingState) -> None:
        """Internal: alias tracking dicts onto self so existing attribute
        accesses (``self.executed_cell_codes``, etc.) keep working.

        Kept as a separate method so ``set_tracking_state`` can also forward
        to the simulator.
        """
        self.tracking_state = state
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes
        self.variable_lineage = state.variable_lineage
        self.lineage = state.lineage
        self.executed_file_deps = state.executed_file_deps
        self.vars_with_mutation_lineage = state.vars_with_mutation_lineage
        self.executed_input_lineages = state.executed_input_lineages

    def set_tracking_state(self, state: TrackingState) -> None:
        """Wire all tracking dictionaries from a shared :class:`TrackingState`.

        This is the preferred way to configure tracking state.  All fields
        are aliases to the same mutable containers so mutations are visible
        across ``CashMagics``, ``StatementProcessor``, and ``UpstreamChecker``.
        Also forwards to the simulator so both views stay synchronised.
        """
        self._wire_state(state)
        if hasattr(self, "simulator"):
            self.simulator.set_tracking_state(state)

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
            logger.debug("[UPSTREAM_DEBUG]   current variable_lineage keys: %s", list(self.variable_lineage.keys()))
            if cell_id:
                logger.debug("[UPSTREAM_DEBUG]   cell_id: %s", cell_id)

        self.current_cell_id = cell_id

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

        # Simulate the notebook statement by statement and compare the virtual
        # lineage with the in-memory state to find changed code.
        all_metrics, total_restore_time, total_execution_time = self._check_notebook_based(
            cell_code,
            required_inputs,
            process_statement_callback,
            global_ttl,
            effects,
            notebook_path=notebook_path,
            progress_callback=progress_callback,
            control_structure_callback=control_structure_callback,
        )

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
            if var_name not in cached_virtual_lineage or var_name not in self.variable_lineage:
                continue
            virtual_hash = cached_virtual_lineage[var_name]
            actual_hash = self.variable_lineage[var_name]
            if actual_hash != virtual_hash:
                logger.debug(
                    "[UPSTREAM_DEBUG]   -> Downstream advancement fallback: "
                    "resetting '%s' lineage from %s to virtual %s",
                    var_name,
                    actual_hash[:8],
                    virtual_hash[:8],
                )
                self.lineage.reset_to(var_name, virtual_hash)

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
        cell_id = getattr(self, "current_cell_id", None)

        current_cell_idx = self._resolve_current_cell_idx(
            cell_code, notebook_cells, cell_id, cells_with_ids, required_inputs, current_cell_outputs
        )
        if current_cell_idx is not None:
            self.last_cell_index = current_cell_idx
            if cell_id:
                self.simulator.cache.last_index_by_cell_id[cell_id] = current_cell_idx

        return notebook_cells, current_cell_idx

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

    def _evict_orphaned_definitions(self, notebook_cells: list[str], cell_code: str) -> None:
        """Evict variables whose producing definition no longer exists.

        A variable cash previously produced but that NO current cell statically
        produces is orphaned — its definition was removed, commented out, or
        renamed. It survives in ``user_ns`` with a still-matching lineage, so a
        cached consumer keeps serving a stale value instead of the ``NameError``
        a from-start run would raise. Evict the orphan and its transitive
        consumers from the namespace and all tracking dicts so the consumers
        re-execute their producers (and fail or recompute) on this run.

        Conservative by construction: a candidate must be a real, plainly-named
        user variable currently bound in ``user_ns`` — modules (a ``from X
        import Y`` keeps the tracked source module X in lineage though no cell
        outputs it), cash-internal / dunder names (``_cash_magics``, ``_v``), and
        tracking-only entries with no live binding (an exception ``as e`` that
        Python already unbound) are excluded so they are never wrongly evicted.
        The produced-set spans ALL cells PLUS the current cell (a var produced or
        mutated anywhere is safe — guards an unsaved current cell the on-disk
        notebook view may omit), and if any cell fails to parse the pass is
        skipped rather than risk a wrong eviction.
        """
        produced: set[str] = set()
        for cell in (*notebook_cells, cell_code):
            try:
                produced |= _cell_writes(cell)
            except (SyntaxError, ValueError):
                return  # can't be sure what is produced — do nothing

        user_ns = self.shell.user_ns
        orphaned = {
            v
            for v in set(self.variable_lineage) - produced
            if v in user_ns and not v.startswith("_") and not isinstance(user_ns[v], types.ModuleType)
        }
        if not orphaned:
            return

        # Cascade to transitive consumers via the recorded input lineages.
        to_evict = set(orphaned)
        changed = True
        while changed:
            changed = False
            for var, inputs in self.executed_input_lineages.items():
                if var not in to_evict and not inputs.keys().isdisjoint(to_evict):
                    to_evict.add(var)
                    changed = True

        state = self.tracking_state
        dict_attrs = (
            "variable_lineage",
            "executed_input_lineages",
            "current_session_hashes",
            "variable_hashes",
            "variable_sources",
            "executed_cell_codes",
            "executed_cell_hashes",
            "executed_file_deps",
            "granular_preserved_vars",
            "module_attribute_deps",
            "from_import_sources",
            "from_import_components",
        )
        for var in to_evict:
            self.shell.user_ns.pop(var, None)
            for attr in dict_attrs:
                getattr(state, attr, {}).pop(var, None)
            state.vars_with_mutation_lineage.discard(var)
            logger.debug("[UPSTREAM] evicted orphaned variable '%s'", var)

    @staticmethod
    def _module_level_reads(cell_code: str) -> set[str] | None:
        """Names *cell_code* reads when it RUNS, ignoring deferred lookups.

        A name inside a function or class body is resolved when that function
        is called, not when the cell executes -- but that is a premise, not
        the condition. It only says the cell can run in order if the call
        happens AFTER the binding. Two shapes call it inside this very cell
        and so really do read the name now (round 27, r27s5, both silently
        wrong before this):

        * a lambda handed to a call -- ``s.map(lambda v: f(v))`` invokes it
          inside that statement;
        * a function defined and CALLED in the same cell --
          ``def use_it(): return f(21)`` followed by ``use_it()``.

        A helper that is merely defined here and called from a later cell is
        still deferred, and must stay allowed: judging on the statement's
        inputs, which include those names, refused
        ``test_downward_function_dependency``'s notebook, which runs fine from
        the top. So is a lambda that is stored rather than invoked
        (``handlers = {'x': lambda: f()}``).

        What executes at definition time -- decorators, default arguments,
        base classes -- is collected as before.

        ``None`` when the cell cannot be parsed, which the caller treats as
        "do not refuse anything".
        """
        tree = parse_cell_source(cell_code)
        if tree is None:
            return None

        names: set[str] = set()
        called_here = set(called_names(tree, "eager"))

        def visit(node: ast.AST, into: set[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Evaluated now; the body is not -- unless this cell also
                    # calls it, in which case the body runs before the cell is
                    # over and its free names are read now.
                    for sub in (*child.decorator_list, *child.args.defaults, *(d for d in child.args.kw_defaults if d)):
                        visit_expr(sub, into)
                    if child.name in called_here:
                        _absorb_body(child, into)
                    continue
                if isinstance(child, ast.ClassDef):
                    for sub in (*child.decorator_list, *child.bases):
                        visit_expr(sub, into)
                    continue
                if isinstance(child, ast.Lambda):
                    # Reached other than as a call argument (stored, bound to
                    # a name): only its defaults run now.
                    for sub in (*child.args.defaults, *(d for d in child.args.kw_defaults if d)):
                        visit_expr(sub, into)
                    continue
                if isinstance(child, ast.Call):
                    # A lambda passed to a call is invoked by that call.
                    for arg in (*child.args, *(k.value for k in child.keywords)):
                        if isinstance(arg, ast.Lambda):
                            _absorb_body(arg, into)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    into.add(child.id)
                visit(child, into)

        def visit_expr(node: ast.AST, into: set[str]) -> None:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                into.add(node.id)
            visit(node, into)

        def _absorb_body(fn: ast.AST, into: set[str]) -> None:
            """Free names of *fn*'s body, minus what *fn* itself binds.

            Collected into a scratch set so the parameters can be removed:
            ``lambda v: f(v)`` reads ``f`` from outside and binds ``v``
            itself, and reporting ``v`` would make the guard refuse on a name
            no cell can bind.
            """
            inner: set[str] = set()
            body = fn.body if isinstance(fn.body, list) else [fn.body]
            for stmt in body:
                visit_expr(stmt, inner)
            into |= inner - _bound_by(fn)

        visit(tree, names)
        return names

    def _refuse_forward_references(
        self,
        notebook_cells: list[str],
        cell_code: str,
        current_cell_idx: int,
        required_inputs: set[str],
    ) -> None:
        """Raise when this cell reads a name only a LATER cell binds.

        The sibling of ``_evict_orphaned_definitions``: that one catches a name
        NO cell produces any more, this one a name only a cell BELOW produces.
        Both describe a notebook that cannot reproduce itself, and both are
        invisible while the value happens to be sitting in ``user_ns``.

        Round 26, r26s5: a cell read a variable bound in a cell below it. Under
        cash the notebook worked -- the later cell had been run at some point,
        so the name was there -- and a clean in-order run died with
        ``NameError``. Only the uncached oracle caught it; cash reported
        success on a notebook that was already broken.

        It fails rather than warns. A warning would leave cash caching against
        a namespace its own in-order run could not produce, and everything
        keyed on that state would be built on an ordering the notebook does not
        have. ``NameError`` is what the user is going to get anyway; the only
        question is whether they get it now, with the cell number, or on the
        morning they restart.

        Deliberately conservative, because a false positive here stops a cell
        that works:

        * the current cell's own bindings count as above (``x = x + 1``, or a
          name bound by an earlier statement of the same cell);
        * a name bound anywhere above is fine, whatever else rebinds it below
          -- the common shape of a variable set early and reassigned later;
        * if any cell fails to parse, the whole check is skipped rather than
          risk refusing on a half-read notebook.
        """
        if not required_inputs or current_cell_idx is None:
            return
        # Only names this cell reads at MODULE level can break an in-order run.
        # A name referenced inside a `def` is resolved when the function is
        # CALLED, so `def a(n): return b(n) * 2` above `def b` is ordinary
        # Python -- the call site further down runs after both. Judging on
        # `required_inputs`, which includes those deferred free names, refused
        # `test_downward_function_dependency`'s notebook, which runs fine.
        reads = self._module_level_reads(cell_code)
        if reads is None:
            return
        required_inputs = required_inputs & reads
        if not required_inputs:
            return
        above: set[str] = set()
        below: dict[str, int] = {}
        for idx, cell in enumerate((*notebook_cells, cell_code)):
            try:
                outs = _cell_writes(cell)
            except (SyntaxError, ValueError):
                return  # can't be sure what binds what — never refuse on a guess
            if idx <= current_cell_idx or idx >= len(notebook_cells):
                above |= outs  # the current cell's own bindings included
            else:
                for name in outs:
                    below.setdefault(name, idx)

        user_ns = self.shell.user_ns
        forward = sorted(
            (name, below[name]) for name in required_inputs if name in below and name not in above and name in user_ns
        )
        if not forward:
            return
        names = ", ".join(f"`{n}` (cell {i + 1})" for n, i in forward)
        raise ForwardReferenceError(
            f"this cell reads {names}, which nothing above it binds. It works "
            f"right now only because that cell has already run and the name is "
            f"still in memory -- a run from the top, or tomorrow's kernel, "
            f"raises NameError here. cash refuses rather than cache against a "
            f"namespace your own notebook cannot rebuild in order. Move the "
            f"binding above this cell, or move this cell below it."
        )

    def _warn_broken_upstream_cells(
        self,
        notebook_cells: list[str],
        current_cell_idx: int,
    ) -> set[int]:
        """Emit a visible warning for any UPSTREAM cell that cannot be parsed.

        A half-written cell the user has SAVED but not run makes the upstream
        simulator SKIP that cell (see ``VirtualLineage.simulate_one_cell``)
        so unrelated downstream cells keep caching. But the user must
        still be told: the broken cell will not run, and any cell that depends
        on it can no longer have its dependency tracked. Without this, caching
        degrades silently mid-edit while every signal the user has (the badge,
        ``auto_cache_enabled``) still says it is on — the exact trap that cost
        two round-5 testers a long debugging detour.

        Deduped per ``(cell index, cell hash)`` on this checker so a persistent
        break warns once — not on every downstream cell run — but a NEW or
        CHANGED break re-warns, and a fixed cell that is later re-broken warns
        again. Parsing mirrors the simulator exactly (``strip_magics`` then
        ``ast.parse`` on ``\\r\\n``-normalised source) so a VALID cell — the
        multi-line ``%``-format print — is never falsely
        flagged. Returns the set of broken cell indices (0-based).
        """
        broken: dict[int, str] = {}
        for idx in range(min(current_cell_idx, len(notebook_cells))):
            raw = notebook_cells[idx]
            try:
                if not clean_cell_source(raw).strip():
                    continue
            except (ValueError, TypeError):
                continue
            if parse_cell_source(raw) is None:
                broken[idx] = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        for idx, cell_hash in broken.items():
            if self._warned_broken_cells.get(idx) == cell_hash:
                continue  # already warned about this exact break — stay quiet
            raw = notebook_cells[idx]
            snippet = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            what = (
                f"cell {idx + 1} has a syntax error and could not be parsed "
                f"({snippet!r}), so it is skipped and any cell depending on it "
                f"can no longer be dependency-tracked."
            )
            fix = (
                "fix the syntax error, then re-run that cell and the cells "
                "below that use its output; if it is not really code, delete it "
                "or make it a markdown cell."
            )
            log_diagnostic(logger, "NOTEBOOK-CELL-SYNTAX", what, fix)
            # warn_explicit with registry=None bypasses the "once per location"
            # __warningregistry__ dedupe (every break is raised from this one
            # line); our own per-(idx, hash) ledger supplies the dedupe we
            # actually want, and this still consults the user's filters. Mirrors
            # randomness.py's established pattern.
            warn_diagnostic(
                CashUpstreamSyntaxWarning,
                code="NOTEBOOK-CELL-SYNTAX",
                what=what,
                fix=fix,
                location=("<cash>", idx + 1),
            )

        # Replace the ledger with exactly the current break set: a fixed cell
        # drops out (so a later re-break warns again); a changed break re-warned
        # above and its new hash is recorded here.
        self._warned_broken_cells = broken
        return set(broken)

    def _sum_execution_times(self, executed_metrics: list) -> float:
        """Sum ``total_time`` from a list of metric dicts."""
        total = 0.0
        for metrics in executed_metrics:
            if metrics and "total_time" in metrics:
                total += metrics["total_time"]
        return total

    def _check_notebook_based(
        self,
        cell_code: str,
        required_inputs: set[str],
        process_statement_callback: Callable[..., ProcessResult],
        global_ttl: int | None,
        effects: CellEffects | None = None,
        notebook_path: str | None = None,
        progress_callback: Callable[..., None] | None = None,
        control_structure_callback: Callable[..., Any] | None = None,
    ) -> UpstreamResult:
        """Bring the state the cell reads up to date with the notebook above it.

        Find the cell, vet the notebook, simulate, re-run or restore what the
        simulation found stale, then resync the simulation with what ran.
        """
        try:
            notebook_cells, current_cell_idx = self._load_notebook_and_find_cell(
                cell_code, required_inputs, set(effects.outputs) if effects is not None else None, notebook_path
            )
            if notebook_cells is None or current_cell_idx is None:
                return UpstreamResult([], 0.0, 0.0)
            self._vet_notebook(notebook_cells, cell_code, current_cell_idx, required_inputs)

            records_before = self._lineage_records()
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
            statements_to_reexecute, rng_rerun = self._with_rng_chain(
                cell_code, notebook_cells, current_cell_idx, statements_to_reexecute
            )

            executed_metrics = []
            if statements_to_reexecute:
                executed_metrics = self._reexecute_statements(
                    statements_to_reexecute,
                    process_statement_callback,
                    global_ttl,
                    progress_callback=progress_callback,
                    restored_info=restored_info,
                    control_structure_callback=control_structure_callback,
                    annotations=self._statement_directives(notebook_cells),
                    notebook_cells=notebook_cells,
                )
            self._label_rng_rerun_metrics(executed_metrics, rng_rerun)
            total_execution_time = self._sum_execution_times(executed_metrics)

            # (ADR-018): restore the position-correct RNG state
            # right before the current draw runs, so a re-executed draw continues
            # from the stream position (and under the seed) it holds top-to-bottom
            # rather than wherever the live state was last left. Runs after any
            # upstream re-execution above, so it is the last thing to touch the
            # RNG before the cell.
            self._restore_position_rng_state(cell_code, notebook_cells, current_cell_idx)

            self._resync_after_replay(records_before)

            all_metrics = self._in_notebook_order(restored_info + executed_metrics, notebook_cells)
            return UpstreamResult(all_metrics, total_restore_time, total_execution_time)

        except (RuntimeError, SyntaxError):
            raise
        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.debug("[UPSTREAM] Error in notebook-based checking: %s", e)
            raise UpstreamStateError(f"Failed to restore or simulate upstream state: {e}") from e

    def _vet_notebook(
        self, notebook_cells: list[str], cell_code: str, current_cell_idx: int, required_inputs: set[str]
    ) -> None:
        """What must be settled about the notebook before simulating it.

        Raises ForwardReferenceError for a name only a cell below binds.
        """
        self.tracking_state.read_by_later_cells = frozenset().union(
            *(_cell_reads(code) for code in notebook_cells[current_cell_idx + 1 :])
        )
        # Disclose any unparseable UPSTREAM cell. The simulator skips such a
        # cell so unrelated downstream cells keep caching, but the user must be
        # told which cell is broken — otherwise caching degrades silently
        # mid-edit while the badge and auto_cache_enabled still say it is on.
        self._warn_broken_upstream_cells(notebook_cells, current_cell_idx)
        # A variable whose definition was removed/renamed across an edit is
        # orphaned — no cell produces it anymore. Evict it (and its transitive
        # consumers) so they re-run from the start and raise NameError like a
        # fresh kernel, instead of serving a stale value.
        self._evict_orphaned_definitions(notebook_cells, cell_code)
        # ...and the other half: a name only a cell BELOW binds. Same
        # invisible-while-it-works shape, but it raises.
        self._refuse_forward_references(notebook_cells, cell_code, current_cell_idx, required_inputs)
        logger.debug("[UPSTREAM_DEBUG] Current cell found at index %s", current_cell_idx)

    def _with_rng_chain(
        self, cell_code: str, notebook_cells: list[str], current_cell_idx: int, statements: list[str]
    ) -> tuple[list[str], set[str]]:
        """*statements*, with what re-establishes the random stream the cell
        draws from in front; and which of those were added for that alone.

        ADR-017: a bare ``np.random.seed(N)`` binds no variable, so the
        simulator never links it to a downstream draw. An upstream seed cell
        edited but not re-run is re-run first -- re-running a seed is
        idempotent -- and a draw re-executed because an ORDINARY input changed
        still runs from its top-to-bottom stream position.
        """
        before = set(statements)
        statements = self._prepend_stale_seed_cells(cell_code, notebook_cells, statements, current_cell_idx)
        statements = self._prepend_rng_chain_for_reexecuted_draws(notebook_cells, statements, current_cell_idx)
        return statements, {s for s in statements if s not in before and self.cell_touches_rng(s)}

    def _resync_after_replay(self, records_before: dict[str, tuple]) -> None:
        """Bring the simulation's snapshots in line with what the replay recorded.

        After upstream statements run or are restored, ``variable_lineage``
        holds the authoritative lineage of each. A snapshot may hold a
        simulated one that differs (a control structure simulated as one
        unit), and without the sync the next check sees a mismatch and
        repairs again.
        """
        rerecorded = self._rerecorded_since(records_before)
        self._sync_simulation_cache_lineages(rerecorded)
        # The snapshots of the cells replayed here may not know the files
        # behind what the replay restored (see record_replayed_file_deps).
        self.simulator.record_replayed_file_deps(rerecorded)

    def _lineage_records(self) -> dict[str, tuple]:
        """Each variable's recorded lineage and input-lineage map, as held now.

        The map object is kept (not copied): recording a variable replaces it,
        so ``is`` tells a re-recording apart even when the lineage came out the
        same.
        """
        return {v: (h, self.executed_input_lineages.get(v)) for v, h in self.variable_lineage.items()}

    def _rerecorded_since(self, before: dict[str, tuple]) -> set[str]:
        """Variables this upstream pass recorded again (re-executed or restored)."""
        changed = set()
        for v, h in self.variable_lineage.items():
            old = before.get(v)
            if old is None or old[0] != h or old[1] is not self.executed_input_lineages.get(v):
                changed.add(v)
        return changed

    def _should_sync_cache_var(
        self,
        var_name: str,
        cumulative_stmt_codes: set[str],
        cached_vl: dict[str, str],
        idx: int,
    ) -> bool:
        """Return True if *var_name*'s cached lineage should be synced at cache index *idx*.

        A variable is synced only when its current runtime lineage was produced
        by code within cells 0..idx.  Variables produced by later cells are
        excluded to avoid contaminating earlier cache entries.
        """
        if var_name not in self.variable_lineage:
            return False
        if cached_vl[var_name] == self.variable_lineage[var_name]:
            return False  # Already matches, nothing to sync
        producing_code = self.executed_cell_codes.get(var_name)
        if producing_code is None:
            return True
        normalized_code = strip_markers(producing_code).strip()
        if normalized_code not in cumulative_stmt_codes:
            logger.debug(
                "[UPSTREAM_DEBUG] Skipping sync for '%s' in cache entry %d: producing code not in cells 0..%d",
                var_name,
                idx,
                idx,
            )
            return False
        return True

    def plan_cell_run(
        self,
        nodes: list,
        raw_cell: str,
        occurrence_counts: dict[str, int],
    ) -> dict[int, dict] | None:
        """Which of a run of assignments in the cell being run need not run.

        Round 25 (r25s2): a cell rebuilding ``sales`` through a dozen steps
        writes only the last version to disk (``_written_later_in_cell``), and
        after a restart Run All re-ran every step to get back to it. Here the
        run is simulated the way the upstream repair simulates a cell above,
        and the same backward scan finds the latest versions it can restore;
        what they cover need not run.

        Returns ``{index in nodes: metric}`` for each statement that need not
        run -- restored, or skipped because what it built is current or
        overwritten -- or ``None`` to run them all. Every statement not in the
        result runs as it would have, in order, after the restores.
        """
        try:
            vl = self.simulator.virtual_lineage
            planner = self.simulator.planner
            classifier = self.simulator.classifier
            sim = SimulationResult(virtual_lineage=dict(self.variable_lineage))
            trace = sim.trace
            counts = dict(occurrence_counts)
            for node in nodes:
                before = len(trace)
                vl.simulate_one_node(sim, 0, node, counts, {}, raw_cell=raw_cell)
                if len(trace) != before + 1:
                    return None
            if any(entry.files_stale for entry in trace):
                return None  # a file it reads changed: run it
            final: dict[str, str] = {}
            for entry in trace:
                final.update(entry.produced_lineages)
            if set(final) != set().union(*(entry.outputs for entry in trace)):
                return None
            broken = {
                name
                for name, lineage in final.items()
                if name not in self.shell.user_ns or self.variable_lineage.get(name) != lineage
            }
            restored_by_index: dict[int, dict] = {}
            run: list[int] = []
            if broken:
                run, restored, _ = classifier.backward_scan_pass(
                    sim,
                    ClassificationResult(
                        broken_vars=broken,
                        tainted_vars=set(),
                        trace_codes={entry.stmt_code for entry in trace},
                    ),
                )
                while True:
                    size = len(run)
                    # Stricter than the repair's own pass: a statement that runs
                    # reads the version its run made before it, so that version's
                    # producer runs too. The live value may be a LATER version the
                    # scan restored -- ``is_big = sales['a'] > ...`` ran on the
                    # final ``sales`` otherwise.
                    run = sorted(
                        set(run)
                        | {
                            p
                            for i in run
                            for v in (trace[i].inputs or ())
                            if (p := planner.latest_producer(trace, v, before=i)) is not None
                        }
                    )
                    run = planner.complete_later_producers(run, trace)
                    if len(run) == size:
                        break
                for info in restored:
                    position = info.get("position")
                    if isinstance(position, int):
                        info["is_upstream"] = False
                        restored_by_index[position] = info
            run_set = set(run)
            planned: dict[int, dict] = {}
            for i, entry in enumerate(trace):
                if i in run_set:
                    continue
                planned[i] = restored_by_index.get(i) or {
                    "code": entry.stmt_code,
                    "status": CacheStatus.SKIPPED,
                    "is_upstream": False,
                    "saved_time": 0.0,
                    "total_time": 0.0,
                }
            if broken and not restored_by_index:
                return None  # nothing on disk to jump to: run as usual
            return planned
        except Exception:  # noqa: BLE001 - a plan that cannot be made is the ordinary run
            logger.debug("[UPSTREAM] cell run plan failed", exc_info=True)
            return None

    def _sync_simulation_cache_lineages(self, rerecorded: set[str]) -> None:
        """Sync simulation cache virtual lineages with actual runtime lineages.

        Only the *rerecorded* variables -- the ones this upstream pass just
        re-executed or restored -- are synced. Their runtime lineage is fresh.
        Any other variable's runtime lineage is only as fresh as its last run:
        after an upstream edit, a sibling the pass did not need (``a = f(x)``
        when only ``b = g(x)`` was asked for) still holds the value computed
        from the old ``x``, and its snapshot is the only place that knows.
        Syncing it laundered the stale value into a match, and the next cell
        that read it was served the old result (round 22).

        After upstream statements are executed/restored/skipped, ``variable_lineage``
        holds the authoritative lineage for each variable.  The simulation cache
        may store stale ``virtual_lineage`` values from an earlier run where
        forward propagation failed (e.g., the fallback lineage computed
        differently than the runtime lineage because a control structure was
        simulated as a single unit, or ``inspect.getsource`` returned different
        results).

        This method patches every cached ``virtual_lineage`` snapshot so that
        variables get their lineage updated to the authoritative value — but
        **only if the runtime lineage was produced by code within cells 0..idx**.
        Variables whose runtime lineage was produced by a *later* cell (beyond
        idx) are NOT synced.  This prevents downstream mutations from
        contaminating earlier cache entries.

        For example, if cell 2 produces ``df`` via ``df.sort_values(...)`` and
        cell 5 mutates it via ``df['SMA'] = ...``, after cell 5 executes the
        runtime lineage for ``df`` reflects the SMA mutation.  Without the
        scoping fix, syncing would update cell 2's cached virtual_lineage for
        ``df`` to the SMA-mutated lineage.  Then when cell 4 (a display cell)
        runs, reusing cache for cells 0-2 yields a virtual lineage that
        already matches the mutated actual lineage → no restoration → bug.

        With scoping, we check ``executed_cell_codes['df']`` to see which
        statement last produced ``df``'s runtime lineage.  If that statement
        is ``df['SMA'] = ...`` (from cell 5), it won't be found in cells
        0..2's trace segments, so cell 2's cache entry is NOT synced for
        ``df``.
        """
        if not len(self.simulator.cache):
            return

        updated = False
        # For each cache entry at index idx, collect ALL statement codes that
        # appear in the trace segments of cells 0..idx.  We only sync a
        # variable's lineage if the code that produced the current runtime
        # lineage (from executed_cell_codes) is among these statements.
        cumulative_stmt_codes = set()
        #: ``{var: (old, new)}`` synced so far; later entries' recorded inputs
        #: follow (below).
        moved: dict[str, tuple[str, str]] = {}
        for idx in range(len(self.simulator.cache)):
            entry = self.simulator.cache.entry(idx)
            if entry is None:
                continue
            cell_trace = entry.trace_segment
            for trace_entry in cell_trace:
                cumulative_stmt_codes.add(trace_entry.stmt_code)
                # A statement below a synced one read the value it now names.
                # Left behind, a loop there compared its recorded inputs with
                # the old lineage and read as reading changed data on every run
                # after a repair: ``results = {}`` and everything built on it
                # re-ran each time (round 25, r25s1).
                input_hashes = trace_entry.input_hashes
                if moved and isinstance(input_hashes, dict):
                    for var_name, (old, new) in moved.items():
                        if input_hashes.get(var_name) == old:
                            input_hashes[var_name] = new

            cached_vl = entry.virtual_lineage
            for var_name in list(cached_vl.keys()):
                if var_name not in rerecorded:
                    continue
                if not self._should_sync_cache_var(var_name, cumulative_stmt_codes, cached_vl, idx):
                    continue
                # Safe to sync: the runtime lineage was produced by code within
                # cells 0..idx, so this is a valid forward-propagation correction.
                if cached_vl[var_name] != self.variable_lineage[var_name]:
                    moved[var_name] = (cached_vl[var_name], self.variable_lineage[var_name])
                cached_vl[var_name] = self.variable_lineage[var_name]
                updated = True

        if updated and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[UPSTREAM_DEBUG] Synced simulation cache lineages with runtime state (scoped to producing code)"
            )

    def _prepend_stale_seed_cells(
        self,
        cell_code: str,
        notebook_cells: list[str],
        statements: list[str],
        current_cell_idx: int | None,
    ) -> list[str]:
        """Schedule an edited-but-not-rerun seed cell ahead of a draw (ADR-017).

        When the current cell draws from a global RNG and an UPSTREAM cell seeds
        that module with source that has not executed this session (an edited or
        never-run seed), that seed's side effect must be re-established before the
        draw. But re-seeding alone is not enough when draws sit between
        the seed and the current cell: those intervening draws advanced the stream
        under the OLD seed, so the current draw would run from the new seed's
        position 0 instead of the position the chain holds top-to-bottom
        (combined case). So when a seed is stale, re-run the whole RNG
        chain from the earliest stale seed to the current cell, in order: the
        re-seed updates the global epoch, which makes each intervening draw miss
        and recompute under the new seed, advancing the stream correctly.

        Returns *statements* with those cells prepended in notebook order,
        de-duplicated against what is already scheduled. A no-op when the cell
        does not draw or no seed is stale, so warm draws and the plain re-run
        path are untouched. Only cells strictly BEFORE the current one count.
        """
        try:
            drawing = self._current_cell_drawing_modules(cell_code)
            if not drawing:
                return statements
            # Restrict to genuine upstream cells; a None index means "treat all
            # as upstream" (the checker's own fallback), so scan everything then.
            upstream = notebook_cells if current_cell_idx is None else notebook_cells[:current_cell_idx]
            executed = self.tracking_state.executed_cell_source_hashes
            stale = seed_cells_not_yet_run(drawing, upstream, executed)
            if not stale:
                return statements
            # Rebuild the RNG chain from the earliest stale seed forward: every
            # random cell (seed or draw) from there to the current cell, in order.
            earliest = min(idx for _module, idx in stale)
            already = set(statements)
            prepend: list[str] = []
            for idx in range(earliest, len(upstream)):
                src = upstream[idx]
                if not self.cell_touches_rng(src):
                    continue
                if src not in already and src not in prepend:
                    prepend.append(src)
            if prepend and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[UPSTREAM] Rebuilding RNG chain (%d cells) before draw", len(prepend))
            return prepend + statements
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return statements

    def _prepend_rng_chain_for_reexecuted_draws(
        self,
        notebook_cells: list[str],
        statements: list[str],
        current_cell_idx: int | None,
    ) -> list[str]:
        """Re-establish the RNG stream before a *re-executed* upstream draw (ADR-017).

        The RNG state is a side-effect dependency a draw consumes, but it binds no
        variable, so the lineage graph carries no edge from a draw back to its
        seed. When reconstruction re-executes a draw because one of its ORDINARY
        inputs changed (e.g. ``arr = np.random.rand(3) * MULT`` after editing
        ``MULT``), the unchanged upstream ``seed()`` is not scheduled, so the draw
        re-runs from wherever the live stream was left. The result matches neither
        a cache-off run nor a clean top-to-bottom run -- a silently wrong value.

        Fix: re-run the RNG chain that PRECEDES the earliest re-executed draw --
        the seed plus any draws ahead of it in source order that aren't already
        scheduled -- so that re-executed draw lands at the stream position it
        holds top-to-bottom. Statements AT or AFTER the earliest re-executed draw
        stay in the plan (the re-executed draws run there, in order; an unchanged
        later draw keeps its cached value, whose position is unaffected when the
        edit does not change how many values the re-executed draws consume).

        Chain statements come with the definitions they READ -- see
        :meth:`_with_input_definitions`. Selecting purely on "touches an RNG
        module" would schedule ``base = np.random.randn(n)`` without the
        ``n = 500`` beside it, and the reconstruction would raise ``NameError``.
        Its cell-granular sibling never had this problem: whole cells carry
        their siblings along.

        Statement-granular sibling of :meth:`_prepend_stale_seed_cells`, triggered
        by a re-executed DRAW rather than a stale seed. A no-op on the warm path
        (nothing re-executes), when the seed is already scheduled, and when no
        upstream RNG statement precedes the earliest re-executed draw.
        """
        try:
            drawn: set[str] = set()
            seeded_in_plan: set[str] = set()
            for stmt in statements:
                drawn |= get_drawing_rng_modules(stmt)
                seeded_in_plan |= get_seeding_rng_modules(stmt)
            # Only modules whose draw re-executes but whose seed is NOT already
            # being re-run need their chain re-established.
            missing = drawn - seeded_in_plan
            if not missing:
                return statements
            upstream = notebook_cells if current_cell_idx is None else notebook_cells[:current_cell_idx]
            already = set(statements)

            # Every upstream statement in source order, so a chain statement's
            # own inputs stay locatable, plus the positions of those touching a
            # missing module, tagged with whether they draw.
            all_stmts: list[str] = []
            rng_positions: list[tuple[int, bool]] = []
            for cell in upstream:
                tree = parse_cached(cell)
                if tree is None:
                    continue
                for node in tree.body:
                    try:
                        stmt = ast.unparse(node)
                    except (ValueError, TypeError):
                        continue
                    all_stmts.append(stmt)
                    draws = get_drawing_rng_modules(stmt) & missing
                    seeds = get_seeding_rng_modules(stmt) & missing
                    if draws or seeds:
                        rng_positions.append((len(all_stmts) - 1, bool(draws)))

            # The earliest re-executed draw is the boundary: everything strictly
            # before it re-runs to advance the stream; it and everything after
            # stay in the plan.
            boundary = next(
                (k for k, (idx, is_draw) in enumerate(rng_positions) if is_draw and all_stmts[idx] in already),
                None,
            )
            if boundary is None:
                return statements

            chain = {idx for idx, _is_draw in rng_positions[:boundary] if all_stmts[idx] not in already}
            if not chain:
                return statements
            selected = self._with_input_definitions(chain, all_stmts)

            prepend: list[str] = []
            for idx in sorted(selected):
                stmt = all_stmts[idx]
                if stmt not in prepend:
                    prepend.append(stmt)
            if not prepend:
                return statements
            # A definition the plan already scheduled moves up with the chain
            # that reads it (see `_with_input_definitions`).
            rest = list(statements)
            for stmt in prepend:
                if stmt in rest:
                    rest.remove(stmt)
            statements = rest
            logger.debug(
                "[UPSTREAM] Re-establishing RNG chain (%d stmts) before a re-executed draw",
                len(prepend),
            )
            return prepend + statements
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return statements

    def _with_input_definitions(self, chain: set[int], all_stmts: list[str]) -> set[int]:
        """Widen an RNG chain to include the definitions its statements read.

        The chain is chosen by whether a statement touches an RNG module, which
        says nothing about what it *reads*. ``base = np.random.randn(n)`` is a
        draw and gets prepended; the ``n = 500`` beside it in the same cell is
        not a draw and does not, so the reconstruction raises ``NameError``.

        For every free name a chain statement reads, pull in the nearest
        PRECEDING upstream statement that binds it, then repeat for that
        statement's own inputs. Returns the widened index set; the caller
        re-sorts, so source order is preserved and a definition always lands
        ahead of its reader.

        A name already bound in the live namespace needs no statement of its
        own: re-deriving it would re-run work the kernel already holds, which is
        cheap for ``n = 500`` and not cheap for ``n = load_config()``.

        A definition the plan already schedules is selected too, and the
        caller moves it into the prepend: left where it was, it ran after the
        chain, and after a restart ``np.random.seed(0)`` ran ahead of the
        scheduled ``import numpy as np`` and failed. Moving it earlier keeps it
        ahead of its other consumers, which all come later in the plan.
        """
        live = getattr(self.shell, "user_ns", None)
        if not isinstance(live, dict):
            live = {}
        # name -> indices binding it, built once and only if a name goes missing.
        definers: dict[str, list[int]] | None = None

        selected = set(chain)
        queue = sorted(chain)
        while queue:
            idx = queue.pop()
            try:
                inputs, _outputs = CodeAnalyzer.analyze_code_block(all_stmts[idx])
            except (SyntaxError, ValueError, TypeError):
                continue
            for name in inputs:
                # A builtin name the user never bound needs no statement; one
                # they did (`format = "csv"`) is found like any other.
                if name in live or (name in BUILTIN_NAMES and name not in self.variable_lineage):
                    continue
                if definers is None:
                    definers = {}
                    for j, stmt in enumerate(all_stmts):
                        try:
                            _in, outs = CodeAnalyzer.analyze_code_block(stmt)
                        except (SyntaxError, ValueError, TypeError):
                            continue
                        for out in outs:
                            definers.setdefault(out, []).append(j)
                nearest = next((j for j in reversed(definers.get(name, [])) if j < idx), None)
                if nearest is None or nearest in selected:
                    continue
                selected.add(nearest)
                queue.append(nearest)
        return selected

    @staticmethod
    def _label_rng_rerun_metrics(executed_metrics: list, rng_rerun: set[str]) -> None:
        """Explain, on the badge, why an RNG statement was re-executed (Stage 2 UX).

        A seed/draw pulled into the plan only to re-establish the random stream
        would otherwise render as a bare COMPUTED row in the UPSTREAM section
        with no attribution. Stamp its ``miss_reason`` — the badge's "why did
        this re-run?" field — so the reason reads alongside the Stage-1 random
        pill the row already carries. Matched on stripped source; never fatal.
        """
        if not rng_rerun:
            return
        wanted = {s.strip() for s in rng_rerun}
        for m in executed_metrics:
            try:
                if m.get("code", "").strip() in wanted:
                    m["miss_reason"] = "re-run to restore the random stream"
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                continue

    def _current_cell_drawing_modules(self, cell_code: str) -> set[str]:
        """RNG modules this cell draws from — statically OR by prior observation.

        Static analysis sees ``np.random.rand()`` in the cell; the observed set
        (ADR-018) adds modules a call like ``model.fit()`` changed at runtime, so
        an indirect draw is treated like a direct one on re-run.
        """
        modules = set(get_drawing_rng_modules(cell_code))
        digest = hashlib.sha256(cell_code.encode("utf-8")).hexdigest()
        modules |= self.tracking_state.observed_rng_cells.get(digest, set())
        return modules

    @staticmethod
    def _opts_out_of_rng_rewind(cell_code: str) -> bool:
        """True if *cell_code* carries ``# @cash:no-cache``.

        The rewind is what freezes an unseeded value: the statement re-executes
        but lands on the same stream position, so it redraws the same number.
        Caching is not involved -- a cheap draw is under the persistence floor
        and is never stored in the first place.

        ``no-cache`` is the documented way to say "run this for real every
        time", and the warning cash prints on a frozen draw names it directly.
        So it has to switch the REWIND off, not just caching; otherwise the
        statement dutifully re-executes, redraws the identical value, and the
        one escape hatch users are told to reach for silently does nothing.
        """
        for lineno, line in enumerate(cell_code.splitlines(), 1):
            if not line.strip().startswith("#"):
                continue
            ann = parse_annotation_line(line, lineno)
            if ann is not None and ann.no_cache:
                return True
        return False

    def cell_touches_rng(self, src: str) -> bool:
        """True if *src* seeds or draws — statically or by prior observation."""
        if get_seeding_rng_modules(src) or get_drawing_rng_modules(src):
            return True
        digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
        return bool(self.tracking_state.observed_rng_cells.get(digest))

    def _restore_position_rng_state(
        self,
        cell_code: str,
        notebook_cells: list[str],
        current_cell_idx: int | None,
    ) -> None:
        """Restore the RNG to the state it holds just before this cell (ADR-018).

        If the current cell draws, find the nearest UPSTREAM cell that touched the
        RNG (seed or draw) and whose post-state we recorded this session, and
        restore that state. A re-executed draw then continues from the correct
        stream position instead of the last-left live state. A no-op
        unless the cell draws and such a predecessor exists; on a cache HIT the
        statement restores its own post-state afterwards, so this is harmless.
        """
        try:
            drawing = self._current_cell_drawing_modules(cell_code)
            if not drawing:
                return
            if self._opts_out_of_rng_rewind(cell_code):
                return
            end = len(notebook_cells) if current_cell_idx is None else current_cell_idx
            # If an upstream seed is stale (edited-not-rerun), the whole chain of
            # recorded post-states below it is stale too — restoring one would
            # apply the OLD seed's position. Defer to the reseed path
            # (_prepend_stale_seed_cells) instead of using a stale snapshot.
            upstream = notebook_cells[:end]
            if seed_cells_not_yet_run(drawing, upstream, self.tracking_state.executed_cell_source_hashes):
                return
            # PRIMARY: the position this cell itself started from last time.
            # That is precisely what re-executing its draw needs, and it is exact
            # rather than inferred -- the upstream scan below reaches the same
            # value only indirectly, via the NEAREST predecessor's post-state
            # (equal by construction when nothing RNG-touching sits between).
            # Safe to prefer because the fingerprint expires it as soon as the
            # seed behind it changes, the same lineage check that invalidates any
            # other value.
            own = self.tracking_state.rng_pre_states.get(hashlib.sha256(cell_code.encode("utf-8")).hexdigest())
            if own is not None:
                own_state, own_fingerprint = own
                if own_fingerprint == rng_lineage_fingerprint(
                    self.variable_lineage,
                    drawing,
                ):
                    restore_rng_state(own_state)
                    return
                # Seed changed since it was recorded: the saved position belongs
                # to the old seed. Fall through rather than apply it.
            # FALLBACK: nearest upstream cell that touched RNG. Still needed when
            # this cell has no recorded start of its own (never run this session)
            # or its seed moved on, and it stays fresher than a stale own-position
            # when an upstream cell re-ran more recently than this one.
            post_states = self.tracking_state.rng_post_states
            for idx in range(end - 1, -1, -1):
                src = notebook_cells[idx]
                if not self.cell_touches_rng(src):
                    continue
                digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
                state = post_states.get(digest)
                if state is not None:
                    restore_rng_state(state)
                return  # nearest predecessor only, whether or not it was recorded
        except (AttributeError, IndexError, TypeError):  # pragma: no cover - defensive
            return

    @staticmethod
    def _in_notebook_order(metrics: list, notebook_cells: list[str] | None) -> list:
        """*metrics* in the order their statements stand in the notebook.

        The restores came first and the re-runs after them, so the badge's
        Upstream list read ``^CACHED: results[name] = evaluate(...)`` above
        ``^EXECUTED: results = {}`` -- an order nothing ran in (round 25,
        r25s1). The restores carried a simulation-trace position and the
        re-runs none. Both are placed by their statement's place in the
        notebook; a loop's passes carry their loop's (``stmt_code``, stamped
        by ``_reexecute_statements``) and keep their order. A metric whose
        statement is not found keeps its place after the ones that are.
        """
        order: dict[str, int] = {}
        for cell in notebook_cells or ():
            tree = parse_cell_source(cell)
            if tree is None:
                continue
            for node in tree.body:
                at = len(order)
                order.setdefault(ast.unparse(node), at)
                # A restored loop pass is keyed by its body statement.
                for sub in ast.walk(node):
                    if sub is not node and isinstance(sub, ast.stmt):
                        order.setdefault(ast.unparse(sub), at)
        end = len(order)

        def place(item):
            i, m = item
            code = m.get("upstream_statement") or m.get("code") or ""
            code = strip_markers(code).strip()
            return (order.get(code, end), i)

        return [m for _, m in sorted(enumerate(metrics), key=place)]

    @staticmethod
    def _statement_directives(notebook_cells: list[str] | None) -> dict[str, Any]:
        """``{statement code: its # @cash: directives}`` across the notebook.

        A statement re-run as an upstream repair ran with no annotation: r25s4
        put ``# @cash:no-cache-calls`` on a comprehension, and its calls were
        cached whenever a cell below repaired it (round 25). The repair has the
        statement's code, keyed as the simulator keys it; the directive is read
        from its cell as a direct run reads it. Only statements that carry one.
        """

        found: dict[str, Any] = {}
        for cell in notebook_cells or ():
            if "@cash:" not in cell:
                continue
            tree = parse_cell_source(cell)
            if tree is None:
                continue
            clean = clean_cell_source(cell)
            for node in tree.body:
                try:
                    annotation = get_statement_annotations(clean, node)
                    if annotation.has_directives():
                        found.setdefault(ast.unparse(node), annotation)
                except Exception:  # noqa: BLE001 - a directive lookup never breaks a repair
                    continue
        return found

    def _reexecute_statements(
        self,
        statements: list[str],
        process_callback: Callable[..., ProcessResult],
        global_ttl: int | None,
        progress_callback: Callable[..., None] | None = None,
        restored_info: list[ProcessResult] | None = None,
        control_structure_callback: Callable[..., Any] | None = None,
        annotations: dict[str, Any] | None = None,
        notebook_cells: list[str] | None = None,
    ) -> list[ProcessResult]:
        """Re-execute a list of statements and return their metrics.

        *annotations* maps a statement's code to the ``# @cash:`` directives
        written on it in its cell (see :meth:`_statement_directives`).
        """
        executed_metrics = []
        total_upstream_steps = len(statements)

        for stmt_idx, stmt_code in enumerate(statements):
            # The badge's "Upstream" section is what tells the user; this line
            # is for ``%cash_debug`` (and the integration tests that read it).
            if logger.isEnabledFor(logging.DEBUG):
                stmt_short = stmt_code.split("\n")[0][:40]
                if len(stmt_code) > 40:
                    stmt_short += "..."
                logger.debug("Cash: Auto-executing upstream statement: %s", stmt_short)

            try:
                # Check if this is a control structure (for/if/try/with/while).
                # If so, delegate to the control structure callback which handles
                # per-iteration caching for loops instead of executing monolithically.
                ctrl_node = self._try_parse_control_structure(stmt_code)
                if ctrl_node is not None and control_structure_callback is not None:
                    logger.debug("[UPSTREAM] Delegating control structure to per-iteration processor")
                    ctrl_annotation = (annotations or {}).get(stmt_code)
                    ctrl_result = (
                        control_structure_callback(
                            ctrl_node, ttl=global_ttl, silent=True, inherited_annotation=ctrl_annotation
                        )
                        if ctrl_annotation is not None
                        else control_structure_callback(ctrl_node, ttl=global_ttl, silent=True)
                    )
                    for m in ctrl_result.metrics:
                        if m:
                            m["is_upstream"] = True
                            m.setdefault("upstream_statement", stmt_code)
                            executed_metrics.append(m)
                    if not ctrl_result.success:
                        raise ctrl_result.error or RuntimeError("Error in upstream control structure")
                else:
                    # Silent, like the control-structure branch above: this is
                    # ANOTHER cell's statement, and its output belongs to that
                    # cell -- a plain run of this cell never prints it. Re-running
                    # a figure's history (``print('panels', len(axes))`` among
                    # its fills) put that line into an unrelated cell's output
                    # (round 21, replay acceptance corpus). A failure still
                    # surfaces: the processor reports it in ``result['error']``.
                    stmt_annotation = (annotations or {}).get(stmt_code)
                    result = (
                        process_callback(stmt_code, global_ttl, silent=True, annotation=stmt_annotation)
                        if stmt_annotation is not None
                        else process_callback(stmt_code, global_ttl, silent=True)
                    )
                    logger.debug("[UPSTREAM] Callback result for '%s...': %s", stmt_code[:20], result)
                    if result:
                        result["is_upstream"] = True  # Mark as upstream so badge categorizes correctly
                        executed_metrics.append(result)
                        # The processor reports statement failures via the
                        # 'error' field instead of raising - surface those
                        # through the same loud path below.
                        # With its type: ``str(KeyError('f1'))`` is just ``'f1'``,
                        # and nothing below could tell a NameError from it --
                        # every round-25 repair failure took this path and got
                        # "fix the upstream cell" for a cell with nothing wrong.
                        error = result.get("error")
                        if error:
                            text = (
                                f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
                            )
                            raise UpstreamStateError(
                                self._format_upstream_failure(
                                    stmt_code,
                                    text,
                                    planning_gap=self._planning_gap_for(
                                        error, statements[:stmt_idx], stmt_code=stmt_code, notebook_cells=notebook_cells
                                    ),
                                )
                            )
            except UpstreamStateError:
                raise
            except Exception as e:  # noqa: BLE001 - see below
                # ANY exception, not a list of the ones user code was expected
                # to raise. The statement being re-run is the user's, so what it
                # raises is the user's failure to hear about. The list used to
                # be (RuntimeError, NameError, KeyError, TypeError, ValueError),
                # and an ImportError went straight past it into the executor's
                # catch-all for cash's own bugs: a notebook whose cell wrote an
                # .xlsx without openpyxl installed told the user "cash hit an
                # internal error ... Nothing in your code caused this ... Please
                # report it" on the FOLLOWING cell (round-26 rehearsal).
                # Swallowing here served the downstream cell a STALE value
                # while the upstream producer was silently broken -
                # the worst failure mode for a caching layer. Fail the user's
                # cell with the upstream failure instead, like a plain
                # top-to-bottom run would.
                logger.error("[ERROR] Failed to auto-execute statement: %s", e)
                raise UpstreamStateError(
                    self._format_upstream_failure(
                        stmt_code,
                        f"{type(e).__name__}: {e}",
                        planning_gap=self._planning_gap_for(
                            e, statements[:stmt_idx], stmt_code=stmt_code, notebook_cells=notebook_cells
                        ),
                    )
                ) from e

            # Report progress after each statement
            if progress_callback is not None:
                try:
                    all_so_far = (restored_info or []) + executed_metrics
                    progress_callback(all_so_far, stmt_code, stmt_idx + 1, total_upstream_steps)
                except (TypeError, ValueError, AttributeError):
                    pass  # Don't let progress reporting break execution

        return executed_metrics

    def _planning_gap_for(
        self,
        exc: object,
        already_scheduled: list[str],
        *,
        stmt_code: str | None = None,
        notebook_cells: list[str] | None = None,
    ) -> str | None:
        """Say so when the failure is a gap in cash's repair, not the user's code.

        A statement cash re-runs as a repair can fail for two very different
        reasons, which the user cannot tell apart from the error alone:

        * the code really fails -- a top-to-bottom run would fail too, and
          "fix the upstream cell" is right;
        * cash scheduled the statement that READS something without the one
          that WRITES it, or ran it against incomplete state -- cash's problem,
          and there is nothing in the cell to fix.

        Round 14: ``ax.plot(sub[...])`` without ``sub = mm[...]`` four statements
        earlier. Round 25, four projects: ``name 'in_cents' is not defined`` with
        ``in_cents = ...`` above it in the same cell; ``KeyError: 'f1'`` right
        below ``results["f1"] = ...``; ``KeyError: 'logreg'`` for a dict whose
        filling loop was not re-run. Each tester went looking for a bug in a
        correct cell.

        Evidence, strongest first: a statement above the failing one, not
        re-run first, that writes what is missing -- the name, or the key or
        attribute on a variable the failing statement reads; else this exact
        statement ran without error before. ``None`` when neither holds.
        """
        try:
            missing = self._what_is_missing(exc)
            if missing is None:
                return None
            kind, name = missing
            producer = self._unscheduled_producer(kind, name, stmt_code, already_scheduled, notebook_cells)
            if producer is not None:
                code, cell_no = producer
                first = code.split("\n")[0][:60]
                where = f" (cell {cell_no})" if cell_no else ""
                run_it = f"run cell {cell_no}" if cell_no else "run the cell holding it"
                return (
                    f"NOTE: {name!r} is set by {first!r}{where}, which cash did not "
                    f"re-run first. That is a gap in cash's re-execution plan, not "
                    f"something wrong with your code - to continue, {run_it} yourself "
                    f"and then this cell again (or Restart & Run All), and please report it"
                )
            ran_before = {c for c in (self.executed_cell_codes or {}).values() if isinstance(c, str)}
            if stmt_code and stmt_code in ran_before:
                return (
                    "NOTE: this exact statement ran without error before, so cash most "
                    "likely rebuilt it against incomplete state - your code is probably "
                    "fine unless a file it reads changed or a line it needs was removed. "
                    "To continue, run the cells "
                    "above it yourself (or Restart & Run All), and please report it"
                )
        except Exception:  # noqa: BLE001 - a diagnostic must never mask the error
            return None
        return None

    @staticmethod
    def _what_is_missing(exc: object) -> tuple[str, str] | None:
        """``('name', x)``, ``('key', k)`` or ``('attr', a)``: what the three
        errors a statement run against incomplete state raises are missing."""
        text = str(exc)
        if isinstance(exc, NameError) or "is not defined" in text:
            m = re.search(r"name '([^']+)' is not defined", text)
            return ("name", m.group(1)) if m else None
        if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], (str, int)):
            return ("key", str(exc.args[0]))
        if isinstance(exc, AttributeError):
            m = re.search(r"has no attribute '([^']+)'", text)
            return ("attr", m.group(1)) if m else None
        return None

    def _unscheduled_producer(
        self,
        kind: str,
        name: str,
        stmt_code: str | None,
        already_scheduled: list[str],
        notebook_cells: list[str] | None,
    ) -> tuple[str, int | None] | None:
        """The last statement above *stmt_code*, not re-run first, writing *name*."""
        scheduled = set(already_scheduled)
        reads: set[str] = set()
        if stmt_code:
            try:
                reads = {
                    n.id
                    for n in ast.walk(ast.parse(stmt_code))
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
            except SyntaxError:
                reads = set()
        if notebook_cells and stmt_code:
            found: tuple[str, int | None] | None = None
            for cell_idx, cell in enumerate(notebook_cells):
                tree = parse_cell_source(cell)
                if tree is None:
                    continue
                for node in tree.body:
                    code = ast.unparse(node)
                    if code == stmt_code:
                        return found
                    if code not in scheduled and self._writes(node, kind, name, reads):
                        found = (code, cell_idx + 1)
            return None
        if kind == "name":
            for code, outputs in self._known_producers():
                if name in outputs and code not in scheduled:
                    return code, None
        return None

    @staticmethod
    def _writes(node: ast.AST, kind: str, name: str, reads: set[str]) -> bool:
        """Whether *node* writes the missing name, or the missing key or
        attribute on a variable the failing statement reads."""
        for sub in ast.walk(node):
            if kind == "name":
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if sub.name == name:
                        return True
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    if any((a.asname or a.name.split(".")[0]) == name for a in sub.names):
                        return True
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store) and sub.id == name:
                    return True
                continue
            if not (isinstance(sub, (ast.Subscript, ast.Attribute)) and isinstance(sub.ctx, ast.Store)):
                continue
            if not (isinstance(sub.value, ast.Name) and sub.value.id in reads):
                continue
            if isinstance(sub, ast.Attribute):
                if sub.attr == name:
                    return True
            elif isinstance(sub.slice, ast.Constant):
                # ``df["week"] = ...`` is also what ``df.week`` reads.
                if str(sub.slice.value) == name:
                    return True
            elif kind == "key":
                return True  # ``results[name] = ...`` in the loop that fills it
        return False

    def _known_producers(self) -> list[tuple[str, set[str]]]:
        """``(statement code, names it assigns)`` for statements cash has seen.

        Read off ``executed_cell_codes``, which maps a variable to the statement
        that last produced it -- already maintained, so this costs nothing until
        something has actually failed.
        """
        pairs: dict[str, set[str]] = {}
        for var, code in (self.executed_cell_codes or {}).items():
            if isinstance(code, str):
                pairs.setdefault(code, set()).add(var)
        return list(pairs.items())

    @staticmethod
    def _format_upstream_failure(
        stmt_code: str,
        error_text: str,
        planning_gap: str | None = None,
    ) -> str:
        """One-line, embeddable message for an upstream statement failure.

        The executor re-raises this inside the user's cell via a generated
        ``raise ...('''<msg>''')`` statement, so the message must stay a
        single line and must not contain a triple quote.
        """
        stmt_short = stmt_code.split("\n")[0][:60]
        if len(stmt_code) > 60 or "\n" in stmt_code:
            stmt_short += "..."

        # "fix the upstream cell" is right for a statement that genuinely
        # raised, and actively misleading for a NameError. That one usually
        # means the cell simply has not run in THIS kernel -- there is nothing
        # to fix, and the remedy (run that cell) is the one thing the old text
        # never suggested. A round-14 tester lost time to exactly that: five
        # cells blocked at once, the message pointing at a cell they had just
        # read through and found nothing wrong with.
        advice = "fix the upstream cell and re-run"
        missing = re.search(r"NameError: name '([^']+)' is not defined", error_text)
        if planning_gap:
            # The note says what to do; "fix the upstream cell" beside it would
            # send the user to a cell with nothing wrong in it.
            advice = ""
        elif missing:
            advice = (
                f"run the cell that defines '{missing.group(1)}' - it has not "
                f"run in this kernel yet, so there may be nothing to fix"
            )

        msg = (
            f"Upstream statement {stmt_short!r} failed during auto-"
            f"re-execution: {error_text}. Cash stopped instead of running "
            f"this cell against stale upstream state" + (f" - {advice}." if advice else ".")
        )
        if planning_gap:
            msg = f"{msg} {planning_gap}."
        return msg.replace("'''", '"""').replace("\n", " ")

    def _try_parse_control_structure(self, code: str) -> ast.AST | None:
        """Parse code and return the AST node if it's a single control structure."""
        tree = parse_cached(code)
        if tree and len(tree.body) == 1 and is_control_structure(tree.body[0]):
            return tree.body[0]
        return None
