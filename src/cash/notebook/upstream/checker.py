from __future__ import annotations

import ast
import hashlib
import logging
import re
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from ...exceptions import AmbiguousCellError, UpstreamStateError
from ..server_discovery import get_notebook_cells, get_notebook_cells_with_ids
from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from ..analysis import CodeAnalyzer
from ..annotations import extract_annotations_for_statements
from ..cacheability import (
    alias_mutation_sources,
    aliased_sources,
    analyze_statement,
    crossref_reassigned_vars,
    subscript_view_bindings,
    function_arg_mutations,
    function_global_mutations,
    stateful_self_functions,
    stateful_closure_vars,
    partial_arg_mutations,
    mutating_partials,
    reduce_free_mutations,
    _called_function_names,
    standalone_call_arg_targets,
    standalone_method_mutation_receivers,
    selfref_inplace_write_vars,
)
from .simulator import (  # noqa: F401  re-exports for tests + downstream modules
    NotebookSimulator,
    _BUILTIN_NAMES,
    _FORWARD_PROBE_PLACEHOLDER,
    _IncrementalStartResult,
    _SimulationCacheEntry,
    _TraceEntry,
    _normalize_stmt,
)
from ..control_structures import is_control_structure

if TYPE_CHECKING:
    from ..statement import ProcessResult

__all__ = ["UpstreamChecker", "UpstreamResult"]


def _is_live_ndarray(val: Any) -> bool:
    """True if *val* is a numpy ndarray (so ``v = val[slice]`` is a view, not a
    copy). Duck-typed by type module/name to avoid importing numpy here (CAS-74)."""
    t = type(val)
    return t.__name__ == 'ndarray' and t.__module__.split('.')[0] == 'numpy'


class UpstreamResult(NamedTuple):
    """Result of upstream checking and re-execution."""
    metrics: list[ProcessResult]
    restore_time: float
    execution_time: float

logger = logging.getLogger(__name__)


def _nocache_written_vars(cell_code: str) -> set[str]:
    """Variables written by a ``# @cash: no-cache`` statement in *cell_code*.

    These must be excluded from the idempotent-rerun self-write restoration:
    ``no-cache`` means "always run fresh", so a self-modifying var under it
    (``counter = counter + 1``) is meant to ACCUMULATE on re-run, not be
    restored to its input. Returns an empty set when the cell has no
    annotations or can't be parsed.
    """
    try:
        annotations = extract_annotations_for_statements(cell_code)
        if not annotations:
            return set()
        tree = ast.parse(cell_code)
    except (SyntaxError, ValueError):
        return set()

    written: set[str] = set()
    for node in tree.body:
        ann = annotations.get(getattr(node, 'lineno', -1))
        if ann is None or not ann.no_cache:
            continue
        try:
            stmt_code = ast.unparse(node)
            _, outputs = CodeAnalyzer.analyze_code_block(stmt_code)
            written |= outputs
            written |= set(analyze_statement(stmt_code, None).all_mutated_vars)
        except (SyntaxError, ValueError):
            continue
    return written


class UpstreamChecker:
    """
    Manages detection and re-execution of changed upstream statements.

    Uses two complementary strategies:
    1. Lineage-based checking: Compares computed lineage hashes for already-executed variables
    2. Notebook-simulation checking: Simulates execution of notebook statements to detect code changes

    Attributes:
        shell: IPython shell instance
        debug: Enable debug output
        executed_cell_codes: Maps variable names to the statement code that defined them
        executed_cell_hashes: Maps variable names to hash of their defining statement code
        variable_lineage: Maps variable names to their lineage hash (includes input dependencies)
    """

    def __init__(self, shell: ShellProtocol, cash_instance: CashInstanceProtocol | None = None, debug: bool = False, compute_hash_fn: Callable[[Any], str] | None = None, tracking_state: TrackingState | None = None) -> None:
        self.shell: ShellProtocol = shell
        self.cash_instance: CashInstanceProtocol | None = cash_instance
        self.debug = debug
        self.compute_hash_fn: Callable[[Any], str] | None = compute_hash_fn
        self.function_tracker: Any | None = None

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
            debug=debug,
        )

    def reset_caches(self) -> None:
        """Clear simulation and AST caches.

        Should be called when switching notebooks (e.g., on %cash_on)
        to prevent stale simulation data from a previous notebook
        from interfering with the current one.
        """
        self.simulator.reset_caches()

    def _wire_state(self, state: TrackingState) -> None:
        """Internal: alias tracking dicts onto self so existing attribute
        accesses (``self.executed_cell_codes``, etc.) keep working.

        Kept as a separate method so ``set_tracking_state`` can also forward
        to the simulator.
        """
        self._tracking_state = state
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

    def _find_current_cell_index(self, cell_code: str, notebook_cells: list[str], cell_id: str | None = None, cells_with_ids: list[tuple[str, str]] = None) -> int | None:
        """Find the index of the current cell in the notebook.

        Uses a chain of matching strategies: ID match â†’ exact content â†’
        normalized newlines â†’ stripped whitespace.  Returns the first
        unambiguous match, or raises ``AmbiguousCellError`` when multiple
        cells share the same content and no cell ID is available.
        """
        # Strategy 1: Exact cell-ID match (available since IPython 8.3)
        if cell_id and cells_with_ids:
            for i, (nb_cell_id, _) in enumerate(cells_with_ids):
                if nb_cell_id == cell_id:
                    if self.debug:
                        logger.debug("[UPSTREAM_DEBUG] Found cell by ID match at index %s", i)
                    return i

            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Cell ID %s not found in notebook, falling back to content match", cell_id)

        # Strategies 2-4: content matching with progressive normalization
        content_matchers = [
            lambda cell: cell == cell_code,
            lambda cell: cell.replace('\r\n', '\n') == cell_code.replace('\r\n', '\n'),
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

        # Multiple matches with no resolvable cell ID â€” ambiguous
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Ambiguous cell content (matches=%s). Unable to safely determine upstream context.", matches)

        raise AmbiguousCellError(f"Ambiguous cell execution! The current cell content appears {len(matches)} times in the notebook and no cell ID could be resolved. Please ensure cells are unique or save the notebook.")

    def check_and_reexecute(
        self,
        cell_code: str,
        required_inputs: set[str],
        process_statement_callback: Callable[..., ProcessResult],
        global_ttl: int | None = None,
        cell_id: str | None = None,
        progress_callback: Callable[..., None] | None = None,
        control_structure_callback: Callable[..., Any] | None = None
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
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] check_and_reexecute called")
            logger.debug("[UPSTREAM_DEBUG]   cell_code: %s...", cell_code[:50])
            logger.debug("[UPSTREAM_DEBUG]   required_inputs: %s", required_inputs)
            logger.debug("[UPSTREAM_DEBUG]   current variable_lineage keys: %s", list(self.variable_lineage.keys()))
            if cell_id:
                logger.debug("[UPSTREAM_DEBUG]   cell_id: %s", cell_id)

        self._current_cell_id = cell_id
        self.simulator.set_current_cell_id(cell_id)
        # Keep simulator's function_tracker in sync. CashMagics sets
        # ``upstream_checker.function_tracker`` after construction (see
        # magics.py); we propagate it lazily so the simulator picks up the
        # latest reference.
        self.simulator._virtual_lineage.function_tracker = self.function_tracker

        # Phase 1 — Lineage-based staleness check (diagnostic-only).
        # Detects when variables are inconsistent with each other based on
        # their recorded lineage hashes. This phase only LOGS mismatches;
        # Phase 2 (``_check_notebook_based``) handles actual re-execution
        # with full cell ordering context. The notebook subsystem requires
        # a notebook file to function, so there is no "no-notebook"
        # re-execution path here. (The decorator path — ``@cash.cache`` —
        # does not use UpstreamChecker.)
        self._check_lineage_based(required_inputs)

        # Compute current cell outputs so Phase 2 can distinguish read-only inputs
        # from variables the current cell also writes (downstream-advancement case).
        # Also compute the names the cell REASSIGNS (`name = ...`), distinct from
        # in-place mutation, so Phase 2 can restore a stale self-reassigned input.
        # And the names the cell MUTATES in place (`lst.append`, `arr += 1`,
        # `d.update`) so Phase 2 can restore a no-lineage in-place accumulator.
        try:
            _, current_cell_outputs = CodeAnalyzer.analyze_code_block(cell_code)
            current_cell_reassigned = CodeAnalyzer.reassigned_names(cell_code)
            current_cell_mutated = set(analyze_statement(cell_code, None).all_mutated_vars)
            # A `# @cash: no-cache` statement opts out of caching AND of the
            # idempotent-rerun input restoration: its self-modifying vars must
            # accumulate on re-run (the documented "always recompute" contract),
            # not be reset to their input. Drop them from the self-write sets.
            nocache_vars = _nocache_written_vars(cell_code)
            current_cell_reassigned = current_cell_reassigned - nocache_vars
            current_cell_mutated = current_cell_mutated - nocache_vars
            # Receivers mutated in place by a bare method call (``b.items.append``).
            # Such no-output method statements skip the per-statement cache, so a
            # lineage-carrying receiver accumulates on an isolated re-run unless it
            # is restored to its cell-entry base. Scoped to METHOD receivers (NOT
            # subscript/attr writes) so ``df['col']=..`` keeps its per-statement
            # cache (CAS-42). Same no-cache opt-out as the other self-write sets.
            current_cell_method_receivers = set(
                standalone_method_mutation_receivers(ast.parse(cell_code))
            ) - nocache_vars
            # Self-referential in-place subscript/attr writes (``df['a']=df['a']*2``,
            # ``df['a']+=1``, ``df.iloc[i,j]+=x``) are non-idempotent: re-running
            # applies the op again, so a lineage-carrying receiver (DataFrame) must
            # be reset to its cell-entry base or the value accumulates (CAS-54).
            # New-column writes read from OTHER columns (``df['VolAdj']=...``) are
            # NOT self-referential and keep their per-statement cache (CAS-42). Same
            # no-cache opt-out (a no-cache self-write must advance, not reset -- CAS-51).
            current_cell_selfref_vars = set(
                selfref_inplace_write_vars(ast.parse(cell_code))
            ) - nocache_vars
            # A variable passed to a user-defined helper that mutates the
            # corresponding parameter in place (``def add(d): d.append(x)`` +
            # ``add(data)``) is mutated even though the cell never names the
            # mutation — static one-level body analysis attributes it back to the
            # argument so it resets on isolated re-run instead of accumulating
            # (CAS-58). Treated like a method receiver (force-reset + self-write).
            # Only resolve the (notebook-wide) function sources when the current
            # cell actually has a bare-Expr call candidate, so the common case
            # pays nothing.
            current_cell_stateful_funcs: set[str] = set()
            if standalone_call_arg_targets(ast.parse(cell_code)):
                func_sources = self._notebook_function_sources(cell_code)
                func_arg_muts = function_arg_mutations(
                    ast.parse(cell_code), func_sources.get
                ) - nocache_vars
                current_cell_mutated |= func_arg_muts
                current_cell_method_receivers |= func_arg_muts
                # A called function that mutates a module GLOBAL / free variable
                # in place (``def bump(): global g; g += 1`` + ``bump()``) leaves
                # the global accumulating on an isolated re-run because nothing in
                # the cell text names it. Attribute the mutation back to the
                # global and add it to the cell's inputs so the reset loop (which
                # iterates required_inputs) restores its producer's base (CAS-68 A).
                func_global_muts = function_global_mutations(
                    ast.parse(cell_code), func_sources.get
                ) - nocache_vars
                current_cell_mutated |= func_global_muts
                required_inputs = required_inputs | func_global_muts
            # A called function (captured OR bare) that carries mutable state on
            # its own object -- a mutated mutable-default arg (``def collect(x,
            # acc=[]): acc.append(x)``) or a function-attribute counter -- keeps
            # accumulating across calls. Force-reset the function so its ``def``
            # re-runs and recreates fresh state on an isolated re-run (CAS-68 B).
            if _called_function_names(ast.parse(cell_code)):
                func_sources_all = self._notebook_function_sources(cell_code)
                current_cell_stateful_funcs = set(
                    stateful_self_functions(ast.parse(cell_code), func_sources_all.get)
                ) - nocache_vars
                # A closure variable (``c = make_counter()``) whose factory returns
                # an inner function that mutates factory-local state accumulates
                # across calls; force-reset it so ``c = make_counter()`` re-runs
                # and rebuilds the closure fresh (CAS-68 B closure case).
                var_factories = self._notebook_var_factories(cell_code)

                def _resolve_var_factory(name, _fs=func_sources_all, _vf=var_factories):
                    src = _fs.get(_vf.get(name))
                    if not src:
                        return None
                    try:
                        parsed = ast.parse(src)
                    except (SyntaxError, ValueError):
                        return None
                    if parsed.body and isinstance(
                        parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        return parsed.body[0]
                    return None

                current_cell_stateful_funcs |= set(
                    stateful_closure_vars(ast.parse(cell_code), _resolve_var_factory)
                ) - nocache_vars
                # Hidden mutation through functools.partial (a bound mutable arg,
                # or the target's free var) or a function passed to functools.reduce
                # — the higher-order caller invokes it, so the mutation happens but
                # the cell text doesn't name it. Attribute it back and add to the
                # cell's inputs so its producer's base is restored (CAS-72).
                partial_bindings = self._notebook_partial_bindings(cell_code)
                hidden_muts = (
                    partial_arg_mutations(
                        ast.parse(cell_code), partial_bindings.get, func_sources_all.get
                    )
                    | reduce_free_mutations(ast.parse(cell_code), func_sources_all.get)
                ) - nocache_vars
                current_cell_mutated |= hidden_muts
                required_inputs = required_inputs | hidden_muts
                # A partial that binds a MUTATED arg captured the arg's OBJECT, so
                # resetting the arg name is not enough — force-reset the partial
                # too so its producer re-binds it to the fresh arg (CAS-72).
                current_cell_stateful_funcs |= set(
                    mutating_partials(
                        ast.parse(cell_code), partial_bindings.get, func_sources_all.get
                    )
                ) - nocache_vars
            # A bare ``y = x`` alias shares x's object, so an in-place mutation
            # through y (``y.append``/``y[0]+=1``) also mutates the upstream
            # holder x. Attribute it back to x so x resets on isolated re-run
            # instead of accumulating (CAS-60). For a no-lineage source the
            # content-base guard restores it via ``current_cell_mutated``; for a
            # lineage-carrying aliased DataFrame the source must also join the
            # selfref / method-receiver sets so the CAS-54/57 force-reset fires
            # (``df2 = df; df2['a'] = df2['a']*2`` doubled). The selfref set is
            # column-key-scoped, so an aliased NEW-column write (``df2['b'] =
            # df2['a']*2``) is still excluded and keeps its cache (CAS-42).
            alias_tree = ast.parse(cell_code)
            current_cell_mutated |= alias_mutation_sources(alias_tree) - nocache_vars
            current_cell_selfref_vars |= aliased_sources(
                alias_tree, current_cell_selfref_vars
            ) - nocache_vars
            current_cell_method_receivers |= aliased_sources(
                alias_tree, current_cell_method_receivers
            ) - nocache_vars
            # A numpy ``v = arr[slice]`` binding is a VIEW sharing arr's memory, so
            # mutating v (``v += 1``, ``v[i] = x``) mutates arr in place. A list
            # slice is a COPY, so this is gated on arr being a live ndarray at
            # runtime. When the view is mutated, attribute it back to arr so arr
            # resets on isolated re-run instead of accumulating (CAS-74).
            view_bindings = subscript_view_bindings(alias_tree)
            if view_bindings:
                user_ns = self.shell.user_ns
                mutated_here = analyze_statement(cell_code, None).all_mutated_vars
                current_cell_mutated |= {
                    base for alias, base in view_bindings.items()
                    if alias in mutated_here and _is_live_ndarray(user_ns.get(base))
                } - nocache_vars
            # Names reassigned from a permutation of their own prior values
            # (``a, b = b, a`` swap / rotate / temp-swap) read their pre-cell base
            # but on isolated re-run hold the swapped OUTPUT, whose content equals
            # the recorded output hash -- lineage-invisible, so no reset signal
            # fires. Force these to reset to their producers' base (CAS-65).
            current_cell_crossref_reassigned = crossref_reassigned_vars(alias_tree) - nocache_vars
            nocache_vars = set(nocache_vars)
        except (SyntaxError, ValueError):
            logger.debug("[UPSTREAM] Failed to analyze current cell outputs")
            current_cell_outputs = set()
            current_cell_reassigned = set()
            current_cell_mutated = set()
            current_cell_method_receivers = set()
            current_cell_selfref_vars = set()
            current_cell_crossref_reassigned = set()
            current_cell_stateful_funcs = set()
            nocache_vars = set()

        # Phase 2 â€” Notebook-simulation-based staleness check (disk vs. memory).
        # Simulates the notebook statement-by-statement and compares the resulting
        # virtual lineage against the actual in-memory state to find changed code.
        all_metrics, total_restore_time, total_execution_time = self._check_notebook_based(
            cell_code, required_inputs, process_statement_callback, global_ttl,
            current_cell_outputs=current_cell_outputs,
            current_cell_reassigned=current_cell_reassigned,
            current_cell_mutated=current_cell_mutated,
            current_cell_method_receivers=current_cell_method_receivers,
            current_cell_selfref_vars=current_cell_selfref_vars,
            current_cell_crossref_reassigned=current_cell_crossref_reassigned,
            current_cell_stateful_funcs=current_cell_stateful_funcs,
            current_cell_nocache_vars=nocache_vars,
            progress_callback=progress_callback,
            control_structure_callback=control_structure_callback,
        )

        return UpstreamResult(all_metrics, total_restore_time, total_execution_time)

    def _compute_expected_var_lineage(
        self,
        var_name: str,
        last_executed_code: str,
    ) -> str | None:
        """Compute expected lineage hash for *var_name* from its defining code.

        Returns ``None`` if the computation cannot be completed (e.g. the code
        is a control structure, or CodeAnalyzer fails).
        """
        try:
            tree = self.simulator.get_cached_ast(last_executed_code)
            if tree and len(tree.body) == 1 and isinstance(
                tree.body[0], (ast.For, ast.While, ast.If, ast.With, ast.Try)
            ):
                return None
        except (SyntaxError, ValueError, AttributeError):
            logger.debug("[UPSTREAM] Failed to parse AST for control-structure check: %s", var_name)

        stmt_inputs, _ = CodeAnalyzer.analyze_code_block(last_executed_code)

        if var_name in stmt_inputs:
            return None

        input_lineages = [
            self.variable_lineage[inp]
            for inp in stmt_inputs
            if inp in self.variable_lineage
        ]

        source_hash = hashlib.sha256(last_executed_code.encode('utf-8')).hexdigest()

        func_lineage_component = ""
        function_tracker = self.function_tracker if hasattr(self, 'function_tracker') else None
        if function_tracker is not None:
            try:
                func_source_hashes = function_tracker.get_callable_source_hashes(
                    stmt_inputs, self.shell.user_ns
                )
                if func_source_hashes:
                    func_parts = [f"{k}:{v}" for k, v in sorted(func_source_hashes.items())]
                    func_lineage_component = ":" + ":".join(func_parts)
            except (AttributeError, TypeError):
                pass

        expected_lineage_str = (
            f"{source_hash}:{':'.join(sorted(input_lineages))}"
            f"{func_lineage_component}"
        )
        return hashlib.sha256(expected_lineage_str.encode('utf-8')).hexdigest()

    def _notebook_function_sources(self, cell_code: str) -> dict[str, str]:
        """Map ``{function_name: source}`` for every top-level ``def`` across the
        notebook cells plus the current cell.

        Resolves from cell SOURCE (the source of truth) rather than
        ``inspect.getsource`` — the latter fails for cell-defined functions under
        nbclient (no linecache entry). Used by :func:`function_arg_mutations`
        (CAS-58). The current cell is included so a helper defined and used in the
        same cell still resolves; later same-name defs win (last definition).
        """
        sources: dict[str, str] = {}
        cells: list[str] = []
        try:
            from ..server_discovery import get_notebook_path
            path = get_notebook_path()
            if path:
                cells = get_notebook_cells(path) or []
        except (OSError, ValueError, RuntimeError):
            cells = []
        for code in (*cells, cell_code):
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    try:
                        sources[node.name] = ast.unparse(node)
                    except (ValueError, AttributeError):
                        continue
        return sources

    def _notebook_var_factories(self, cell_code: str) -> dict[str, str]:
        """Map ``{var: factory_name}`` for every top-level ``var = factory(...)``
        assignment across the notebook (``c = make_counter()`` → ``make_counter``).

        Used to resolve a closure variable back to the factory that produced it,
        so a stateful closure can be reset by re-running the factory call
        (CAS-68 B closure case). Last assignment wins.
        """
        factories: dict[str, str] = {}
        cells: list[str] = []
        try:
            from ..server_discovery import get_notebook_path
            path = get_notebook_path()
            if path:
                cells = get_notebook_cells(path) or []
        except (OSError, ValueError, RuntimeError):
            cells = []
        for code in (*cells, cell_code):
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)):
                    factories[node.targets[0].id] = node.value.func.id
        return factories

    def _notebook_partial_bindings(self, cell_code: str) -> dict[str, tuple[str, list]]:
        """Map ``{var: (target_func, [bound_arg_vars])}`` for every top-level
        ``var = partial(f, x, ...)`` / ``functools.partial(...)`` across the
        notebook (CAS-72). Bound args are Name ids (or ``None`` for non-Name),
        aligned to ``f``'s positional params. Last assignment wins.
        """
        bindings: dict[str, tuple[str, list]] = {}
        cells: list[str] = []
        try:
            from ..server_discovery import get_notebook_path
            path = get_notebook_path()
            if path:
                cells = get_notebook_cells(path) or []
        except (OSError, ValueError, RuntimeError):
            cells = []
        for code in (*cells, cell_code):
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Call)):
                    continue
                call = node.value
                func = call.func
                is_partial = ((isinstance(func, ast.Name) and func.id == 'partial')
                              or (isinstance(func, ast.Attribute) and func.attr == 'partial'))
                if is_partial and call.args and isinstance(call.args[0], ast.Name):
                    f_name = call.args[0].id
                    bound = [a.id if isinstance(a, ast.Name) else None for a in call.args[1:]]
                    bindings[node.targets[0].id] = (f_name, bound)
        return bindings

    def _check_lineage_based(self, required_inputs: set[str]) -> None:
        """Phase 1 — diagnostic-only lineage staleness check.

        Walks each input, recomputes its expected lineage from
        ``executed_cell_codes`` + current input lineages, and logs when the
        result differs from the stored value. Phase 2
        (:meth:`_check_notebook_based`) handles the actual re-execution
        decision with full notebook context.

        Why this phase is diagnostic-only: Pass 1 uses the CURRENT lineage of
        each input to compute expected output lineage. If an input was
        redefined by a LATER cell (variable shadowing), its current lineage
        reflects the later definition — not the version used when the output
        was originally computed. Re-executing with the wrong input would
        produce incorrect results that poison downstream computation. Phase 2's
        simulation tracks cell ordering and input lineages per-cell, so it
        handles both true staleness AND shadowing correctly.

        File and module lineage components are omitted here (Phase 2 owns
        those) — that may produce false-positive mismatch *logs* for vars with
        file/module deps, which is harmless.
        """
        for var_name in required_inputs:
            if var_name in _BUILTIN_NAMES:
                continue
            if var_name in self.vars_with_mutation_lineage:
                # Mutation-updated lineage is not derivable from executed_cell_codes.
                continue
            if var_name not in self.executed_cell_codes:
                continue
            if var_name not in self.variable_lineage:
                continue

            last_executed_code = self.executed_cell_codes[var_name]
            try:
                expected_lineage = self._compute_expected_var_lineage(
                    var_name, last_executed_code,
                )
                if expected_lineage is None:
                    continue
                current_lineage = self.variable_lineage[var_name]
                if expected_lineage != current_lineage:
                    logger.debug(
                        "[UPSTREAM] Variable '%s' has lineage mismatch "
                        "(expected=%s, actual=%s). Deferring to Phase 2.",
                        var_name, expected_lineage[:8], current_lineage[:8],
                    )
            except (KeyError, TypeError, ValueError, SyntaxError, AttributeError):
                logger.debug("[UPSTREAM] Error in lineage check for variable '%s'", var_name)

    def _resolve_fallback_cache_idx(self, cell_id: str | None) -> int | None:
        """Return the simulation cache index to use for the downstream advancement fallback.

        Returns ``None`` when no suitable cache entry can be found.
        """
        known_cell_idx: int | None = None
        if cell_id is not None:
            known_cell_idx = self.simulator.last_index_for_cell(cell_id)
        if known_cell_idx is None and self.last_cell_index is not None:
            known_cell_idx = self.last_cell_index

        if known_cell_idx is not None and known_cell_idx > 0:
            target_idx = known_cell_idx - 1
            if target_idx < self.simulator.simulation_cache_size():
                return target_idx
        return None

    def _reset_advanced_lineages(
        self,
        overlap_vars: set[str],
        cached_virtual_lineage: dict[str, str],
        cache_idx: int,
    ) -> None:
        """Reset in-memory lineages that are "ahead" of the cached virtual lineage."""
        if self.debug:
            logger.debug(
                "[UPSTREAM_DEBUG]   Downstream advancement fallback: "
                "overlap_vars=%s, cache_idx=%s, last_cell_index=%s",
                overlap_vars, cache_idx, self.last_cell_index,
            )
        for var_name in overlap_vars:
            if var_name not in cached_virtual_lineage or var_name not in self.variable_lineage:
                continue
            virtual_hash = cached_virtual_lineage[var_name]
            actual_hash = self.variable_lineage[var_name]
            if actual_hash != virtual_hash:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG]   -> Downstream advancement fallback: "
                        "resetting '%s' lineage from %s to virtual %s",
                        var_name, actual_hash[:8], virtual_hash[:8],
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
        if not (required_inputs and current_cell_outputs and self.simulator.simulation_cache_size()):
            return

        overlap_vars = required_inputs & current_cell_outputs
        if not overlap_vars:
            return

        cache_idx = self._resolve_fallback_cache_idx(cell_id)
        if cache_idx is None:
            return

        entry = self.simulator.simulation_cache_entry(cache_idx)
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
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Current cell not found in notebook")
            logger.debug("[UPSTREAM_DEBUG]   Looking for: %s...", cell_code.strip()[:60])
            for i, c in enumerate(notebook_cells[:5]):
                logger.debug("[UPSTREAM_DEBUG]   Cell %d: %s...", i, c.strip()[:60])

        # UNSAVED CELL UPSTREAM RESOLUTION
        missing_inputs: set[str] = set()
        if required_inputs:
            for inp in required_inputs:
                if inp in _BUILTIN_NAMES or inp.startswith('_'):
                    continue
                if inp not in self.shell.user_ns:
                    missing_inputs.add(inp)

        if missing_inputs and notebook_cells:
            if self.debug:
                logger.debug(
                    "[UPSTREAM_DEBUG]   Unsaved cell has missing inputs: %s", missing_inputs
                )
                logger.debug(
                    "[UPSTREAM_DEBUG]   Treating all %d saved cells as upstream",
                    len(notebook_cells),
                )
            return len(notebook_cells)

        # DOWNSTREAM ADVANCEMENT FALLBACK (unsaved cell, no missing inputs)
        self._handle_downstream_advancement_fallback(
            cell_id, required_inputs, current_cell_outputs
        )

        from ..server_discovery import invalidate_notebook_path_cache
        invalidate_notebook_path_cache()
        return None

    def _load_notebook_and_find_cell(
        self,
        cell_code: str,
        required_inputs: set[str],
        current_cell_outputs: set[str] | None,
    ) -> tuple[list[str] | None, int | None]:
        """Load the notebook and resolve the current cell index.

        Returns ``(notebook_cells, current_cell_idx)``.  If the notebook cannot
        be found or the cell index cannot be resolved, ``current_cell_idx`` may
        be ``None``, which the caller should treat as "return early with empty".
        ``notebook_cells`` is ``None`` when no notebook file exists.
        """
        from ..server_discovery import get_notebook_path
        notebook_path = get_notebook_path()
        notebook_cells = get_notebook_cells(notebook_path)

        if not notebook_cells:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] No notebook file found, skipping notebook check")
            return None, None

        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Found %d notebook cells", len(notebook_cells))

        cells_with_ids = get_notebook_cells_with_ids(notebook_path)
        cell_id = getattr(self, '_current_cell_id', None)

        current_cell_idx = self._resolve_current_cell_idx(
            cell_code, notebook_cells, cell_id, cells_with_ids,
            required_inputs, current_cell_outputs
        )
        if current_cell_idx is not None:
            self.last_cell_index = current_cell_idx
            if cell_id:
                self.simulator.record_cell_id_index(cell_id, current_cell_idx)

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
            return self._handle_unsaved_cell(
                cell_code, cell_id, required_inputs, current_cell_outputs, notebook_cells
            )
        return current_cell_idx

    def _evict_orphaned_definitions(self, notebook_cells: list[str], cell_code: str) -> None:
        """Evict variables whose producing definition no longer exists (CAS-62).

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
                _, outs = CodeAnalyzer.analyze_code_block(cell)
            except (SyntaxError, ValueError):
                return  # can't be sure what is produced — do nothing
            produced |= outs

        user_ns = self.shell.user_ns
        orphaned = {
            v for v in set(self.variable_lineage) - produced
            if v in user_ns
            and not v.startswith('_')
            and not isinstance(user_ns[v], types.ModuleType)
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

        state = self._tracking_state
        dict_attrs = (
            'variable_lineage', 'executed_input_lineages', 'current_session_hashes',
            'variable_hashes', 'variable_sources', 'executed_cell_codes',
            'executed_cell_hashes', 'executed_file_deps', 'executed_file_mtimes',
            'granular_preserved_vars', 'module_attribute_deps', 'from_import_sources',
        )
        for var in to_evict:
            self.shell.user_ns.pop(var, None)
            for attr in dict_attrs:
                getattr(state, attr, {}).pop(var, None)
            state.vars_with_mutation_lineage.discard(var)
            if self.debug:
                logger.debug("[UPSTREAM] CAS-62 evicted orphaned variable '%s'", var)

    def _sum_execution_times(self, executed_metrics: list) -> float:
        """Sum ``total_time`` from a list of metric dicts."""
        total = 0.0
        for metrics in executed_metrics:
            if metrics and 'total_time' in metrics:
                total += metrics['total_time']
        return total

    def _check_notebook_based(
        self,
        cell_code: str,
        required_inputs: set[str],
        process_statement_callback: Callable[..., ProcessResult],
        global_ttl: int | None,
        current_cell_outputs: set[str] | None = None,
        current_cell_reassigned: set[str] | None = None,
        current_cell_mutated: set[str] | None = None,
        current_cell_method_receivers: set[str] | None = None,
        current_cell_selfref_vars: set[str] | None = None,
        current_cell_crossref_reassigned: set[str] | None = None,
        current_cell_stateful_funcs: set[str] | None = None,
        current_cell_nocache_vars: set[str] | None = None,
        progress_callback: Callable[..., None] | None = None,
        control_structure_callback: Callable[..., Any] | None = None
    ) -> UpstreamResult:
        """
        Check if upstream notebook content differs from executed state.
        Simulates execution of all upstream statements.

        Returns:
            Tuple of (all_upstream_metrics, total_restore_time, total_execution_time)
        """
        try:
            notebook_cells, current_cell_idx = self._load_notebook_and_find_cell(
                cell_code, required_inputs, current_cell_outputs
            )
            if notebook_cells is None or current_cell_idx is None:
                return UpstreamResult([], 0.0, 0.0)

            # CAS-62: a variable whose definition was removed/renamed across an
            # edit is orphaned — no cell produces it anymore. Evict it (and its
            # transitive consumers) so they re-run from the start and raise
            # NameError like a fresh kernel, instead of serving a stale value.
            self._evict_orphaned_definitions(notebook_cells, cell_code)

            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Current cell found at index %s", current_cell_idx)
                logger.debug("[UPSTREAM_DEBUG] Will simulate %s upstream cells", current_cell_idx)

            statements_to_reexecute, restored_info, total_restore_time = self.simulator.simulate_upstream(
                current_cell_idx,
                notebook_cells,
                required_inputs,
                current_cell_outputs=current_cell_outputs,
                current_cell_reassigned=current_cell_reassigned,
                current_cell_mutated=current_cell_mutated,
                current_cell_method_receivers=current_cell_method_receivers,
                current_cell_selfref_vars=current_cell_selfref_vars,
                current_cell_crossref_reassigned=current_cell_crossref_reassigned,
                current_cell_stateful_funcs=current_cell_stateful_funcs,
                current_cell_nocache_vars=current_cell_nocache_vars
            )

            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Simulation result: %s stmts to re-execute, %s stmts restored from cache",
                      len(statements_to_reexecute), len(restored_info))

            executed_metrics = []
            if statements_to_reexecute:
                executed_metrics = self._reexecute_statements(
                    statements_to_reexecute, process_statement_callback, global_ttl,
                    progress_callback=progress_callback,
                    restored_info=restored_info,
                    control_structure_callback=control_structure_callback,
                )
            total_execution_time = self._sum_execution_times(executed_metrics)

            # CRITICAL: Sync simulation cache with actual runtime lineages.
            # After upstream statements execute (or skip/restore), variable_lineage
            # holds the authoritative lineage for each variable.  The simulation
            # cache may store stale virtual_lineage values from an earlier run
            # where forward propagation failed (e.g., fallback lineage computed
            # differently than runtime lineage).  Without this sync, subsequent
            # re-executions will always see a lineage mismatch and trigger
            # unnecessary upstream restoration.
            self._sync_simulation_cache_lineages()

            all_metrics = restored_info + executed_metrics

            all_metrics.sort(key=lambda m: m.get('position', 999999))

            return UpstreamResult(all_metrics, total_restore_time, total_execution_time)

        except (RuntimeError, SyntaxError):
            raise
        except (KeyError, TypeError, ValueError, OSError) as e:
            logger.debug("[UPSTREAM] Error in notebook-based checking: %s", e)
            raise UpstreamStateError(
                f"Failed to restore or simulate upstream state: {e}"
            ) from e

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
        normalized_code = re.sub(r'# __iteration_context__:.*?\n', '', producing_code).strip()
        if normalized_code not in cumulative_stmt_codes:
            if self.debug:
                logger.debug(
                    "[UPSTREAM_DEBUG] Skipping sync for '%s' in cache entry %d: "
                    "producing code not in cells 0..%d",
                    var_name, idx, idx,
                )
            return False
        return True

    def _sync_simulation_cache_lineages(self) -> None:
        """Sync simulation cache virtual lineages with actual runtime lineages.

        After upstream statements are executed/restored/skipped, ``variable_lineage``
        holds the authoritative lineage for each variable.  The simulation cache
        may store stale ``virtual_lineage`` values from an earlier run where
        forward propagation failed (e.g., the fallback lineage computed
        differently than the runtime lineage because a control structure was
        simulated as a single unit, or ``inspect.getsource`` returned different
        results).

        This method patches every cached ``virtual_lineage`` snapshot so that
        variables get their lineage updated to the authoritative value â€” but
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
        already matches the mutated actual lineage â†’ no restoration â†’ bug.

        With scoping, we check ``executed_cell_codes['df']`` to see which
        statement last produced ``df``'s runtime lineage.  If that statement
        is ``df['SMA'] = ...`` (from cell 5), it won't be found in cells
        0..2's trace segments, so cell 2's cache entry is NOT synced for
        ``df``.
        """
        if not self.simulator.simulation_cache_size():
            return

        updated = False
        # For each cache entry at index idx, collect ALL statement codes that
        # appear in the trace segments of cells 0..idx.  We only sync a
        # variable's lineage if the code that produced the current runtime
        # lineage (from executed_cell_codes) is among these statements.
        cumulative_stmt_codes = set()
        for idx in range(self.simulator.simulation_cache_size()):
            entry = self.simulator.simulation_cache_entry(idx)
            if entry is None:
                continue
            cell_trace = entry.trace_segment
            for trace_entry in cell_trace:
                # trace_entry format: (stmt_code, outputs, inputs, input_hashes, produced_lineages, files_stale)
                if len(trace_entry) >= 1:
                    cumulative_stmt_codes.add(trace_entry[0])  # stmt_code

            cached_vl = entry.virtual_lineage
            for var_name in list(cached_vl.keys()):
                if not self._should_sync_cache_var(var_name, cumulative_stmt_codes, cached_vl, idx):
                    continue
                # Safe to sync: the runtime lineage was produced by code within
                # cells 0..idx, so this is a valid forward-propagation correction.
                cached_vl[var_name] = self.variable_lineage[var_name]
                updated = True

        if updated and self.debug:
            logger.debug("[UPSTREAM_DEBUG] Synced simulation cache lineages with runtime state (scoped to producing code)")

    def _reexecute_statements(
        self,
        statements: list[str],
        process_callback: Callable[..., ProcessResult],
        global_ttl: int | None,
        progress_callback: Callable[..., None] | None = None,
        restored_info: list[ProcessResult] | None = None,
        control_structure_callback: Callable[..., Any] | None = None
    ) -> list[ProcessResult]:
        """Re-execute a list of statements and return their metrics."""
        executed_metrics = []
        total_upstream_steps = len(statements)

        for stmt_idx, stmt_code in enumerate(statements):
            # The badge's "Upstream" section is the canonical user-facing
            # signal that upstream re-execution happened — so we stay quiet
            # on stdout in normal use. Debug mode emits both a logger entry
            # and a stdout line for troubleshooting and integration tests.
            if self.debug:
                logger.debug("[UPSTREAM] Auto-executing: %s...", stmt_code[:50])
                stmt_short = stmt_code.split('\n')[0][:40]
                if len(stmt_code) > 40:
                    stmt_short += "..."
                print(f"Cash: Auto-executing upstream statement: {stmt_short}")

            try:
                # Check if this is a control structure (for/if/try/with/while).
                # If so, delegate to the control structure callback which handles
                # per-iteration caching for loops instead of executing monolithically.
                ctrl_node = self._try_parse_control_structure(stmt_code)
                if ctrl_node is not None and control_structure_callback is not None:
                    if self.debug:
                        logger.debug("[UPSTREAM] Delegating control structure to per-iteration processor")
                    ctrl_result = control_structure_callback(ctrl_node, ttl=global_ttl, silent=True)
                    for m in ctrl_result.metrics:
                        if m:
                            m['is_upstream'] = True
                            executed_metrics.append(m)
                    if not ctrl_result.success:
                        raise ctrl_result.error or RuntimeError("Error in upstream control structure")
                else:
                    result = process_callback(stmt_code, global_ttl, silent=False)
                    if self.debug:
                        logger.debug("[UPSTREAM] Callback result for '%s...': %s", stmt_code[:20], result)
                    if result:
                        result['is_upstream'] = True  # Mark as upstream so badge categorizes correctly
                        executed_metrics.append(result)
            except (RuntimeError, NameError, KeyError, TypeError, ValueError) as e:
                logger.error("[ERROR] Failed to auto-execute statement: %s", e)

            # Report progress after each statement
            if progress_callback is not None:
                try:
                    all_so_far = (restored_info or []) + executed_metrics
                    progress_callback(all_so_far, stmt_code, stmt_idx + 1, total_upstream_steps)
                except (TypeError, ValueError, AttributeError):
                    pass  # Don't let progress reporting break execution

        return executed_metrics

    def _try_parse_control_structure(self, code: str) -> ast.AST | None:
        """Parse code and return the AST node if it's a single control structure."""
        tree = self.simulator.get_cached_ast(code)
        if tree and len(tree.body) == 1 and is_control_structure(tree.body[0]):
            return tree.body[0]
        return None

