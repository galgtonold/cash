from __future__ import annotations

import ast
import hashlib
import logging
import re
import types
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from ...exceptions import AmbiguousCellError, CashUpstreamSyntaxWarning, UpstreamStateError
from ..server_discovery import get_notebook_cells, get_notebook_cells_with_ids
from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from ..analysis import CodeAnalyzer
from ..annotations import extract_annotations_for_statements, parse_annotation_line
from ..randomness import (
    get_drawing_rng_modules,
    get_seeding_rng_modules,
    restore_rng_state,
    rng_lineage_fingerprint,
    seed_cells_not_yet_run,
)
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
    object_protocol_mutations,
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

        # CAS-173: per-session ledger of already-warned broken upstream cells,
        # keyed by cell index -> cell source hash. Keeps the "cell N has a
        # syntax error" warning to once per distinct break (not once per
        # downstream cell run) while still re-warning when the break changes or
        # a fixed cell is broken again.
        self._warned_broken_cells: dict[int, str] = {}

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
        # Re-arm the broken-upstream-cell warning for the new notebook (CAS-173):
        # its cell indices/hashes are meaningless across a notebook switch.
        self._warned_broken_cells.clear()

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

        # Resolve the notebook path ONCE for the whole cell check (CAS-150) and
        # thread it through the analysis helpers + Phase 2, instead of each site
        # re-running discovery. The negative cache in server_discovery bounds a
        # failed probe, and this collapses the success case to a single resolve.
        notebook_path = self._resolve_notebook_path()

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
                func_sources = self._notebook_function_sources(cell_code, notebook_path)
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
                func_sources_all = self._notebook_function_sources(cell_code, notebook_path)
                current_cell_stateful_funcs = set(
                    stateful_self_functions(ast.parse(cell_code), func_sources_all.get)
                ) - nocache_vars
                # A closure variable (``c = make_counter()``) whose factory returns
                # an inner function that mutates factory-local state accumulates
                # across calls; force-reset it so ``c = make_counter()`` re-runs
                # and rebuilds the closure fresh (CAS-68 B closure case).
                var_factories = self._notebook_var_factories(cell_code, notebook_path)

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
                partial_bindings = self._notebook_partial_bindings(cell_code, notebook_path)
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
            # Object-protocol hidden state (CAS-69/70/71/73): a with-statement, a
            # custom-dunder op (``s[k]=v`` / ``del s[k]`` / ``v=s[k]`` / ``a(x)``),
            # a constructor, a decorated call, or an instance / class method whose
            # body mutates hidden state. Resolve the class / wrapper / context
            # manager, analyse the invoked method, and route the mutation to the
            # matching reset channel: a free var (mutated + inputs, CAS-68 A), the
            # receiver instance (method-receiver), or the class def (stateful
            # re-run).
            op_tree = ast.parse(cell_code)
            # Cheap current-cell trigger: any call / subscript / del / with can
            # dispatch to a custom method whose body hides a mutation. The
            # analysis is a no-op when nothing resolves to a notebook class /
            # decorated function, so this only gates the (memoised) source scans.
            has_op_trigger = any(
                isinstance(n, (ast.Call, ast.Subscript, ast.Delete,
                               ast.With, ast.AsyncWith))
                for n in ast.walk(op_tree)
            )
            # A top-level ``class Sub(Base):`` defining a subclass triggers the
            # base's ``__init_subclass__`` hook during CLASS CREATION — a hidden
            # mutation with no Call/Subscript/etc. node of its own — so gate the
            # object-protocol scan on a based class def too (CAS-103).
            has_op_trigger = has_op_trigger or any(
                isinstance(n, ast.ClassDef) and n.bases
                for n in op_tree.body
            )
            if has_op_trigger:
                class_sources = self._notebook_class_sources(cell_code, notebook_path)
                op_func_sources = self._notebook_function_sources(cell_code, notebook_path)
                op_var_factories = self._notebook_var_factories(cell_code, notebook_path)
                # ``@ClassName def task`` binds ``task`` to a ClassName INSTANCE
                # (a class-based decorator), and ``h = g`` aliases one name to
                # another; both feed the object-protocol resolvers (CAS-80).
                op_decorated_instances = self._notebook_decorated_instances(cell_code, notebook_path)
                op_name_aliases = self._notebook_name_aliases(cell_code, notebook_path)

                def _resolve_alias(name, _al=op_name_aliases):
                    # follow ``h = g = ...`` chains (bounded by the alias count)
                    for _ in range(len(_al) + 1):
                        nxt = _al.get(name)
                        if nxt is None:
                            return name
                        name = nxt
                    return name

                def _instance_class(var, _vf=op_var_factories, _cs=class_sources):
                    cls = _vf.get(var)
                    return cls if cls in _cs else None

                def _decorated_class(var, _di=op_decorated_instances, _cs=class_sources):
                    cls = _di.get(var)
                    return cls if cls in _cs else None

                def _op_var_factory(name, _fs=op_func_sources, _vf=op_var_factories):
                    name = _resolve_alias(name)
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

                op_resets = object_protocol_mutations(
                    op_tree, class_sources.get, _instance_class,
                    op_func_sources.get, _op_var_factory,
                    decorated_class=_decorated_class,
                )
                # The free-var channel resets via the CAS-68 A content-base path,
                # which already suppresses cross-cell accumulators — apply it
                # directly.
                op_free = op_resets.free_vars - nocache_vars
                current_cell_mutated |= op_free
                required_inputs = required_inputs | op_free
                # The receiver and class-def channels re-derive the object
                # (restore the receiver to its cell-entry base / re-run the class
                # def). That is safe ONLY when the current cell is the SOLE
                # in-place mutator: if ANOTHER cell also mutates the same receiver
                # (``dag.add_edge(..)`` in a setup cell builds ``dag`` up across a
                # loop, then ``dag.topo_sort()`` here) or class var (``w0 =
                # Widget()`` then ``w = Widget()`` — both bump ``Widget.count``),
                # the re-derivation loses the other cell's contribution and would
                # corrupt EVEN THE FIRST run. Suppress those — a cross-cell
                # accumulator needs a per-cell-base snapshot this reset cannot
                # provide. The FILED CAS-69/70/71/73 forms construct the receiver
                # once (no cross-cell in-place build) and are unaffected. [CAS-75]
                op_receivers = op_resets.receivers - nocache_vars
                op_class_defs = op_resets.class_defs - nocache_vars
                # A base ``__init_subclass__`` free-var registry (CAS-103) hides
                # its mutation behind class creation, so — unlike an ordinary
                # CAS-68 free var — the simulator's content-base guard cannot see
                # that a SIBLING subclass cell also appends to it. Guard it with
                # the same cross-cell suppression as the receiver / class-def
                # channels: if another cell also registers into it, this cell is
                # not the sole mutator and the free-var reset would drop that
                # cell's contribution on run_all.
                op_init_free = op_resets.init_subclass_free_vars - nocache_vars
                if op_receivers or op_class_defs or op_init_free:
                    other_receivers: set[str] = set()
                    other_class_defs: set[str] = set()
                    other_init_free: set[str] = set()
                    seen_current = False
                    for _code in self._other_notebook_cells(cell_code, notebook_path):
                        if not seen_current and _code == cell_code:
                            seen_current = True
                            continue
                        try:
                            other = object_protocol_mutations(
                                ast.parse(_code), class_sources.get,
                                _instance_class, op_func_sources.get, _op_var_factory,
                            )
                        except (SyntaxError, ValueError):
                            continue
                        other_receivers |= other.receivers
                        other_class_defs |= other.class_defs
                        other_init_free |= other.init_subclass_free_vars
                    op_receivers = op_receivers - other_receivers
                    op_class_defs = op_class_defs - other_class_defs
                    op_init_free = op_init_free - other_init_free
                current_cell_mutated |= op_receivers
                current_cell_method_receivers |= op_receivers
                current_cell_stateful_funcs |= op_class_defs
                # Survivors (sole-mutator subclass cells) join the free-var reset.
                current_cell_mutated |= op_init_free
                required_inputs = required_inputs | op_init_free
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
            notebook_path=notebook_path,
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

    def _resolve_notebook_path(self) -> str | None:
        """Resolve the current notebook path once per cell's upstream check.

        Single choke point for path discovery (CAS-150): the per-statement /
        per-upstream-cell analysis helpers used to each call ``get_notebook_path``
        independently, so one cell re-probed discovery 5-15 times — catastrophic
        when a stale Jupyter runtime makes each probe block on a network timeout.
        Resolving once and threading the result collapses that to a single probe.

        Also emits the once-per-session "upstream tracking disabled" advisory when
        discovery fails, so a user knows cash's headline feature is off rather
        than inferring it from silently-stale results.
        """
        from ..server_discovery import (
            get_notebook_path,
            warn_notebook_not_found_once,
        )
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

    def _notebook_function_sources(self, cell_code: str, notebook_path: str | None) -> dict[str, str]:
        """Map ``{function_name: source}`` for every top-level ``def`` across the
        notebook cells plus the current cell.

        Resolves from cell SOURCE (the source of truth) rather than
        ``inspect.getsource`` — the latter fails for cell-defined functions under
        nbclient (no linecache entry). Used by :func:`function_arg_mutations`
        (CAS-58). The current cell is included so a helper defined and used in the
        same cell still resolves; later same-name defs win (last definition).
        """
        sources: dict[str, str] = {}
        cells = self._notebook_cells_for(notebook_path)
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

    def _other_notebook_cells(self, cell_code: str, notebook_path: str | None) -> list[str]:
        """The notebook's cell sources (which already include the current cell;
        the caller skips its first occurrence of *cell_code*). Used to detect a
        class variable mutated by more than one cell — a cross-cell accumulator
        whose class-def reset must be suppressed (CAS-73 / CAS-75)."""
        return self._notebook_cells_for(notebook_path)

    def _notebook_class_sources(self, cell_code: str, notebook_path: str | None) -> dict[str, str]:
        """Map ``{class_name: source}`` for every top-level ``class`` across the
        notebook cells plus the current cell.

        Resolves from cell SOURCE (like :meth:`_notebook_function_sources`) so a
        class defined in a cell resolves under nbclient, where
        ``inspect.getsource`` has no linecache entry. Used by
        :func:`object_protocol_mutations` to analyse a receiver's method /
        dunder bodies (CAS-69/70/71/73). Later same-name defs win.
        """
        sources: dict[str, str] = {}
        cells = self._notebook_cells_for(notebook_path)
        for code in (*cells, cell_code):
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    try:
                        sources[node.name] = ast.unparse(node)
                    except (ValueError, AttributeError):
                        continue
        return sources

    def _notebook_decorated_instances(self, cell_code: str, notebook_path: str | None) -> dict[str, str]:
        """Map ``{func_name: decorator_name}`` for every top-level ``@Deco def f``
        across the notebook (CAS-80). When the decorator is a class, ``f`` is an
        INSTANCE of it (a class-based decorator: ``@Counter def task`` → ``task``
        is a ``Counter``). The caller filters to decorators that resolve to a
        class. Only bare-``Name`` decorators are captured. Last definition wins."""
        instances: dict[str, str] = {}
        cells = self._notebook_cells_for(notebook_path)
        for code in (*cells, cell_code):
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Name):
                            instances[node.name] = dec.id
                            break
        return instances

    def _notebook_name_aliases(self, cell_code: str, notebook_path: str | None) -> dict[str, str]:
        """Map ``{alias: source}`` for every top-level ``alias = source`` bare-Name
        binding across the notebook (``h = g``), so a called alias resolves to the
        decorator/factory of the name it aliases (CAS-80). Last binding wins."""
        aliases: dict[str, str] = {}
        cells = self._notebook_cells_for(notebook_path)
        for code in (*cells, cell_code):
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            for node in tree.body:
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Name)):
                    aliases[node.targets[0].id] = node.value.id
        return aliases

    def _notebook_var_factories(self, cell_code: str, notebook_path: str | None) -> dict[str, str]:
        """Map ``{var: factory_name}`` for every top-level ``var = factory(...)``
        assignment across the notebook (``c = make_counter()`` → ``make_counter``).

        Used to resolve a closure variable back to the factory that produced it,
        so a stateful closure can be reset by re-running the factory call
        (CAS-68 B closure case). Last assignment wins.
        """
        factories: dict[str, str] = {}
        cells = self._notebook_cells_for(notebook_path)
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

    def _notebook_partial_bindings(self, cell_code: str, notebook_path: str | None) -> dict[str, tuple[str, list]]:
        """Map ``{var: (target_func, [bound_arg_vars])}`` for every top-level
        ``var = partial(f, x, ...)`` / ``functools.partial(...)`` across the
        notebook (CAS-72). Bound args are Name ids (or ``None`` for non-Name),
        aligned to ``f``'s positional params. Last assignment wins.
        """
        bindings: dict[str, tuple[str, list]] = {}
        cells = self._notebook_cells_for(notebook_path)
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
        notebook_path: str | None,
    ) -> tuple[list[str] | None, int | None]:
        """Load the notebook and resolve the current cell index.

        ``notebook_path`` is the path resolved once at the start of the cell's
        upstream check (CAS-150) and threaded here, so Phase 2 does not re-run
        discovery.  Returns ``(notebook_cells, current_cell_idx)``.  If the
        notebook cannot be found or the cell index cannot be resolved,
        ``current_cell_idx`` may be ``None``, which the caller should treat as
        "return early with empty".  ``notebook_cells`` is ``None`` when no
        notebook file exists.
        """
        # Pass the (possibly None) resolved path straight through: when it is
        # None, get_notebook_cells re-resolves internally, but that lookup is
        # bounded by the negative cache (CAS-150), so it stays cheap while
        # preserving the get_notebook_cells seam that callers/tests patch.
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

    def _warn_broken_upstream_cells(
        self,
        notebook_cells: list[str],
        current_cell_idx: int,
    ) -> set[int]:
        """Emit a visible warning for any UPSTREAM cell that cannot be parsed.

        A half-written cell the user has SAVED but not run makes the upstream
        simulator SKIP that cell (see ``VirtualLineage._simulate_one_cell``,
        CAS-173) so unrelated downstream cells keep caching. But the user must
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
        multi-line ``%``-format print of CAS-163 included — is never falsely
        flagged. Returns the set of broken cell indices (0-based).
        """
        broken: dict[int, str] = {}
        for idx in range(min(current_cell_idx, len(notebook_cells))):
            raw = notebook_cells[idx]
            try:
                clean = CodeAnalyzer.strip_magics(raw.replace('\r\n', '\n'))
            except (ValueError, TypeError):
                continue
            if not clean.strip():
                continue
            try:
                ast.parse(clean)
            except SyntaxError:
                broken[idx] = hashlib.sha256(raw.encode('utf-8')).hexdigest()

        for idx, cell_hash in broken.items():
            if self._warned_broken_cells.get(idx) == cell_hash:
                continue  # already warned about this exact break — stay quiet
            raw = notebook_cells[idx]
            snippet = next((ln.strip() for ln in raw.splitlines() if ln.strip()), '')
            if len(snippet) > 60:
                snippet = snippet[:57] + '...'
            message = (
                f"Cash: cell {idx + 1} has a syntax error and could not be "
                f"parsed ({snippet!r}). Caching continues for cells that do "
                f"not depend on it, but cell {idx + 1} will not run and any "
                f"cell that depends on it can no longer be dependency-tracked "
                f"until you fix it."
            )
            logger.warning(message)
            # warn_explicit with registry=None bypasses the "once per location"
            # __warningregistry__ dedupe (every break is raised from this one
            # line); our own per-(idx, hash) ledger supplies the dedupe we
            # actually want, and this still consults the user's filters. Mirrors
            # randomness.py's established pattern.
            warnings.warn_explicit(
                message, CashUpstreamSyntaxWarning,
                filename='<cash>', lineno=idx + 1, registry=None,
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
            if metrics and 'total_time' in metrics:
                total += metrics['total_time']
        return total

    def _check_notebook_based(
        self,
        cell_code: str,
        required_inputs: set[str],
        process_statement_callback: Callable[..., ProcessResult],
        global_ttl: int | None,
        notebook_path: str | None = None,
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
                cell_code, required_inputs, current_cell_outputs, notebook_path
            )
            if notebook_cells is None or current_cell_idx is None:
                return UpstreamResult([], 0.0, 0.0)

            # CAS-173: disclose any unparseable UPSTREAM cell BEFORE simulating.
            # The simulator skips such a cell (VirtualLineage._simulate_one_cell)
            # so unrelated downstream cells keep caching, but the user must be
            # told which cell is broken — otherwise caching degrades silently
            # mid-edit while the badge and auto_cache_enabled still say it is on.
            self._warn_broken_upstream_cells(notebook_cells, current_cell_idx)

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

            # ADR-017 / CAS-225: a bare ``np.random.seed(N)`` binds no variable,
            # so the simulator never links it to a downstream draw. If the
            # current cell draws and an upstream seed cell was edited but not
            # re-run, re-execute that seed cell first — its side effect (the new
            # seed) must be in place before the draw, and re-running a seed is
            # idempotent. The unrun guard keeps warm draws untouched (an
            # unchanged seed cell's source is already in the executed set).
            # Snapshot before the RNG prepends so we can tell which statements
            # were pulled in solely to re-establish the random stream, and label
            # them for the badge (Stage 2 of the randomness UX).
            _before_rng_prepend = set(statements_to_reexecute)
            statements_to_reexecute = self._prepend_stale_seed_cells(
                cell_code, notebook_cells, statements_to_reexecute, current_cell_idx,
            )

            # A draw re-executed because an ORDINARY input changed (not the seed)
            # must still run from its top-to-bottom stream position; re-establish
            # its upstream RNG chain when the seed itself isn't being re-run.
            statements_to_reexecute = self._prepend_rng_chain_for_reexecuted_draws(
                notebook_cells, statements_to_reexecute, current_cell_idx,
            )
            rng_rerun = {
                s for s in statements_to_reexecute
                if s not in _before_rng_prepend and self._cell_touches_rng(s)
            }

            executed_metrics = []
            if statements_to_reexecute:
                executed_metrics = self._reexecute_statements(
                    statements_to_reexecute, process_statement_callback, global_ttl,
                    progress_callback=progress_callback,
                    restored_info=restored_info,
                    control_structure_callback=control_structure_callback,
                )
            self._label_rng_rerun_metrics(executed_metrics, rng_rerun)
            total_execution_time = self._sum_execution_times(executed_metrics)

            # CAS-226 / CAS-227 (ADR-018): restore the position-correct RNG state
            # right before the current draw runs, so a re-executed draw continues
            # from the stream position (and under the seed) it holds top-to-bottom
            # rather than wherever the live state was last left. Runs after any
            # upstream re-execution above, so it is the last thing to touch the
            # RNG before the cell.
            self._restore_position_rng_state(cell_code, notebook_cells, current_cell_idx)

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

    def _prepend_stale_seed_cells(
        self, cell_code: str, notebook_cells: list[str], statements: list[str],
        current_cell_idx: int | None,
    ) -> list[str]:
        """Schedule an edited-but-not-rerun seed cell ahead of a draw (ADR-017).

        When the current cell draws from a global RNG and an UPSTREAM cell seeds
        that module with source that has not executed this session (an edited or
        never-run seed), that seed's side effect must be re-established before the
        draw (CAS-225). But re-seeding alone is not enough when draws sit between
        the seed and the current cell: those intervening draws advanced the stream
        under the OLD seed, so the current draw would run from the new seed's
        position 0 instead of the position the chain holds top-to-bottom
        (CAS-226/227 combined case). So when a seed is stale, re-run the whole RNG
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
            executed = self._tracking_state.executed_cell_source_hashes
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
                if not self._cell_touches_rng(src):
                    continue
                if src not in already and src not in prepend:
                    prepend.append(src)
            if prepend and self.debug:
                logger.debug("[UPSTREAM] Rebuilding RNG chain (%d cells) before draw", len(prepend))
            return prepend + statements
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return statements

    def _prepend_rng_chain_for_reexecuted_draws(
        self, notebook_cells: list[str], statements: list[str], current_cell_idx: int | None,
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

            # Every upstream RNG statement touching a missing module, in source
            # order, tagged with whether it draws.
            ordered: list[tuple[str, bool]] = []
            for cell in upstream:
                try:
                    tree = ast.parse(cell)
                except SyntaxError:
                    continue
                for node in tree.body:
                    try:
                        stmt = ast.unparse(node)
                    except (ValueError, TypeError):
                        continue
                    draws = get_drawing_rng_modules(stmt) & missing
                    seeds = get_seeding_rng_modules(stmt) & missing
                    if draws or seeds:
                        ordered.append((stmt, bool(draws)))

            # The earliest re-executed draw is the boundary: everything strictly
            # before it re-runs to advance the stream; it and everything after
            # stay in the plan.
            boundary = next(
                (i for i, (stmt, is_draw) in enumerate(ordered)
                 if is_draw and stmt in already),
                None,
            )
            if boundary is None:
                return statements

            prepend: list[str] = []
            for stmt, _is_draw in ordered[:boundary]:
                if stmt not in already and stmt not in prepend:
                    prepend.append(stmt)
            if not prepend:
                return statements
            if self.debug:
                logger.debug(
                    "[UPSTREAM] Re-establishing RNG chain (%d stmts) before a re-executed draw",
                    len(prepend),
                )
            return prepend + statements
        except (AttributeError, IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return statements

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
                if m.get('code', '').strip() in wanted:
                    m['miss_reason'] = "re-run to restore the random stream"
            except (AttributeError, TypeError):  # pragma: no cover - defensive
                continue

    def _current_cell_drawing_modules(self, cell_code: str) -> set[str]:
        """RNG modules this cell draws from — statically OR by prior observation.

        Static analysis sees ``np.random.rand()`` in the cell; the observed set
        (ADR-018) adds modules a call like ``model.fit()`` changed at runtime, so
        an indirect draw is treated like a direct one on re-run.
        """
        modules = set(get_drawing_rng_modules(cell_code))
        digest = hashlib.sha256(cell_code.encode('utf-8')).hexdigest()
        modules |= self._tracking_state.observed_rng_cells.get(digest, set())
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
        for line in cell_code.splitlines():
            if not line.strip().startswith('#'):
                continue
            ann = parse_annotation_line(line)
            if ann is not None and ann.no_cache:
                return True
        return False

    def _cell_touches_rng(self, src: str) -> bool:
        """True if *src* seeds or draws — statically or by prior observation."""
        if get_seeding_rng_modules(src) or get_drawing_rng_modules(src):
            return True
        digest = hashlib.sha256(src.encode('utf-8')).hexdigest()
        return bool(self._tracking_state.observed_rng_cells.get(digest))

    def _restore_position_rng_state(
        self, cell_code: str, notebook_cells: list[str], current_cell_idx: int | None,
    ) -> None:
        """Restore the RNG to the state it holds just before this cell (ADR-018).

        If the current cell draws, find the nearest UPSTREAM cell that touched the
        RNG (seed or draw) and whose post-state we recorded this session, and
        restore that state. A re-executed draw then continues from the correct
        stream position instead of the last-left live state (CAS-227). A no-op
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
            if seed_cells_not_yet_run(drawing, upstream, self._tracking_state.executed_cell_source_hashes):
                return
            # PRIMARY: the position this cell itself started from last time.
            # That is precisely what re-executing its draw needs, and it is exact
            # rather than inferred -- the upstream scan below reaches the same
            # value only indirectly, via the NEAREST predecessor's post-state
            # (equal by construction when nothing RNG-touching sits between).
            # Safe to prefer because the fingerprint expires it as soon as the
            # seed behind it changes, the same lineage check that invalidates any
            # other value.
            own = self._tracking_state.rng_pre_states.get(
                hashlib.sha256(cell_code.encode('utf-8')).hexdigest()
            )
            if own is not None:
                own_state, own_fingerprint = own
                if own_fingerprint == rng_lineage_fingerprint(
                    self.variable_lineage, drawing,
                ):
                    restore_rng_state(own_state)
                    return
                # Seed changed since it was recorded: the saved position belongs
                # to the old seed. Fall through rather than apply it.
            # FALLBACK: nearest upstream cell that touched RNG. Still needed when
            # this cell has no recorded start of its own (never run this session)
            # or its seed moved on, and it stays fresher than a stale own-position
            # when an upstream cell re-ran more recently than this one.
            post_states = self._tracking_state.rng_post_states
            for idx in range(end - 1, -1, -1):
                src = notebook_cells[idx]
                if not self._cell_touches_rng(src):
                    continue
                digest = hashlib.sha256(src.encode('utf-8')).hexdigest()
                state = post_states.get(digest)
                if state is not None:
                    restore_rng_state(state)
                return  # nearest predecessor only, whether or not it was recorded
        except (AttributeError, IndexError, TypeError):  # pragma: no cover - defensive
            return

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
                        # The processor reports statement failures via the
                        # 'error' field instead of raising - surface those
                        # through the same loud path below (CAS-87).
                        if result.get('error'):
                            raise UpstreamStateError(
                                self._format_upstream_failure(
                                    stmt_code, str(result['error'])
                                )
                            )
            except UpstreamStateError:
                raise
            except (RuntimeError, NameError, KeyError, TypeError, ValueError) as e:
                # Swallowing here served the downstream cell a STALE value
                # while the upstream producer was silently broken (CAS-87) -
                # the worst failure mode for a caching layer. Fail the user's
                # cell with the upstream failure instead, like a plain
                # top-to-bottom run would.
                logger.error("[ERROR] Failed to auto-execute statement: %s", e)
                raise UpstreamStateError(
                    self._format_upstream_failure(
                        stmt_code, f"{type(e).__name__}: {e}"
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

    @staticmethod
    def _format_upstream_failure(stmt_code: str, error_text: str) -> str:
        """One-line, embeddable message for an upstream statement failure.

        The executor re-raises this inside the user's cell via a generated
        ``raise ...('''<msg>''')`` statement, so the message must stay a
        single line and must not contain a triple quote.
        """
        stmt_short = stmt_code.split('\n')[0][:60]
        if len(stmt_code) > 60 or '\n' in stmt_code:
            stmt_short += '...'
        msg = (
            f"Upstream statement {stmt_short!r} failed during auto-"
            f"re-execution: {error_text}. Cash stopped instead of running "
            f"this cell against stale upstream state - fix the upstream "
            f"cell and re-run."
        )
        return msg.replace("'''", '"""').replace('\n', ' ')

    def _try_parse_control_structure(self, code: str) -> ast.AST | None:
        """Parse code and return the AST node if it's a single control structure."""
        tree = self.simulator.get_cached_ast(code)
        if tree and len(tree.body) == 1 and is_control_structure(tree.body[0]):
            return tree.body[0]
        return None

