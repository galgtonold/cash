"""For-loop per-iteration caching strategy.

**Boundary rule:** ``ForLoopHandler`` owns the per-iteration decomposition
of ``for`` loops — binding loop targets, building iteration contexts,
routing a loop to single-unit mode or a split, and dispatching each
body statement through the statement processor with an
``# __iteration_context__:`` cache-key discriminator.

It does NOT own:
- Shared lineage / mutation / badge helpers (in
  :mod:`control_structures.helpers`).
- The decision to run a loop as one unit (:mod:`.single_unit_policy`) or to
  learn it as a split (:mod:`.split_policy`); it only acts on them.
- Dispatch of nested ``if`` / ``try`` / single-unit fallback (delegated
  back through the orchestrator passed at construction time).

Tests can construct ``ForLoopHandler`` directly with mock dependencies
and exercise it without going through ``ControlStructureProcessor.process()``.
"""

from __future__ import annotations

import ast
import io
import logging
import time as _time
from typing import TYPE_CHECKING, Any

from cash.control_markers import iteration_digest, mark_iteration

from ...analysis.mutations import accumulator_loop_body_shape, cacheable_accumulator_loop
from ...lineage_tag import own_tag
from ...tracking.file_tracker import FileAccessTracker
from ...value_hash import compute_hash
from ..cache_status import CacheStatus
from ..loop_split import loop_source_hash, split_nodes
from . import helpers as _helpers
from . import single_unit_policy
from .common import (
    ControlStructureResult,
    bind_target_values,
    build_iteration_context,
    compute_context_hash,
    extract_target_names,
    is_control_structure,
)
from .split_policy import PROBE_ITERS as _SPLIT_PROBE_ITERS
from .split_policy import LoopSplitPolicy

if TYPE_CHECKING:
    from ..statement import ProcessResult, StatementProcessor

logger = logging.getLogger(__name__)


def _stamped(m: dict) -> list[dict]:
    """*m* and its intercepted sub-call events (``m['decorator_calls']``).

    The events need the same loop-nesting stamps as the metric, or the badge
    view-builder renders them as siblings of the loop instead of inside it.
    A malformed event (not a dict) is skipped: badge plumbing must never
    break statement execution.
    """
    return [m, *(e for e in m.get("decorator_calls") or () if isinstance(e, dict))]


def _prepend(target: dict, field: str, value: Any) -> None:
    chain = target.setdefault(field, [])
    if not chain or chain[0] != value:
        chain.insert(0, value)


def _stamp_loop_header(m: dict, loop_header: str) -> None:
    """Stamp *m* and its call events with an enclosing for-loop's header.

    ``loop_header`` is the innermost enclosing loop: the first handler in the
    recursion to see the metric wins. ``loop_header_chain`` is the whole
    enclosing chain, outermost-first: each handler prepends its own header,
    so the outermost one leaves the complete chain.
    """
    for target in _stamped(m):
        target.setdefault("loop_header", loop_header)
        _prepend(target, "loop_header_chain", loop_header)


def _stamp_body_index(m: dict, body_idx: int) -> None:
    """Stamp *m* and its call events with their statement's index in a loop body.

    ``body_index_chain`` is outermost-first like ``loop_header_chain``: a
    metric in for-b inside for-a gets ``[idx_of_for-b_in_for-a, idx_in_for-b]``,
    and the view-builder sorts each loop level by its own entry, so the body
    renders in source order even when nested controls split the metric
    stream. ``body_index`` on the metric is the innermost index.
    """
    for target in _stamped(m):
        _prepend(target, "body_index_chain", body_idx)
    m.setdefault("body_index", body_idx)


def _rebound_names(node: ast.For) -> set[str]:
    """The names a statement of *node*'s body binds."""
    return {
        n.id for stmt in node.body for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


class ForLoopHandler:
    """Per-iteration caching for ``for`` loops.

    Constructor deps are the same triple ``ControlStructureProcessor``
    itself carries plus a ``dispatcher`` reference used purely to recurse
    into nested control structures and to fall back to the single-unit
    execution path on the orchestrator.  In tests, pass a ``MagicMock``
    dispatcher — strategy-specific behaviour can be exercised without
    touching the orchestrator.
    """

    def __init__(self, shell, statement_processor: StatementProcessor, dispatcher):
        self.shell = shell
        self.statement_processor = statement_processor
        self.dispatcher = dispatcher
        self._split_policy = LoopSplitPolicy(statement_processor)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        node: ast.For,
        ttl: int | None,
        silent: bool,
        parent_context: dict[str, Any] | None,
        raw_cell: str | None = None,
        inherited_annotation=None,
        prev_node: ast.stmt | None = None,
    ):
        """
        Process a for loop with per-iteration caching.

        For each iteration:
        1. Bind the loop target variable(s) into user_ns.
        2. Build an iteration context (target values + iterable lineage).
        3. Process each body statement through the statement processor with
           an ``# __iteration_context__: <hash>`` comment prepended, and with
           its own ``@cash:`` annotation resolved from *raw_cell*.

        The statement processor's mutation detection (which runs before any
        cache lookup) will automatically set ``skip_cache=True`` for
        statements that mutate external variables, ensuring they always
        re-execute.

        **Fast-loop optimization**: When the estimated per-iteration caching
        overhead exceeds the likely computation cost (e.g., tight numeric
        loops with many iterations of cheap array operations), the loop is
        executed as a single cacheable unit instead of per-iteration.

        *prev_node* is the loop's immediately-preceding top-level sibling in
        the same cell, or ``None`` — passed through so the single-unit branch
        can compute ``force_outputs`` for a pure accumulator-loop shape
        (without it, that branch is refused outright by the
        in-place-mutation detector, since nothing suppresses that refusal for
        a matching ``out = []`` / ``out.append(f(e))`` loop). ``None`` by
        default so nested / direct callers with no notion of a preceding
        sibling are unaffected.
        """

        all_metrics: list[ProcessResult] = []
        rest_metrics: list[ProcessResult] = []
        target_names = extract_target_names(node.target)

        # A directive on the loop HEADER scopes to the loop, so it flows down
        # into every body statement. Resolved once here rather than per
        # iteration — it is a property of the source, not of the iteration.
        loop_annotation = _helpers.resolve_header_annotation(
            raw_cell,
            node,
            inherited_annotation,
        )

        logger.debug("[CONTROL] Processing FOR loop with targets: %s", target_names)

        try:
            if self._whole_before_evaluating(node):
                # The unit evaluates the header, once (`runs_whole_unevaluated`).
                logger.debug("[CONTROL] Fast-loop: executing as single unit, header unevaluated")
                return self.dispatcher.execute_as_single_unit(
                    node,
                    ttl,
                    silent,
                    raw_cell,
                    inherited_annotation,
                    force_outputs=self._single_unit_outputs(node, prev_node),
                )
            iter_code = ast.unparse(node.iter)
            iterable, header_files = self._evaluate_iterable(iter_code)

            # Pre-compute the original for-loop header line so each body
            # metric can carry it (used by the badge renderer to show the
            # source-faithful header `for cat in df['category'].unique():`
            # instead of synthesising one from observed iteration values).
            loop_header = f"for {ast.unparse(node.target)} in {iter_code}:"

            whole = self._run_whole(
                node, iterable, ttl, silent, parent_context, raw_cell, inherited_annotation, prev_node
            )
            if whole is not None:
                return whole

            total_iterations, cached_iterations = self._run_decomposed(
                node,
                iterable,
                header_files,
                target_names,
                ttl,
                silent,
                all_metrics,
                parent_context,
                raw_cell,
                loop_annotation,
                inherited_annotation,
                rest_metrics,
            )

            for m in all_metrics:
                if isinstance(m, dict):
                    _stamp_loop_header(m, loop_header)
            all_metrics.extend(rest_metrics)

            return ControlStructureResult(
                success=True,
                metrics=all_metrics,
                total_iterations=total_iterations,
                cached_iterations=cached_iterations,
                computed_iterations=total_iterations - cached_iterations,
            )

        except Exception as e:  # broad fallback wrapping arbitrary user for-loop body code
            # Handed back to the cell, which raises it; logged above debug it
            # would print the traceback a second time.
            logger.debug("[CONTROL] Error in for loop: %s", e, exc_info=True)
            all_metrics.extend(rest_metrics)  # a failure in the unit that ran the rest
            return ControlStructureResult(success=False, metrics=all_metrics, error=e)

    # ------------------------------------------------------------------
    # The steps of process()
    # ------------------------------------------------------------------

    def _whole_before_evaluating(self, node: ast.For) -> bool:
        """Whether the loop runs as one unit with its header left to the unit
        (`single_unit_policy.runs_whole_unevaluated`). A loop with a recorded
        split verdict is evaluated first: `_run_whole` splits it."""
        user_ns = getattr(self.shell, "user_ns", None) or {}
        if not single_unit_policy.runs_whole_unevaluated(node, user_ns):
            return False
        store = self._split_policy.store()
        if store is None:
            return True
        try:
            return store.get(loop_source_hash(node)) is None
        except (OSError, ValueError, RecursionError):
            logger.debug("[LOOP_SPLIT] verdict lookup failed", exc_info=True)
            return False

    def _evaluate_iterable(self, iter_code: str) -> tuple[Any, set[str]]:
        """Evaluate the loop's iterator, and the files evaluating it read.

        A loop is decomposed per-iteration and every body statement is
        tracked, but this expression is not a statement, so the files it
        opens must be recorded here. `for line in
        DATA.read_text().splitlines():` puts the only read of DATA in the
        header; untracked, the dict the loop fills would have no file
        dependency at all, and after the file changes a stale table would be
        served with a clean badge. The same rule as the body's reads
        (`inherit_body_file_deps`).

        `propagate_to_parent` is required, not tidiness. A manual
        `with FileAccessTracker(...)` is isolated by default, so a plain
        nested tracker would RECORD this read here and hide it from the
        statement-level tracker this loop runs inside. Propagating registers
        the read with both, which is what the decorator does for a cached
        call nested inside another.
        """
        with FileAccessTracker(self.shell.user_ns, propagate_to_parent=True) as iter_tracker:
            iterable = eval(iter_code, self.shell.user_ns, self.shell.user_ns)
        return iterable, set(iter_tracker.get_accessed_files())

    def _run_whole(
        self,
        node: ast.For,
        iterable: Any,
        ttl: int | None,
        silent: bool,
        parent_context: dict[str, Any] | None,
        raw_cell: str | None,
        inherited_annotation,
        prev_node: ast.stmt | None,
    ) -> ControlStructureResult | None:
        """Run the loop as a recorded split or as one unit, when it is to be
        run that way; None when it is to be decomposed per iteration."""
        # A loop with a recorded verdict is split HERE too, not only in the
        # simulator. The two cover different entry points and must agree,
        # which they do by reading the same persisted ``k``:
        #
        # * the simulator's split is what the re-execution planner
        #   dispatches when an upstream edit re-plans this cell;
        # * this branch is what splits a DIRECT re-run of the cell, which
        #   never goes through the planner at all.
        #
        # Splitting in only one of them is wrong either way: runtime-only
        # leaves the planner re-running the whole loop against entries
        # written for halves (a stale value), simulator-only leaves a plain
        # re-run paying full decomposition.
        user_ns = getattr(self.shell, "user_ns", None) or {}
        verdict_k = self._split_policy.recorded_k(node, iterable, user_ns)
        if verdict_k is not None:
            return self._run_split(node, verdict_k, ttl, silent, parent_context, raw_cell, inherited_annotation)

        # Fast-loop heuristic: if per-iteration decomposition would be too
        # expensive relative to the computation, execute as a single unit.
        #
        # The single-unit path re-executes the loop FROM SOURCE, which
        # evaluates ``node.iter`` a SECOND time (it was already evaluated by
        # the caller). That double-eval is harmless for a re-iterable
        # container produced by a side-effect-free header, but for a one-shot
        # consumable (a stored generator, ``iter(...)``, ``map``/``zip``, an
        # open file, or a side-effecting call like ``drain()``) the second
        # evaluation drains an already-exhausted source, corrupting the
        # first-run result. Only take the fast path when re-evaluating the
        # header is provably safe; otherwise the loop is decomposed, which
        # consumes the single, already evaluated ``iterable``.
        #
        # The header's safety is asked first: it is the cheaper question, and
        # sizing a file loop reads the start of the file.
        #
        # A header that only names an iterator (`for x in it:`) is evaluated
        # again too, but that is a lookup: the unit gets the same iterator,
        # not yet drawn from (`header_names_the_iterator`).
        names_iterator = single_unit_policy.header_names_the_iterator(node.iter, iterable, user_ns)
        if not (
            (names_iterator or single_unit_policy.header_safe_to_reevaluate(node.iter, iterable, user_ns))
            and single_unit_policy.should_run_as_single_unit(node, iterable, user_ns)
        ):
            return None
        logger.debug("[CONTROL] Fast-loop: executing as single unit (overhead > benefit)")
        if names_iterator:
            # The user's own iterator, handle or bar: the unit draws from it.
            return self.dispatcher.execute_as_single_unit(
                node,
                ttl,
                silent,
                raw_cell,
                inherited_annotation,
                force_outputs=self._single_unit_outputs(node, prev_node),
            )
        # `for line in open(path)` was opened here and will be opened again by
        # the unit: this handle is never read, so it is closed rather than
        # left for the collector.
        if isinstance(iterable, io.IOBase):
            iterable.close()
        elif type(iterable).__module__.partition(".")[0] == "tqdm":
            iterable.leave = False  # a bar drawn here is never advanced: it is cleared, not left at 0%
            iterable.close()
        # Single-unit mode makes the loop ONE cache entry, so the unit
        # annotation (whole range) is the right scope — a body directive has
        # no finer entry to attach to here.
        return self.dispatcher.execute_as_single_unit(
            node,
            ttl,
            silent,
            raw_cell,
            inherited_annotation,
            force_outputs=self._single_unit_outputs(node, prev_node),
        )

    @staticmethod
    def _single_unit_outputs(node: ast.For, prev_node: ast.stmt | None) -> set[str] | None:
        """The outputs a single-unit run of a pure accumulator loop must keep.

        A pure accumulator loop's body (``out.append(f(e))``) reads as an
        in-place mutation to the per-statement analyzer, which would
        otherwise refuse to cache the whole unit outright -- and a large,
        cheap accumulator loop would get no caching from either mechanism
        (decomposition never runs here). These outputs, from the narrow
        shape detector, suppress exactly that refusal (and capture the
        leaked loop variable) so the single-unit path is actually cacheable.
        ``None`` for every other loop, which keeps its behaviour unchanged.
        """
        acc_loop = cacheable_accumulator_loop(node, prev_node)
        if acc_loop is None:
            return None
        acc, loop_vars, _iter_node, _expr_call = acc_loop
        force_outputs = {acc, *loop_vars}
        logger.debug("[CONTROL] Single-unit accumulator loop -> force_outputs=%s", force_outputs)
        return force_outputs

    def _run_decomposed(
        self,
        node: ast.For,
        iterable: Any,
        header_files: set[str],
        target_names: list,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        parent_context: dict[str, Any] | None,
        raw_cell: str | None,
        loop_annotation,
        inherited_annotation=None,
        rest_metrics: list | None = None,
    ) -> tuple[int, int]:
        """Run the loop iteration by iteration, each body statement its own
        cache entry; returns ``(iterations, fully cached iterations)``.

        Also measures the first iterations for a later split verdict, and
        gives the variables the loop changed the files it read.

        A loop over an iterator of unknown length that its header only names
        (``for x in it:``) runs its first passes one by one; once it has run
        as many as :func:`single_unit_policy.iterations_for_one_unit` asks,
        the rest runs as one unit from source, which draws the rest from the
        same iterator (its metrics go to *rest_metrics*). Before, every pass
        of `for (name, act) in parsed:` over 87,464 items went through the
        per-statement machinery: 21.9 s against 0.19 s plain.
        """
        iterable_lineage = _helpers.get_iterable_lineage(self.shell, self.statement_processor, node.iter)
        logger.debug("[CONTROL] Iterable lineage: %s...", iterable_lineage[:20] if iterable_lineage else "None")

        # Measure the first few iterations so this loop can be judged for
        # splitting on a LATER run. Nothing is split here.
        user_ns = getattr(self.shell, "user_ns", None) or {}
        probe_n = self._split_policy.eligible(node, iterable, user_ns)
        probe_elapsed = 0.0

        file_deps = self.statement_processor.tracking_state.executed_file_deps
        body_names = _rebound_names(node) if file_deps is not None else set()
        # Seeded with the header's reads: `inherit_body_file_deps` gives every
        # variable the loop mutated the files the loop read, and the iterable
        # is as much a read as the body is.
        body_files: set[str] = set(header_files)

        rest_after = None
        if (
            single_unit_policy.header_names_the_iterator(node.iter, iterable, user_ns)
            and single_unit_policy.estimated_iterations(node.iter, iterable, user_ns) is None
        ):
            rest_after = single_unit_policy.iterations_for_one_unit(node)
        run_rest = False

        total_iterations = 0
        cached_iterations = 0
        loop_pass = _helpers.LoopPass()
        passes = getattr(self.dispatcher, "loop_passes", None)
        if isinstance(passes, list):
            passes.append(loop_pass)
        try:
            for idx, iteration_value in enumerate(iterable):
                total_iterations += 1
                iter_started = _time.perf_counter()
                if self._process_one_iteration(
                    node,
                    iteration_value,
                    iterable_lineage,
                    target_names,
                    ttl,
                    silent,
                    all_metrics,
                    parent_context,
                    raw_cell,
                    loop_annotation,
                ):
                    cached_iterations += 1
                if probe_n is not None and idx < _SPLIT_PROBE_ITERS:
                    probe_elapsed += _time.perf_counter() - iter_started
                # A name the body rebinds (`d = pd.read_csv(f)`) holds only its
                # latest iteration's file, so the accumulator's inheritance is
                # gathered as the loop goes. One changed in place
                # (`parts.append(d)`) keeps every file and is read once at the
                # end: read per iteration, its growing set would make the
                # gathering quadratic.
                for name in body_names:
                    body_files.update(file_deps.get(name, ()))
                if rest_after is not None and total_iterations >= rest_after:
                    # Still the very iterator this loop draws from: the unit
                    # evaluates the header again.
                    run_rest = user_ns.get(node.iter.id) is iterable
                    if run_rest:
                        break
        finally:
            if isinstance(passes, list) and passes and passes[-1] is loop_pass:
                passes.pop()

        if run_rest:
            logger.debug("[CONTROL] %d passes over an iterator: the rest runs as one unit", total_iterations)
            rest = self.dispatcher.execute_as_single_unit(
                node, ttl, silent, raw_cell, inherited_annotation, update_lineage=False
            )
            if rest_metrics is not None:
                rest_metrics.extend(rest.metrics)
            if not rest.success:
                raise rest.error or RuntimeError("Error in the rest of the loop")
            total_iterations += 1

        if probe_n is not None:
            self._split_policy.record_verdict(node, probe_elapsed, probe_n)

        # After all iterations, update lineage for mutated variables
        _helpers.update_lineage_after_execution(
            self.shell,
            self.statement_processor,
            node,
            ast.unparse(node),
            body_files=body_files,
            unchanged=loop_pass.unchanged(self.shell, self.statement_processor, node),
        )
        return total_iterations, cached_iterations

    # ------------------------------------------------------------------
    # Per-iteration processing
    # ------------------------------------------------------------------

    def _process_one_iteration(
        self,
        node: ast.For,
        iteration_value: Any,
        iterable_lineage: str | None,
        target_names: list,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        parent_context: dict | None,
        raw_cell: str | None = None,
        loop_annotation=None,
    ) -> bool:
        """Process a single loop iteration; return True if fully cached."""
        loop_var_digests = self._bind_iteration(node, iteration_value)
        # `loop_var_digests` is fully populated over these same bindings.
        # Handing it over stops `build_iteration_context` recomputing an
        # identical `compute_hash` on an identical object -- a full duplicate
        # of the most expensive thing an iteration does when the loop target
        # is large.
        iteration_context = build_iteration_context(target_names, self.shell.user_ns, parent_context, loop_var_digests)
        if iterable_lineage:
            iteration_context["__iterable_lineage__"] = iterable_lineage

        context_hash = compute_context_hash(iteration_context)
        loop_vars = {k: v for k, v in iteration_context.items() if not k.startswith("__")}
        # Pushed once for the WHOLE iteration's body, not per statement: an
        # intercepted sub-call needs the CURRENT iteration's loop-var values
        # as a key discriminator wherever it sits, including inside a nested
        # `if`/`try` or `for`. `loop_vars_scope`'s `finally` pops even if a
        # body statement raises.
        with self.statement_processor.loop_vars_scope(loop_vars, loop_var_digests):
            return self._run_body(
                node, iteration_context, context_hash, loop_vars, ttl, silent, all_metrics, raw_cell, loop_annotation
            )

    def _bind_iteration(self, node: ast.For, iteration_value: Any) -> dict[str, str]:
        """Bind *iteration_value* to the loop's targets and record each
        target's lineage; returns each target's content digest.

        The digests are for ``loop_vars_scope`` (not written into
        ``variable_lineage``), so a call's key build can look them up without
        the staleness that dict carries: ``variable_lineage`` is a FLAT,
        never-popped dict, and a nested loop reusing this iteration's target
        name would overwrite the entry for the rest of this iteration, with
        nothing to restore it once the inner loop ends. ``loop_vars_scope``'s
        stack is popped when this iteration's body finishes, so
        ``call_key._loop_var_digest`` reads the digest from there. The
        ``variable_lineage`` write serves bare-Name argument resolution in
        the base cache key (``compute_cache_key``'s lineage ladder).
        """
        bindings = bind_target_values(node.target, iteration_value, self.shell.user_ns)
        loop_var_digests: dict[str, str] = {}
        for name, val in bindings.items():
            try:
                # The loop variable's hash IS the per-iteration cache-key
                # discriminator: its whole content.
                full = compute_hash(val)
                # `variable_lineage[name]` and `loop_var_digests[name]` want
                # different things. `variable_lineage` wants PROVENANCE, and
                # `val`'s own `_cash_lineage_hash` is the cheap right answer.
                # `loop_var_digests` wants CONTENT, the call key's
                # discriminator: a tag names where a value came from, not
                # what it holds, and if two frames carried one tag,
                # `for df in [df_a, df_b]:` would collapse iteration 2 onto
                # iteration 1's cached value. Only `compute_hash` of
                # `val` answers it soundly -- one full hash per iteration.
                tag = own_tag(val)
                h = tag if tag is not None else full
                self.statement_processor.tracking_state.lineage.record(name, h)
                loop_var_digests[name] = full
            except (TypeError, ValueError, AttributeError) as exc:
                logger.debug("[CONTROL] Failed to hash loop variable %s: %s", name, exc)
        return loop_var_digests

    def _run_body(
        self,
        node: ast.For,
        iteration_context: dict[str, Any],
        context_hash: str,
        loop_vars: dict[str, Any],
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        raw_cell: str | None,
        loop_annotation,
    ) -> bool:
        """Run one iteration's body statements; True if none of them computed."""
        iteration_cached = True
        for body_idx, body_node in enumerate(node.body):
            before_count = len(all_metrics)
            if is_control_structure(body_node):
                was_computed = self._execute_loop_body_nested_control(
                    body_node,
                    ttl,
                    silent,
                    iteration_context,
                    context_hash,
                    loop_vars,
                    all_metrics,
                    raw_cell,
                    loop_annotation,
                )
            else:
                was_computed = self._execute_loop_body_statement(
                    body_node,
                    context_hash,
                    loop_vars,
                    ttl,
                    silent,
                    all_metrics,
                    raw_cell,
                    loop_annotation,
                )
            for m in all_metrics[before_count:]:
                if isinstance(m, dict):
                    _stamp_body_index(m, body_idx)
            if was_computed:
                iteration_cached = False
        return iteration_cached

    def _execute_loop_body_nested_control(
        self,
        body_node: ast.AST,
        ttl: int | None,
        silent: bool,
        iteration_context: dict[str, Any],
        context_hash: str,
        loop_vars: dict[str, Any],
        all_metrics: list,
        raw_cell: str | None = None,
        loop_annotation=None,
    ) -> bool:
        """Run a control structure nested in the loop body; True if any of it computed.

        Its metrics get this iteration's marker and loop variables so the
        badge keeps them inside the loop, and the loop's annotation flows
        down into it.
        """

        def tag(m: dict[str, Any]) -> None:
            code = m.get("code", "")
            if iteration_digest(code) is None:
                m["code"] = mark_iteration(code, context_hash)
            if loop_vars and "loop_vars" not in m:
                m["loop_vars"] = loop_vars

        result = _helpers.run_nested_structure(
            self.dispatcher,
            body_node,
            ttl,
            silent,
            iteration_context,
            raw_cell,
            loop_annotation,
            all_metrics,
            tag,
        )
        return result.computed_iterations > 0

    def _execute_loop_body_statement(
        self,
        body_node: ast.AST,
        context_hash: str,
        loop_vars: dict[str, Any],
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        raw_cell: str | None = None,
        loop_annotation=None,
    ) -> bool:
        """Run one plain statement of the loop body; True if it computed.

        *context_hash*, the digest of the iteration context (loop variable
        values and the iterable's lineage), goes into the cache key as a
        marker, so each iteration is its own entry.
        """
        metrics = _helpers.run_marked_statement(
            self.statement_processor,
            body_node,
            lambda code: mark_iteration(code, context_hash),
            ttl,
            silent,
            raw_cell,
            loop_annotation,
            all_metrics,
            {"loop_vars": loop_vars} if loop_vars else {},
        )
        return metrics.get("status") == CacheStatus.COMPUTED

    # ------------------------------------------------------------------
    # Split execution
    # ------------------------------------------------------------------

    def _run_split(self, node, k, ttl, silent, parent_context, raw_cell, inherited_annotation):
        """Execute a split loop as head + tail.

        Halves come from ``loop_split.split_nodes`` -- the same derivation the
        simulator uses -- so both sides run the same two statements. The head
        recurses through :meth:`process` (and is itself unsplittable, being a
        half); the tail takes the ordinary single-unit path. A head that
        fails ends the loop there, with its error, as the unsplit loop would.
        """
        # ``recorded_k`` returned a k, so the loop has no ``else`` to refuse.
        head, tail = split_nodes(node, k)

        # A tail's accumulator is populated by its own head, so the shape is
        # read from the BODY rather than requiring a fresh preceding seed.
        force_outputs = None
        shape = accumulator_loop_body_shape(node)
        if shape is not None:
            acc, loop_vars = shape
            force_outputs = {acc, *loop_vars}
        logger.debug("[LOOP_SPLIT] executing split at k=%d (force_outputs=%s)", k, force_outputs)

        head_res = self.process(head, ttl, silent, parent_context, raw_cell, inherited_annotation)
        if not head_res.success:
            return head_res
        tail_res = self.dispatcher.execute_as_single_unit(
            tail,
            ttl,
            silent,
            raw_cell,
            inherited_annotation,
            force_outputs=force_outputs,
        )
        return ControlStructureResult(
            success=tail_res.success,
            metrics=list(head_res.metrics) + list(tail_res.metrics),
            error=tail_res.error,
            total_iterations=head_res.total_iterations + tail_res.total_iterations,
            cached_iterations=head_res.cached_iterations + tail_res.cached_iterations,
            computed_iterations=head_res.computed_iterations + tail_res.computed_iterations,
        )
