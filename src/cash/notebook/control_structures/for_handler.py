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
import contextlib
import logging
import time as _time
from typing import TYPE_CHECKING, Any

from cash.control_markers import iteration_digest, mark_iteration

from ...analysis.cacheability import accumulator_loop_body_shape, cacheable_accumulator_loop
from ...lineage_tag import own_tag
from ...object_hashing import compute_hash_full
from ...tracking.file_tracker import FileAccessTracker
from ..cache_status import CacheStatus
from ..loop_split import split_nodes
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
    from ..statement import ProcessResult

logger = logging.getLogger(__name__)


def _stamp_call_events_loop_header(m: dict, loop_header: str) -> None:
    """Propagate *m*'s ``loop_header``/``loop_header_chain`` onto its call events.

    ``m['decorator_calls']`` may hold intercepted (on by default, CAS-243)
    sub-call events. Those need the identical loop-nesting stamp
    the enclosing metric just got, or the badge view-builder has nothing to
    key nesting on and renders them as siblings of the loop instead of
    inside it. Mirrors the caller's own first-writer-wins / prepend rules
    exactly so the two can never disagree.

    Defensive by construction: a malformed event (not a dict) is skipped
    rather than raising -- this must never break statement execution.
    """
    for event in m.get("decorator_calls") or ():
        if not isinstance(event, dict):
            continue
        if "loop_header" not in event:
            event["loop_header"] = loop_header
        echain = event.setdefault("loop_header_chain", [])
        if not echain or echain[0] != loop_header:
            echain.insert(0, loop_header)


def _stamp_call_events_body_index(m: dict, body_idx: int) -> None:
    """Propagate *m*'s ``body_index_chain`` onto its call events. See above."""
    for event in m.get("decorator_calls") or ():
        if not isinstance(event, dict):
            continue
        echain = event.setdefault("body_index_chain", [])
        if not echain or echain[0] != body_idx:
            echain.insert(0, body_idx)


class ForLoopHandler:
    """Per-iteration caching for ``for`` loops.

    Constructor deps are the same triple ``ControlStructureProcessor``
    itself carries plus a ``dispatcher`` reference used purely to recurse
    into nested control structures and to fall back to the single-unit
    execution path on the orchestrator.  In tests, pass a ``MagicMock``
    dispatcher — strategy-specific behaviour can be exercised without
    touching the orchestrator.
    """

    def __init__(self, shell, statement_processor, debug: bool, dispatcher):
        self.shell = shell
        self.statement_processor = statement_processor
        self.debug = debug
        self.dispatcher = dispatcher
        self._split_policy = LoopSplitPolicy(statement_processor, debug)

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
        can compute ``force_outputs`` for a pure accumulator-loop shape (CAS-259
        follow-up: without it, that branch is refused outright by the
        in-place-mutation detector, since nothing suppresses that refusal for
        a matching ``out = []`` / ``out.append(f(e))`` loop). ``None`` by
        default so nested / direct callers with no notion of a preceding
        sibling are unaffected.
        """

        all_metrics: list[ProcessResult] = []
        total_iterations = 0
        cached_iterations = 0
        computed_iterations = 0

        target_names = extract_target_names(node.target)

        # A directive on the loop HEADER scopes to the loop, so it flows down
        # into every body statement. Resolved once here rather than per
        # iteration — it is a property of the source, not of the iteration.
        loop_annotation = _helpers.resolve_header_annotation(
            raw_cell,
            node,
            inherited_annotation,
        )

        if self.debug:
            logger.debug("[CONTROL] Processing FOR loop with targets: %s", target_names)

        try:
            # Evaluate the iterator, watching what it READS.
            #
            # A loop is decomposed per-iteration and every body statement is
            # tracked, but this expression is not a statement -- so before
            # round 27 the files it opened were recorded against nothing.
            # `for line in DATA.read_text().splitlines():` put the only read
            # of DATA here, the body never touched the file, and the dict the
            # loop filled came out with no file dependency at all:
            # `%cash_provenance ALIAS` reported `Code: ALIAS = {}`. Change the
            # file, run a cell below, and the stale table was served with a
            # clean badge (r27s4: nine wrong exports, no `Upstream:` block).
            #
            # The same rule as the body's reads, which `inherit_body_file_deps`
            # has applied since round 23 -- the header was simply never part
            # of it.
            #
            # `propagate_to_parent` is required, not tidiness. A manual
            # `with FileAccessTracker(...)` is isolated by default, so a
            # plain nested tracker would RECORD this read here and hide it
            # from the statement-level tracker this loop runs inside --
            # moving the bug rather than fixing it. Propagating registers
            # the read with both, which is what the decorator does for a
            # cached call nested inside another.
            iter_code = ast.unparse(node.iter)
            with FileAccessTracker(self.shell.user_ns, propagate_to_parent=True) as _iter_tracker:
                iterable = eval(iter_code, self.shell.user_ns, self.shell.user_ns)
            _header_files = set(_iter_tracker.get_accessed_files())

            # Pre-compute the original for-loop header line so each body
            # metric can carry it (used by the badge renderer to show the
            # source-faithful header `for cat in df['category'].unique():`
            # instead of synthesising one from observed iteration values).
            target_code = ast.unparse(node.target)
            loop_header = f"for {target_code} in {iter_code}:"

            # A loop with a recorded verdict is split HERE too, not only in
            # the simulator. The two cover different entry points and must
            # agree, which they do by reading the same persisted ``k``:
            #
            # * the simulator's split is what the re-execution planner
            #   dispatches when an upstream edit re-plans this cell;
            # * this branch is what splits a DIRECT re-run of the cell, which
            #   never goes through the planner at all.
            #
            # Splitting in only one of them is the bug that reverted three
            # earlier attempts -- runtime-only left the planner re-running the
            # whole loop against entries written for halves (stale value),
            # simulator-only left a plain re-run paying full decomposition.
            user_ns = getattr(self.shell, "user_ns", None) or {}
            _verdict_k = self._split_policy.recorded_k(node, iterable, user_ns)
            if _verdict_k is not None:
                return self._run_split(node, _verdict_k, ttl, silent, parent_context, raw_cell, inherited_annotation)

            # Fast-loop heuristic: if per-iteration decomposition would be too
            # expensive relative to the computation, execute as a single unit.
            #
            # The single-unit path re-executes the loop FROM SOURCE, which
            # evaluates ``node.iter`` a SECOND time (it was already evaluated
            # above).  That double-eval is harmless for a re-iterable container
            # produced by a side-effect-free header, but for a one-shot
            # consumable (a stored generator, ``iter(...)``, ``map``/``zip``, an
            # open file, or a side-effecting call like ``drain()``) the second
            # evaluation drains an already-exhausted source, corrupting the
            # first-run result.  Only take the fast path when re-evaluating the
            # header is provably safe; otherwise fall through to the
            # per-iteration path below, which consumes the single, already
            # evaluated ``iterable``.
            if single_unit_policy.should_run_as_single_unit(
                node, iterable, user_ns, debug=self.debug
            ) and single_unit_policy.header_safe_to_reevaluate(node.iter, iterable, user_ns):
                if self.debug:
                    logger.debug("[CONTROL] Fast-loop: executing as single unit (overhead > benefit)")
                # Single-unit mode makes the loop ONE cache entry, so the unit
                # annotation (whole range) is the right scope — a body directive
                # has no finer entry to attach to here.
                #
                # A pure accumulator loop's body (``out.append(f(e))``) reads as
                # an in-place mutation to the per-statement analyzer, which
                # would otherwise refuse to cache this whole unit outright --
                # CAS-259 shipped without this and every large/cheap
                # accumulator loop got ZERO caching from either mechanism
                # (decomposition never runs here; the single unit was refused).
                # force_outputs, computed from the narrow shape detector,
                # suppresses exactly that refusal reason (and captures the
                # leaked loop variable) so the chosen single-unit path is
                # actually cacheable. ``None`` for every other single-unit
                # loop, which keeps their behaviour unchanged.
                force_outputs = None
                acc_loop = cacheable_accumulator_loop(node, prev_node)
                if acc_loop is not None:
                    acc, loop_vars, _iter_node, _expr_call = acc_loop
                    force_outputs = {acc, *loop_vars}
                    if self.debug:
                        logger.debug(
                            "[CONTROL] Single-unit accumulator loop -> force_outputs=%s",
                            force_outputs,
                        )
                return self.dispatcher.execute_as_single_unit(
                    node,
                    ttl,
                    silent,
                    raw_cell,
                    inherited_annotation,
                    force_outputs=force_outputs,
                )

            # all iterations.
            iterable_lineage = _helpers.get_iterable_lineage(self.shell, self.statement_processor, node.iter)
            if self.debug:
                logger.debug("[CONTROL] Iterable lineage: %s...", iterable_lineage[:20] if iterable_lineage else "None")

            # Measure the first few iterations so this loop can be judged for
            # splitting on a LATER run. Nothing is split here.
            _probe_n = self._split_policy.eligible(node, iterable, user_ns)
            _probe_elapsed = 0.0

            # The files each iteration's body read. A name the body rebinds
            # (`d = pd.read_csv(f)`) holds only its latest iteration's file, so
            # the accumulator's inheritance is gathered as the loop goes. Only
            # rebound names: one changed in place (`parts.append(d)`) keeps
            # every file and is read once at the end -- read per iteration, its
            # growing set made the gathering quadratic again.
            _file_deps = getattr(self.statement_processor, "executed_file_deps", None)
            _body_names = (
                {
                    n.id
                    for stmt in node.body
                    for n in ast.walk(stmt)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
                if _file_deps is not None
                else set()
            )
            # Seeded with the header's reads: `inherit_body_file_deps` gives
            # every variable the loop mutated the files the loop read, and the
            # iterable is as much a read as the body is.
            _body_files: set[str] = set(_header_files)

            for _idx, iteration_value in enumerate(iterable):
                total_iterations += 1
                _iter_started = _time.perf_counter()
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
                else:
                    computed_iterations += 1
                if _probe_n is not None and _idx < _SPLIT_PROBE_ITERS:
                    _probe_elapsed += _time.perf_counter() - _iter_started
                for _name in _body_names:
                    _body_files.update(_file_deps.get(_name, ()))

            if _probe_n is not None:
                self._split_policy.record_verdict(node, _probe_elapsed, _probe_n)

            # After all iterations, update lineage for mutated variables
            _helpers.update_lineage_after_execution(
                self.shell,
                self.statement_processor,
                node,
                ast.unparse(node),
                debug=self.debug,
                body_files=_body_files,
            )

            # Stamp every body metric with this for-loop's source header.
            # ``loop_header`` itself = innermost enclosing for-loop (the
            # first for-handler in the recursion to see this metric wins).
            # ``loop_header_chain`` = full enclosing chain, outermost-first:
            # we PREPEND this loop's header on each recursion frame so the
            # outermost call ends up with the complete chain. Used by the
            # view-builder to nest for-loop groups instead of rendering
            # them as siblings.
            for m in all_metrics:
                if not isinstance(m, dict):
                    continue
                if "loop_header" not in m:
                    m["loop_header"] = loop_header
                chain = m.setdefault("loop_header_chain", [])
                if not chain or chain[0] != loop_header:
                    chain.insert(0, loop_header)
                # A statement's intercepted (on by default) sub-call
                # events need this SAME stamp, or the view-builder has no way
                # to tell they belong inside this loop and renders them as
                # siblings instead (CAS-243 task 9). ``event`` is a dict
                # inside ``m['decorator_calls']`` -- stamped in lockstep with
                # ``m`` itself, same first-writer-wins / prepend rules, so
                # nesting can never disagree between the two.
                _stamp_call_events_loop_header(m, loop_header)

            return ControlStructureResult(
                success=True,
                metrics=all_metrics,
                total_iterations=total_iterations,
                cached_iterations=cached_iterations,
                computed_iterations=computed_iterations,
            )

        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user for-loop body code
            # Handed back to the cell, which raises it: logged at ERROR it printed
            # the traceback a second time, through cash (round 25, r25s2/r25s3).
            logger.debug("[CONTROL] Error in for loop: %s", e, exc_info=True)
            return ControlStructureResult(success=False, metrics=all_metrics, error=e)

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

        bindings = bind_target_values(node.target, iteration_value, self.shell.user_ns)
        # `loop_var_digests` collects the SAME hash computed just below for
        # `variable_lineage`, THIS iteration's binding only -- pushed through
        # `loop_vars_scope` (not written into `variable_lineage`) so a call's
        # key build can look it up without the staleness that dict carries.
        # `variable_lineage` is a FLAT, never-popped dict: a nested loop
        # reusing this iteration's target name would overwrite the entry
        # for the rest of this iteration, with nothing to restore it once
        # the inner loop ends. `loop_vars_scope`'s stack has real scope
        # discipline (popped when this iteration's body finishes), so
        # `call_unit._loop_var_digest` sources the digest from there instead
        # -- see that function's docstring for the live repro that found
        # this. The `variable_lineage` write below is UNCHANGED and still
        # needed for its own, separate purpose (bare-Name argument
        # resolution in the base cache key, `compute_cache_key`'s lineage
        # ladder) -- this task does not touch that.
        loop_var_digests: dict[str, str] = {}
        for name, val in bindings.items():
            try:
                # The loop variable's hash IS the per-iteration cache-key
                # discriminator: a sampled hash keyed two iterations over
                # arrays that agreed in the sample onto ONE entry - wrong
                # result on the first run. Hash full content here.
                full = compute_hash_full(val)
                # `variable_lineage[name]` and `loop_var_digests[name]`
                # WANT DIFFERENT THINGS and must not be conflated -- a lesson
                # learned the hard way (CAS-243 review, round 5): `val`'s own
                # `_cash_lineage_hash`, when present, may itself have been
                # derived from a SAMPLED hash -- `update_mutated_variable_lineages`
                # (control_structures/helpers.py) computes a mutated
                # accumulator's new lineage from `statement_processor.compute_hash(val)`
                # (the sampling hash) and `LineageStore.record` stamps that
                # onto `val._cash_lineage_hash`. Two content-different
                # DataFrames that agree on the sampled portion (a DataFrame's
                # `compute_hash` samples shape + dtypes + `head(5)` --
                # `object_hashing.py`'s `_hash_dataframe_or_series` -- so two that
                # differ only past row 5 hash equal there) then carry the
                # SAME `_cash_lineage_hash` -- so if this loop
                # variable is later bound to each of them in turn (`for df in
                # [df_a, df_b]:`), preferring that attribute for the CALL KEY
                # would collapse iteration 2 onto iteration 1's cached value.
                # Reproduced live in a real kernel: `SL2 [1, 1]` instead of
                # the oracle's `[1, 2]`. This is round 1's lesson one layer
                # further out -- a sampled hash is never sound as a key
                # discriminator, even smuggled in through an attribute rather
                # than passed directly.
                #
                # `variable_lineage` wants PROVENANCE (did this binding come
                # from the same upstream computation as before?), where
                # `_cash_lineage_hash` is the right, deliberately-cheaper
                # answer and has been for as long as this line has existed.
                # `loop_var_digests` wants CONTENT (are two iterations'
                # bindings the same VALUE?), which only `compute_hash_full`
                # can answer soundly -- so it is computed directly from `val`,
                # never through the attribute. Cost is unchanged: still one
                # full hash per iteration (not per call, and not per call
                # multiplied by however many cached calls read this loop
                # var), just no longer skippable via the attribute shortcut
                # for THIS consumer specifically.
                tag = own_tag(val)
                h = tag if tag is not None else full
                self.statement_processor.variable_lineage[name] = h
                loop_var_digests[name] = full
            except (TypeError, ValueError, AttributeError) as exc:
                if self.debug:
                    logger.warning("[CONTROL] Failed to hash loop variable %s: %s", name, exc)

        # `loop_var_digests` is fully populated by the loop above, over these
        # same bindings. Handing it over stops `build_iteration_context`
        # recomputing an identical `compute_hash_full` on an identical object --
        # a full duplicate of the most expensive thing an iteration does when
        # the loop target is large.
        iteration_context = build_iteration_context(target_names, self.shell.user_ns, parent_context, loop_var_digests)
        if iterable_lineage:
            iteration_context["__iterable_lineage__"] = iterable_lineage

        context_hash = compute_context_hash(iteration_context)
        loop_vars = {k: v for k, v in iteration_context.items() if not k.startswith("__")}
        iteration_cached = True
        # Track the AST body index of each emitted metric so the view-
        # builder can render the for-loop's body in source order even
        # when nested controls split the metric stream (some iterations
        # produce control_context'd metrics, others don't; without the
        # index the renderer would group all "before"/"after" stmts then
        # show the control after them, instead of the source order
        # before / if / after).
        #
        # ``body_index_chain`` is recorded outermost-first (analogous to
        # ``loop_header_chain``): a metric in for-b inside for-a gets
        # chain ``[idx_of_for-b_in_for-a, idx_in_for-b]``. The view-builder
        # uses chain[depth] when sorting items inside a specific for-loop
        # level. ``body_index`` itself is the *innermost* index (the
        # tail of the chain).
        # Pushed once for the WHOLE iteration's body, not per statement: an
        # intercepted (on by default) sub-call needs the CURRENT
        # iteration's loop-var values as a key discriminator wherever it sits
        # -- a plain body statement, or nested inside an `if`/`try` reached via
        # `_execute_loop_body_nested_control` below -- so one push covering the
        # whole body loop reaches every statement this iteration executes,
        # including a nested `for`'s own (further-nested) push on top of it.
        # `loop_vars_scope`'s `finally` pops even if a body statement raises.
        #
        # `getattr(..., None)` guarded rather than called directly: this is
        # the file where a caching optimisation, if it broke, would break the
        # USER'S LOOP rather than merely its caching -- a `statement_processor`
        # without this method would otherwise raise `AttributeError` out of
        # `process()`, and `cell_executor.py` re-raises that as the user's own
        # error, so their loop would not run at all. Unreachable today (one
        # construction site, `magics.py`), but requirement 5 (never let a
        # caching optimisation be why user code fails) should hold at both
        # ends of this wire, not just inside `CallUnit`.
        loop_vars_scope = getattr(self.statement_processor, "loop_vars_scope", None)
        scope = (
            loop_vars_scope(loop_vars, loop_var_digests) if loop_vars_scope is not None else contextlib.nullcontext()
        )
        with scope:
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
                        iteration_context,
                        ttl,
                        silent,
                        all_metrics,
                        raw_cell,
                        loop_annotation,
                    )
                for m in all_metrics[before_count:]:
                    if not isinstance(m, dict):
                        continue
                    chain = m.setdefault("body_index_chain", [])
                    # Prepend this loop's body_idx (outermost wins by being
                    # at index 0). Innermost handler runs first and ends up
                    # at the chain tail; outer handlers prepend their idx.
                    if not chain or chain[0] != body_idx:
                        chain.insert(0, body_idx)
                    if "body_index" not in m:
                        m["body_index"] = body_idx
                    # Same stamp, same reason, onto this statement's intercepted
                    # sub-call events (CAS-243 task 9) -- see the loop_header
                    # stamp above for why.
                    _stamp_call_events_body_index(m, body_idx)
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
        iteration_context: dict[str, Any],
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        raw_cell: str | None = None,
        loop_annotation=None,
    ) -> bool:
        """Run one plain statement of the loop body; True if it computed.

        The iteration context (loop variable values and the iterable's
        lineage) goes into the cache key as a marker, so each iteration is
        its own entry.
        """
        context_hash = compute_context_hash(iteration_context)
        loop_vars = {k: v for k, v in iteration_context.items() if not k.startswith("__")}
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
        half); the tail takes the ordinary single-unit path.
        """

        try:
            head, tail = split_nodes(node, k)
        except ValueError:
            return None

        # A tail's accumulator is populated by its own head, so the shape is
        # read from the BODY rather than requiring a fresh preceding seed.
        force_outputs = None
        shape = accumulator_loop_body_shape(node)
        if shape is not None:
            acc, loop_vars = shape
            force_outputs = {acc, *loop_vars}
        if self.debug:
            logger.debug("[LOOP_SPLIT] executing split at k=%d (force_outputs=%s)", k, force_outputs)

        head_res = self.process(head, ttl, silent, parent_context, raw_cell, inherited_annotation)
        tail_res = self.dispatcher.execute_as_single_unit(
            tail,
            ttl,
            silent,
            raw_cell,
            inherited_annotation,
            force_outputs=force_outputs,
        )
        return ControlStructureResult(
            success=bool(head_res.success and tail_res.success),
            metrics=list(head_res.metrics) + list(tail_res.metrics),
            total_iterations=getattr(head_res, "total_iterations", 0) or 0,
        )
