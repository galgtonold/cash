from __future__ import annotations

"""For-loop per-iteration caching strategy.

**Boundary rule:** ``ForLoopHandler`` owns the per-iteration decomposition
of ``for`` loops — binding loop targets, building iteration contexts,
deciding when to fall back to single-unit mode, and dispatching each
body statement through the statement processor with an
``# __iteration_context__:`` cache-key discriminator.

It does NOT own:
- Shared lineage / mutation / badge helpers (in
  :mod:`control_structures.helpers`).
- Dispatch of nested ``if`` / ``try`` / single-unit fallback (delegated
  back through the orchestrator passed at construction time).

Tests can construct ``ForLoopHandler`` directly with mock dependencies
and exercise it without going through ``ControlStructureProcessor.process()``.
"""

import ast
import contextlib
import logging
import time as _time
from typing import TYPE_CHECKING, Any

from . import helpers as _helpers
from ..cache_status import CacheStatus

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

    # Approximate per-statement overhead in seconds (analysis + cache + capture).
    # Measured empirically: ~8-10ms per process() call.
    _PER_STMT_OVERHEAD_SEC = 0.008

    # If estimated overhead exceeds this factor times the estimated compute
    # cost, execute the loop as a single unit.
    _OVERHEAD_FACTOR_THRESHOLD = 3.0

    # Minimum number of iterations before single-unit mode is even considered.
    # Loops with few iterations benefit greatly from per-iteration caching
    # (granular invalidation, partial re-computation on changes) and the
    # absolute overhead is small regardless.
    _MIN_ITERATIONS_FOR_SINGLE_UNIT = 50

    # Minimum estimated overhead (in seconds) to trigger single-unit mode.
    _MIN_OVERHEAD_SEC = 1.0

    # -- Learning a loop split (CAS-261 step 2) -----------------------------
    #
    # The constants above are a STATIC guess -- an assumed per-statement cost
    # times an iteration count, decided without seeing the loop run. That
    # leaves a band where the guess says "decompose" and reality disagrees: n
    # below the single-unit threshold, every call below
    # ``call_unit._COST_FLOOR_S``, so neither mechanism caches anything while
    # per-iteration machinery is charged on every pass. Measured at n=124 on
    # a warm rerun against a cash-off arm, cash was SLOWER than not using
    # cash: 0.1ms body 22ms off vs 215ms on; 2.5ms body 320ms vs 617ms.
    #
    # This handler's whole role is to MEASURE such a loop and record a
    # verdict. It deliberately splits nothing: the split is applied by
    # ``upstream/virtual_lineage.py``, because the re-execution planner runs
    # the statements the simulator modelled. A runtime-side split is both
    # unnecessary and, on its own, a stale-value bug -- see
    # ``notebook/loop_split.py`` for the measurement.

    # Iterations measured before judging, and the split point thereafter.
    # Small on purpose: the head re-runs on every warm pass and per-iteration
    # overhead is exactly what the split removes, so a long head keeps the
    # cost it is meant to eliminate (k=10 measured ~25ms warm on a 0.1ms body
    # against 22ms for cash-off -- no gain at all; k=5 roughly halves it).
    _SPLIT_PROBE_ITERS = 5

    # At or above this, a call clears ``call_unit._COST_FLOOR_S`` once the
    # ~2.2ms of measured decomposition overhead is subtracted, so per-call
    # caching already covers it -- and covers it BETTER, being incremental
    # (append one item, re-run one call) where a single unit is
    # all-or-nothing. Splitting such a loop trades a better mechanism for a
    # worse one.
    _SPLIT_MAX_ITER_SEC = 0.006

    # Projected remaining cost below which a split is not worth a store.
    _SPLIT_MIN_REMAINING_SEC = 0.1

    # Builtin callables that PRODUCE an iterable without side effects, so the
    # single-unit fast path may re-evaluate the loop header a second time
    # (once here, once inside ``_execute_as_single_unit``) and get the same
    # iteration.  Any *other* call in the loop iterable (a bare user function
    # like ``drain()``, or an unknown name) may be a one-shot consumable whose
    # second evaluation drains an already-exhausted source — those are routed
    # to the per-iteration path, which iterates the single, already-evaluated
    # iterator.  Method calls (``df['c'].unique()``, ``d.items()``) are assumed
    # to be pure accessors and stay on the fast path so re-iterable containers
    # (ndarray/Series/DataFrame/dict views) keep the current behaviour.
    _PURE_ITER_PRODUCERS = frozenset({
        'range', 'sorted', 'reversed', 'list', 'tuple', 'set', 'frozenset',
        'dict', 'enumerate', 'zip', 'map', 'filter', 'iter', 'bytes',
        'bytearray', 'str',
    })

    def __init__(self, shell, statement_processor, debug: bool, dispatcher):
        self.shell = shell
        self.statement_processor = statement_processor
        self.debug = debug
        self.dispatcher = dispatcher

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
        from .processor import (
            ControlStructureResult,
            extract_target_names,
        )

        all_metrics: list[ProcessResult] = []
        total_iterations = 0
        cached_iterations = 0
        computed_iterations = 0

        target_names = extract_target_names(node.target)

        # A directive on the loop HEADER scopes to the loop, so it flows down
        # into every body statement. Resolved once here rather than per
        # iteration — it is a property of the source, not of the iteration.
        loop_annotation = _helpers.resolve_header_annotation(
            raw_cell, node, inherited_annotation,
        )

        if self.debug:
            logger.debug("[CONTROL] Processing FOR loop with targets: %s", target_names)

        try:
            # Evaluate the iterator
            iter_code = ast.unparse(node.iter)
            iterable = eval(iter_code, self.shell.user_ns, self.shell.user_ns)

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
            _verdict_k = self._recorded_split_k(node, iterable)
            if _verdict_k is not None:
                return self._run_split(node, _verdict_k, ttl, silent,
                                       parent_context, raw_cell,
                                       inherited_annotation)

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
            if (self._should_execute_loop_as_single_unit(node, iterable, parent_context)
                    and self._iter_header_safe_to_reevaluate(node.iter, iterable)):
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
                from ..cacheability import cacheable_accumulator_loop
                force_outputs = None
                acc_loop = cacheable_accumulator_loop(node, prev_node)
                if acc_loop is not None:
                    acc, loop_vars, _iter_node, _expr_call = acc_loop
                    force_outputs = {acc, *loop_vars}
                    if self.debug:
                        logger.debug(
                            "[CONTROL] Single-unit accumulator loop -> "
                            "force_outputs=%s", force_outputs,
                        )
                return self.dispatcher._execute_as_single_unit(
                    node, ttl, silent, raw_cell, inherited_annotation,
                    force_outputs=force_outputs,
                )

            # all iterations.
            iterable_lineage = _helpers.get_iterable_lineage(
                self.shell, self.statement_processor, node.iter
            )
            if self.debug:
                logger.debug("[CONTROL] Iterable lineage: %s...",
                             iterable_lineage[:20] if iterable_lineage else 'None')

            # Measure the first few iterations so this loop can be judged for
            # splitting on a LATER run. Nothing is split here.
            _probe_n = self._split_eligible(node, iterable)
            _probe_elapsed = 0.0

            for _idx, iteration_value in enumerate(iterable):
                total_iterations += 1
                _iter_started = _time.perf_counter()
                if self._process_one_iteration(
                    node, iteration_value, iterable_lineage, target_names,
                    ttl, silent, all_metrics, parent_context,
                    raw_cell, loop_annotation,
                ):
                    cached_iterations += 1
                else:
                    computed_iterations += 1
                if _probe_n is not None and _idx < self._SPLIT_PROBE_ITERS:
                    _probe_elapsed += _time.perf_counter() - _iter_started

            if _probe_n is not None:
                self._record_split_verdict(node, _probe_elapsed, _probe_n)

            # After all iterations, update lineage for mutated variables
            _helpers.update_lineage_after_execution(
                self.shell, self.statement_processor, node, ast.unparse(node), debug=self.debug
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
                computed_iterations=computed_iterations
            )

        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user for-loop body code
            logger.error("[CONTROL] Error in for loop: %s", e, exc_info=True)
            return ControlStructureResult(
                success=False,
                metrics=all_metrics,
                error=e
            )

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
        from .processor import (
            bind_target_values,
            build_iteration_context,
            compute_context_hash,
            is_control_structure,
        )

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
                from cash.notebook.object_hashing import compute_hash_full
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
                h = val._cash_lineage_hash if hasattr(val, '_cash_lineage_hash') else full
                self.statement_processor.variable_lineage[name] = h
                loop_var_digests[name] = full
            except (TypeError, ValueError, AttributeError) as exc:
                if self.debug:
                    logger.warning("[CONTROL] Failed to hash loop variable %s: %s", name, exc)

        iteration_context = build_iteration_context(target_names, self.shell.user_ns, parent_context)
        if iterable_lineage:
            iteration_context['__iterable_lineage__'] = iterable_lineage

        context_hash = compute_context_hash(iteration_context)
        loop_vars = {k: v for k, v in iteration_context.items() if not k.startswith('__')}
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
        loop_vars_scope = getattr(self.statement_processor, 'loop_vars_scope', None)
        scope = (
            loop_vars_scope(loop_vars, loop_var_digests)
            if loop_vars_scope is not None else contextlib.nullcontext()
        )
        with scope:
            for body_idx, body_node in enumerate(node.body):
                before_count = len(all_metrics)
                if is_control_structure(body_node):
                    was_computed = self._execute_loop_body_nested_control(
                        body_node, ttl, silent, iteration_context, context_hash, loop_vars, all_metrics,
                        raw_cell, loop_annotation,
                    )
                else:
                    was_computed = self._execute_loop_body_statement(
                        body_node, iteration_context, ttl, silent, all_metrics,
                        raw_cell, loop_annotation,
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

    def _process_body_statement(
        self,
        code: str,
        iteration_context: dict[str, Any],
        ttl: int | None,
        silent: bool,
        annotation=None,
    ) -> dict[str, Any]:
        """
        Process a single body statement with iteration context in the cache key.

        The iteration context (loop variable values + iterable lineage) is
        injected as a comment, making the cache key unique per-iteration.

        The statement processor's mutation detection will automatically detect
        statements like ``a.append(x)`` or ``d[k] = v`` and set
        ``skip_cache=True``, ensuring they always re-execute.

        *annotation* is the body statement's own ``@cash:`` directives, resolved
        by the caller against the original cell source. It cannot be recovered
        from *code*: ``ast.unparse`` drops comments, so by the time a body
        statement gets here its directive is already gone from the text.
        """
        from .processor import compute_context_hash

        context_hash = compute_context_hash(iteration_context)
        modified_code = f"# __iteration_context__: {context_hash}\n{code}"

        result = self.statement_processor.process_statement(
            modified_code, ttl, silent, annotation=annotation,
        )

        # Attach human-readable loop variable values to the metrics
        loop_vars = {
            k: v for k, v in iteration_context.items()
            if not k.startswith('__')
        }
        if loop_vars:
            result['loop_vars'] = loop_vars

        return result

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
        """Process a nested control structure inside a for loop body.

        Recurses into the orchestrator's ``process``, injects the iteration
        context comment into nested metrics for correct badge grouping, and
        flushes output immediately for real-time streaming.

        The enclosing loop's annotation is passed down as *inherited*, so a
        directive on the outer loop reaches statements in the inner one.

        Returns True if any nested iterations were computed (not cached).
        Raises on error, annotating the exception with the body node's line number.
        """
        result = self.dispatcher.process(
            body_node, ttl, silent, iteration_context, raw_cell, loop_annotation,
        )
        # Inject __iteration_context__ into nested metrics so the badge
        # renderer keeps them inside the loop group.
        for m in result.metrics:
            code = m.get('code', '')
            if '# __iteration_context__:' not in code:
                m['code'] = f"# __iteration_context__: {context_hash}\n{code}"
            if loop_vars and 'loop_vars' not in m:
                m['loop_vars'] = loop_vars
            if not m.get('_output_flushed'):
                _helpers.flush_metrics_output(m)
        all_metrics.extend(result.metrics)
        if not result.success:
            err = result.error or RuntimeError("Error in nested control structure")
            # Preserve _cash_error_lineno from nested error, or fall back to
            # this node's line number.
            if not hasattr(err, '_cash_error_lineno'):
                with contextlib.suppress(AttributeError, TypeError):
                    err._cash_error_lineno = getattr(body_node, 'lineno', None)
            raise err
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
        """Process a plain (non-control-structure) statement inside a for loop body.

        Unparsed the node, delegates to ``_process_body_statement``, flushes
        output immediately, and raises on error — annotating the exception with
        the body node's source line number.

        The statement's OWN annotation is resolved here, under the loop's. Body
        statements are separate cache entries, so a ``# @cash:no-cache`` on one
        must not leak onto its siblings — resolving per statement rather than
        applying the loop's whole-range scan is what keeps the sibling cached.

        Returns True if the statement was freshly computed (status == 'COMPUTED').
        """
        stmt_code = ast.unparse(body_node)
        annotation = _helpers.resolve_statement_annotation(
            raw_cell, body_node, loop_annotation,
        )
        metrics = self._process_body_statement(
            stmt_code, iteration_context, ttl, silent, annotation,
        )
        _helpers.flush_metrics_output(metrics)
        all_metrics.append(metrics)
        if metrics.get('status') == CacheStatus.ERROR:
            err = metrics.get('error', RuntimeError(f"Error executing: {stmt_code}"))
            # Annotate with the body statement's original line number from the
            # cell AST so _show_clean_error can point to the exact line, not
            # the for-loop header.
            with contextlib.suppress(AttributeError, TypeError):
                err._cash_error_lineno = getattr(body_node, 'lineno', None)
            raise err
        return metrics.get('status') == CacheStatus.COMPUTED

    # ------------------------------------------------------------------
    # Fast-loop heuristic
    # ------------------------------------------------------------------

    def _split_store(self):
        """Shared split store, or ``None`` if unresolvable (means: learn nothing)."""
        if getattr(self, "_split_store_cache", "unset") != "unset":
            return self._split_store_cache
        from ..loop_split import store_for_backend
        cash_instance = getattr(self.statement_processor, "cash_instance", None)
        self._split_store_cache = store_for_backend(
            getattr(cash_instance, "backend", None))
        return self._split_store_cache

    def _split_eligible(self, node: ast.For, iterable) -> int | None:
        """Iteration count if this loop could be split, else ``None``.

        Checked before measuring, so an ineligible loop pays only these
        predicates. Each exclusion is load-bearing:

        * **Already a half** -- splitting a half recurses.
        * **No** ``len()`` -- a generator cannot be sliced for the tail, and
          re-iterating it drains an exhausted source.
        * **Not sliceable** -- the tail's source is ``<iter expr>[k:]``;
          ``set``/``dict`` are sized but cannot form it.
        * **Header unsafe to re-evaluate** -- both halves evaluate it. Reuses
          the single-unit path's own guard, so there is one rule rather than
          two that can drift.
        * ``break``/``continue`` -- head+tail is NOT equivalent when a break
          in the head must skip the tail.
        * ``for ... else`` -- ``else`` has one completion point; a split has
          none.
        * **File I/O in the body** -- needs per-iteration dep tracking.
        """
        from ..loop_split import is_split_half
        if node.orelse or is_split_half(node):
            return None
        try:
            n = len(iterable)
        except TypeError:
            return None
        if n <= self._SPLIT_PROBE_ITERS:
            return None
        try:
            iterable[0:0]
        except Exception:  # noqa: BLE001 - any failure means not sliceable
            return None
        if not self._iter_header_safe_to_reevaluate(node.iter, iterable):
            return None
        if self._has_file_io_calls(node.body):
            return None
        for sub in ast.walk(ast.Module(body=list(node.body), type_ignores=[])):
            if isinstance(sub, (ast.Break, ast.Continue)):
                return None
        return n

    def _split_conf(self, name: str, default: float) -> float:
        """A split threshold, read from config on every decision.

        Not captured at construction, so ``cash.configure(...)`` takes effect
        immediately -- the contract ``min_execution_time_to_cache_seconds``
        already has. The class constants stay as the defaults and as the
        fallback when no config is reachable (the statement processor's
        ``cash_instance`` is a MagicMock in a fair number of unit tests).

        These are configurable because the judgement is a WALL-CLOCK
        measurement of the first few iterations, and wall clock is not a
        property of the code alone: a kernel descheduled mid-probe measures a
        cheap body as an expensive one and silently declines to split. A test
        cannot escape that by moving its workload, because the interesting
        cases sit BELOW the ceiling and descheduling only pushes measurements
        UP -- so the threshold is the only end that can be moved far enough.
        """
        cash_instance = getattr(self.statement_processor, "cash_instance", None)
        value = getattr(getattr(cash_instance, "config", None), name, None)
        # isinstance, NOT float(): a MagicMock's __float__ returns 1.0 rather
        # than raising, so a try/except would hand back 1.0 for every mocked
        # cash_instance -- a 1-second ceiling that silently changes the verdict.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return default

    def _should_split(self, elapsed: float, done: int, n: int) -> bool:
        """Judge from MEASURED cost whether the remainder is worth one unit.

        *elapsed* covers ``done`` iterations including per-iteration
        decomposition overhead -- correct, because that overhead is most of
        what a split recovers for a cheap body.
        """
        if done <= 0:
            return False
        max_iter = self._split_conf(
            "loop_split_max_iter_seconds", self._SPLIT_MAX_ITER_SEC)
        min_remaining = self._split_conf(
            "loop_split_min_remaining_seconds", self._SPLIT_MIN_REMAINING_SEC)
        per_iter = elapsed / done
        if per_iter >= max_iter:
            return False
        split = (n - done) * per_iter >= min_remaining
        if self.debug:
            logger.debug("[LOOP_SPLIT] per_iter=%.2fms remaining=%.0fms n=%d -> %s",
                         per_iter * 1000, (n - done) * per_iter * 1000, n, split)
        return split

    def _record_split_verdict(self, node: ast.For, elapsed: float, n: int) -> None:
        """Record that this loop should be split on later runs.

        Recording only -- the split is applied by the simulator; this handler
        never executes one.
        """
        if not self._should_split(elapsed, self._SPLIT_PROBE_ITERS, n):
            return
        store = self._split_store()
        if store is None:
            return
        try:
            from ..loop_split import loop_source_hash
            store.record(loop_source_hash(node), self._SPLIT_PROBE_ITERS)
            if self.debug:
                logger.debug("[LOOP_SPLIT] recorded k=%d; splits from next run",
                             self._SPLIT_PROBE_ITERS)
        except Exception:  # noqa: BLE001 - learning must never break execution
            logger.debug("[LOOP_SPLIT] could not record a verdict", exc_info=True)

    def _recorded_split_k(self, node: ast.For, iterable) -> int | None:
        """Persisted ``k`` for this loop, or ``None`` if it must not be split.

        Eligibility is re-checked against the LIVE iterable even when a
        verdict exists: the verdict was recorded for this loop's source, but
        the value bound to its iterable can change between runs (a list
        becoming a generator, say), and a tail is only derivable for a sized,
        sliceable one.
        """
        store = self._split_store()
        if store is None:
            return None
        if self._split_eligible(node, iterable) is None:
            return None
        try:
            from ..loop_split import loop_source_hash
            return store.get(loop_source_hash(node))
        except Exception:  # noqa: BLE001 - a lookup must never break the loop
            logger.debug("[LOOP_SPLIT] verdict lookup failed", exc_info=True)
            return None

    def _run_split(self, node, k, ttl, silent, parent_context, raw_cell,
                   inherited_annotation):
        """Execute a split loop as head + tail.

        Halves come from ``loop_split.split_nodes`` -- the same derivation the
        simulator uses -- so both sides run the same two statements. The head
        recurses through :meth:`process` (and is itself unsplittable, being a
        half); the tail takes the ordinary single-unit path.
        """
        from ..cacheability import accumulator_loop_body_shape
        from ..loop_split import split_nodes
        from .processor import ControlStructureResult

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
            logger.debug("[LOOP_SPLIT] executing split at k=%d (force_outputs=%s)",
                         k, force_outputs)

        head_res = self.process(head, ttl, silent, parent_context, raw_cell,
                                inherited_annotation)
        tail_res = self.dispatcher._execute_as_single_unit(
            tail, ttl, silent, raw_cell, inherited_annotation,
            force_outputs=force_outputs,
        )
        return ControlStructureResult(
            success=bool(head_res.success and tail_res.success),
            metrics=list(head_res.metrics) + list(tail_res.metrics),
            total_iterations=getattr(head_res, "total_iterations", 0) or 0,
        )

    def _iter_header_safe_to_reevaluate(self, iter_node: ast.AST, iterable) -> bool:
        """Whether the loop header may be safely evaluated a second time.

        The single-unit fast path re-executes the loop from source, evaluating
        ``iter_node`` again after :meth:`process` already evaluated it once.
        That is only correct when the second evaluation reproduces the same
        iteration — i.e. the header is a re-iterable container built by a
        side-effect-free expression.

        Returns ``False`` (route to the per-iteration path, which consumes the
        single already-evaluated iterator) when EITHER:

        * the evaluated value is a *self-iterator* — ``iter(x) is x`` — a
          generator, ``map``/``zip``/``filter``/``enumerate``, an open file,
          a csv reader, or ``iter(...)``: iterating it a second time yields
          nothing because the first pass exhausted it; OR
        * the header contains a *call to a bare name that is not a known-pure
          iterable producer* (``drain()``, ``next_batch()``): such a call may
          mutate/consume external state, so a second evaluation returns a
          different (often empty) result.

        Method calls (``df['c'].unique()``, ``d.items()``) are treated as pure
        accessors and kept on the fast path, so re-iterable containers keep
        the current byte-identical behaviour.
        """
        # One-shot self-iterators: re-iterating drains an exhausted source.
        # (Most lack ``__len__`` and never reach the single-unit heuristic, but
        # a custom self-iterator that defines ``__len__`` would — guard it.)
        try:
            if iter(iterable) is iterable:
                return False
        except Exception:  # noqa: BLE001 - defensive; non-iterables fail later anyway
            pass

        # A bare-name call to anything other than a known side-effect-free
        # iterable producer may consume/mutate state on re-evaluation.
        for sub in ast.walk(iter_node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id not in self._PURE_ITER_PRODUCERS:
                    return False
        return True

    def _should_execute_loop_as_single_unit(
        self,
        node: ast.For,
        iterable,
        parent_context: dict[str, Any] | None
    ) -> bool:
        """
        Decide whether a for loop should be executed as a single cacheable
        unit instead of being decomposed per-iteration.

        Per-iteration decomposition adds ~8-10ms of overhead per body
        statement (AST analysis, cache key computation, mutation detection,
        side-effect scanning, analytics recording, cache I/O).  For tight
        numeric loops with many iterations of cheap operations, this overhead
        can be 100-300× the actual compute time.

        Heuristic: execute as single unit when ALL of:
        1. The loop has many iterations (> _MIN_ITERATIONS_FOR_SINGLE_UNIT)
           — small loops always benefit from per-iteration caching since the
           absolute overhead is small and granular invalidation is valuable.
        2. The estimated overhead (iterations × body_stmts × per_stmt_cost) is
           significant (> 1s).
        3. No body statement performs file I/O — file dependencies need
           per-iteration tracking.

        Previously there was also a "not nested inside another loop" rule.
        That was over-defensive: the outer loop in a convergence study
        benefits from per-iteration caching (and rule 1 keeps it
        per-iteration if it has few iterations), but a tight inner loop
        with 200+ trivial iterations costs ~14s of per-statement machinery
        and gains nothing from per-iteration cache granularity since users
        rarely edit inner-loop bodies during exploration. The outer loop's
        per-iteration cache key still correctly invalidates on inner-body
        changes because the inner loop's single-unit cache key is part of
        the outer iteration's snapshot.

        **Correctness tradeoff**: In fast-loop mode the loop executes as one
        opaque unit, so per-iteration cache keys are not computed.  This means:

        - Resuming a partial loop (e.g., adding 3 more items to ``range(100)``)
          re-runs ALL iterations instead of only the new ones.
        - Variables mutated inside the loop (e.g., ``results.append(x)``) still
          get correct lineage because ``_update_lineage_after_execution`` tracks
          the whole-loop output.
        - File-I/O statements are excluded (rule 3) to preserve per-file mtime
          invalidation.

        The performance gain (up to 300×) justifies this tradeoff for tight
        numeric loops where per-iteration staleness detection has no value.
        """
        # (Nesting check intentionally removed — see docstring.)

        # Estimate iteration count
        try:
            n_iterations = len(iterable)
        except TypeError:
            # Generators, iterators without __len__ — can't estimate
            return False

        # Small loops always benefit from per-iteration caching — the
        # absolute overhead is small and granular invalidation is valuable.
        if n_iterations <= self._MIN_ITERATIONS_FOR_SINGLE_UNIT:
            return False

        # Count body statements (including nested control structure bodies)
        n_body_stmts = self._count_body_statements(node.body)

        # Estimate overhead
        estimated_overhead = n_iterations * n_body_stmts * self._PER_STMT_OVERHEAD_SEC

        if estimated_overhead < self._MIN_OVERHEAD_SEC:
            return False

        # Check for file I/O patterns — those need per-iteration tracking
        if self._has_file_io_calls(node.body):
            return False

        if self.debug:
            logger.debug(
                "[FAST_LOOP] Estimated overhead: %.1fs (%s iters × %s stmts × %.0fms/stmt)",
                estimated_overhead, n_iterations, n_body_stmts, self._PER_STMT_OVERHEAD_SEC*1000
            )

        return True

    def _count_body_statements(self, body: list[ast.AST]) -> int:
        """Count the total number of executable statements in a loop body,
        including statements inside nested control structures."""
        count = 0
        for node in body:
            if isinstance(node, ast.If):
                count += self._count_body_statements(node.body)
                count += self._count_body_statements(node.orelse)
            elif isinstance(node, (ast.For, ast.While)):
                count += self._count_body_statements(node.body)
            elif isinstance(node, ast.Try):
                count += self._count_body_statements(node.body)
                for handler in node.handlers:
                    count += self._count_body_statements(handler.body)
            elif isinstance(node, ast.With):
                count += self._count_body_statements(node.body)
            else:
                count += 1
        return count

    def _has_file_io_calls(self, body: list[ast.AST]) -> bool:
        """Check if any statement in the body performs file I/O.

        File I/O needs per-iteration tracking for proper dependency
        invalidation, so loops with file operations should not use the
        fast-loop optimization.

        Returns True if file I/O patterns are detected."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name is None:
                    continue

                # Check for file I/O patterns
                if func_name in ('open', 'read', 'write', 'read_csv',
                                 'to_csv', 'read_excel', 'to_excel',
                                 'save', 'load', 'savez', 'savetxt',
                                 'loadtxt', 'read_parquet', 'to_parquet'):
                    return True

        return False

    @staticmethod
    def _get_call_name(node: ast.Call) -> str | None:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
