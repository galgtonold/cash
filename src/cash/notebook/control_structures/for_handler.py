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
from typing import TYPE_CHECKING, Any

from . import helpers as _helpers
from ..cache_status import CacheStatus

if TYPE_CHECKING:
    from ..statement import ProcessResult

logger = logging.getLogger(__name__)


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
    ):
        """
        Process a for loop with per-iteration caching.

        For each iteration:
        1. Bind the loop target variable(s) into user_ns.
        2. Build an iteration context (target values + iterable lineage).
        3. Process each body statement through the statement processor with
           an ``# __iteration_context__: <hash>`` comment prepended.

        The statement processor's mutation detection (which runs before any
        cache lookup) will automatically set ``skip_cache=True`` for
        statements that mutate external variables, ensuring they always
        re-execute.

        **Fast-loop optimization**: When the estimated per-iteration caching
        overhead exceeds the likely computation cost (e.g., tight numeric
        loops with many iterations of cheap array operations), the loop is
        executed as a single cacheable unit instead of per-iteration.
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

            # Fast-loop heuristic: if per-iteration decomposition would be too
            # expensive relative to the computation, execute as a single unit.
            if self._should_execute_loop_as_single_unit(node, iterable, parent_context):
                if self.debug:
                    logger.debug("[CONTROL] Fast-loop: executing as single unit (overhead > benefit)")
                return self.dispatcher._execute_as_single_unit(node, ttl, silent)

            # all iterations.
            iterable_lineage = _helpers.get_iterable_lineage(
                self.shell, self.statement_processor, node.iter
            )
            if self.debug:
                logger.debug("[CONTROL] Iterable lineage: %s...",
                             iterable_lineage[:20] if iterable_lineage else 'None')

            for iteration_value in iterable:
                total_iterations += 1
                if self._process_one_iteration(
                    node, iteration_value, iterable_lineage, target_names,
                    ttl, silent, all_metrics, parent_context,
                ):
                    cached_iterations += 1
                else:
                    computed_iterations += 1

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
    ) -> bool:
        """Process a single loop iteration; return True if fully cached."""
        from .processor import (
            bind_target_values,
            build_iteration_context,
            compute_context_hash,
            is_control_structure,
        )

        bindings = bind_target_values(node.target, iteration_value, self.shell.user_ns)
        for name, val in bindings.items():
            try:
                h = val._cash_lineage_hash if hasattr(val, '_cash_lineage_hash') else self.statement_processor.compute_hash(val)
                self.statement_processor.variable_lineage[name] = h
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
        for body_idx, body_node in enumerate(node.body):
            before_count = len(all_metrics)
            if is_control_structure(body_node):
                was_computed = self._execute_loop_body_nested_control(
                    body_node, ttl, silent, iteration_context, context_hash, loop_vars, all_metrics,
                )
            else:
                was_computed = self._execute_loop_body_statement(
                    body_node, iteration_context, ttl, silent, all_metrics,
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
            if was_computed:
                iteration_cached = False
        return iteration_cached

    def _process_body_statement(
        self,
        code: str,
        iteration_context: dict[str, Any],
        ttl: int | None,
        silent: bool
    ) -> dict[str, Any]:
        """
        Process a single body statement with iteration context in the cache key.

        The iteration context (loop variable values + iterable lineage) is
        injected as a comment, making the cache key unique per-iteration.

        The statement processor's mutation detection will automatically detect
        statements like ``a.append(x)`` or ``d[k] = v`` and set
        ``skip_cache=True``, ensuring they always re-execute.
        """
        from .processor import compute_context_hash

        context_hash = compute_context_hash(iteration_context)
        modified_code = f"# __iteration_context__: {context_hash}\n{code}"

        result = self.statement_processor.process_statement(modified_code, ttl, silent)

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
    ) -> bool:
        """Process a nested control structure inside a for loop body.

        Recurses into the orchestrator's ``process``, injects the iteration
        context comment into nested metrics for correct badge grouping, and
        flushes output immediately for real-time streaming.

        Returns True if any nested iterations were computed (not cached).
        Raises on error, annotating the exception with the body node's line number.
        """
        result = self.dispatcher.process(body_node, ttl, silent, iteration_context)
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
    ) -> bool:
        """Process a plain (non-control-structure) statement inside a for loop body.

        Unparsed the node, delegates to ``_process_body_statement``, flushes
        output immediately, and raises on error — annotating the exception with
        the body node's source line number.

        Returns True if the statement was freshly computed (status == 'COMPUTED').
        """
        stmt_code = ast.unparse(body_node)
        metrics = self._process_body_statement(stmt_code, iteration_context, ttl, silent)
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
