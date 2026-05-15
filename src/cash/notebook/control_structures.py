from __future__ import annotations

"""
Control Structure Processing for Statement-Level Caching

This module provides handlers for processing control structures (for, while, if,
with, try).

**For loops** are decomposed per-iteration: each body statement is passed to the
statement processor individually, with an iteration context marker in the code.
This ensures:
- Expensive pure computations inside loops are cached per-iteration.
- Statements that mutate external variables (e.g., ``a.append(x)``) are detected
  by the statement processor's mutation detection and executed directly (no cache).
- If a loop iteration was cached previously and nothing changed, it restores
  instantly.

**If statements** and **try/except blocks** are decomposed per-statement:
each branch statement is processed individually with a control-context marker.
This gives correct per-statement caching, badge display, and ensures that
side-effect statements like ``print()`` always execute.

**All other control structures** (while, with) are executed as single cacheable
units — the entire code is passed to the statement processor, which handles
cache key computation, mutation detection, and side-effect checking.

Loops containing ``break`` or ``continue`` are also executed as single units,
because decomposing them per-iteration is not possible (those statements must
execute inside a loop context).
"""

import ast
import contextlib
import hashlib
import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cash.notebook.cache_status import CacheStatus

if TYPE_CHECKING:
    from .statement_processor import ProcessResult

__all__ = ["ControlStructureResult", "ControlStructureProcessor", "is_control_structure", "get_control_structure_type", "contains_break_or_continue", "extract_target_names", "bind_target_values", "build_iteration_context", "compute_context_hash"]

logger = logging.getLogger(__name__)

@dataclass
class ControlStructureResult:
    """Result from executing a control structure."""
    success: bool
    metrics: list[ProcessResult]  # Metrics for each processed statement
    error: Exception | None = None
    total_iterations: int = 0
    cached_iterations: int = 0
    computed_iterations: int = 0

def is_control_structure(node: ast.AST) -> bool:
    """Check if an AST node is a control structure that should be processed."""
    return isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try))

def get_control_structure_type(node: ast.AST) -> str | None:
    """Get the type of control structure for an AST node."""
    if isinstance(node, ast.For):
        return 'for'
    if isinstance(node, ast.While):
        return 'while'
    if isinstance(node, ast.If):
        return 'if'
    if isinstance(node, ast.With):
        return 'with'
    if isinstance(node, ast.Try):
        return 'try'
    return None

def contains_break_or_continue(nodes: list[ast.AST]) -> bool:
    """
    Check if any of the given AST nodes (or their children) contain break or
    continue statements.

    These statements cannot be executed outside of a loop context, so loops
    containing them must be executed as a single unit.
    """
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, (ast.Break, ast.Continue)):
                return True
    return False

# --- Helper Functions (Module Level) ---

def extract_target_names(target: ast.AST) -> list[str]:
    """Extract variable names from a for loop target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(extract_target_names(elt))
        return names
    return []

def bind_target_values(target: ast.AST, value, user_ns: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively bind loop target variables to their values,
    respecting nested tuple/list unpacking structure.

    For ``for a, (b, c) in data``, the AST target is
    Tuple(Name('a'), Tuple(Name('b'), Name('c'))) and the
    iteration value is e.g. (1, (2, 3)).  This function
    correctly maps a->1, b->2, c->3 and assigns into user_ns.

    Returns dict of {name: value} for all bound variables.
    """
    bindings: dict[str, Any] = {}
    if isinstance(target, ast.Name):
        user_ns[target.id] = value
        bindings[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        vals = list(value)
        for elt, val in zip(target.elts, vals, strict=False):
            bindings.update(bind_target_values(elt, val, user_ns))
    elif isinstance(target, ast.Starred):
        if isinstance(target.value, ast.Name):
            user_ns[target.value.id] = value
            bindings[target.value.id] = value
    return bindings

def build_iteration_context(
    target_names: list[str],
    user_ns: dict[str, Any],
    parent_context: dict[str, Any] | None
) -> dict[str, Any]:
    """
    Build a context dict containing current iteration variable values.

    Used to differentiate cache keys across loop iterations.
    """
    context = dict(parent_context) if parent_context else {}

    for name in target_names:
        if name in user_ns:
            value = user_ns[name]
            try:
                hash(value)
                context[name] = value
            except TypeError:
                context[name] = repr(value)

    return context

def compute_context_hash(context: dict[str, Any]) -> str:
    """Compute a hash of the iteration context."""
    items = sorted(context.items())
    context_str = str(items)
    return hashlib.sha256(context_str.encode('utf-8')).hexdigest()[:16]

class ControlStructureProcessor:
    """
    Processor for control structures.

    **For loops** are decomposed per-iteration.  Each body statement is passed
    to the statement processor with an iteration-context comment injected into
    the code.  The statement processor's existing mutation detection decides
    per-statement whether caching is safe:

    - Statements that only *read* external variables and assign to local
      outputs are cached per-iteration (e.g. ``stats = expensive(...)``).
    - Statements that *mutate* external variables in-place (e.g.
      ``a.append(x)``, ``d[k] = v``) get ``skip_cache=True`` from the
      mutation detector and are executed directly every time.

    **If and try/except** are decomposed per-statement for correct caching
    and output handling.  **Other control structures** (while, with) are
    executed as single cacheable units through the statement processor.
    """

    def __init__(
        self,
        shell,
        statement_processor,  # The StatementProcessor instance
        debug: bool = False
    ):
        self.shell = shell
        self.statement_processor = statement_processor
        self.debug = debug

    @staticmethod
    def _flush_metrics_output(metrics: dict[str, Any]) -> None:
        """Immediately print stdout/stderr from a metrics dict.

        This is called after each body statement so that output streams
        in real-time instead of being batched until the entire control
        structure finishes.  The metrics dict is marked with
        ``_output_flushed=True`` so that the caller in ``magics.py``
        does not replay the same output a second time.
        """
        if metrics.get('stdout'):
            print(metrics['stdout'], end='', flush=True)
        if metrics.get('stderr'):
            print(metrics['stderr'], end='', file=sys.stderr, flush=True)
        metrics['_output_flushed'] = True

    def process(
        self,
        node: ast.AST,
        ttl: int | None = None,
        silent: bool = False,
        parent_context: dict[str, Any] | None = None
    ) -> ControlStructureResult:
        """
        Process a control structure node.

        For loops are decomposed per-iteration; other control structures are
        executed as single units.

        Args:
            node: The AST node representing the control structure
            ttl: Time-to-live for cache entries
            silent: Suppress output
            parent_context: Iteration context from an enclosing loop (for nesting)

        Returns:
            ControlStructureResult with metrics
        """
        if isinstance(node, ast.For):
            # For loops with break/continue must be executed as single units
            if contains_break_or_continue(node.body):
                if self.debug:
                    logger.debug("[CONTROL] Loop contains break/continue, executing as single unit")
                return self._execute_as_single_unit(node, ttl, silent)
            return self._process_for(node, ttl, silent, parent_context)
        if isinstance(node, ast.If):
            return self._execute_if_per_statement(node, ttl, silent)
        if isinstance(node, ast.Try):
            return self._execute_try_per_statement(node, ttl, silent)
        return self._execute_as_single_unit(node, ttl, silent)

    # ------------------------------------------------------------------
    # For-loop per-iteration processing
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
        for body_node in node.body:
            if is_control_structure(body_node):
                was_computed = self._execute_loop_body_nested_control(
                    body_node, ttl, silent, iteration_context, context_hash, loop_vars, all_metrics,
                )
            else:
                was_computed = self._execute_loop_body_statement(
                    body_node, iteration_context, ttl, silent, all_metrics,
                )
            if was_computed:
                iteration_cached = False
        return iteration_cached

    def _process_for(
        self,
        node: ast.For,
        ttl: int | None,
        silent: bool,
        parent_context: dict[str, Any] | None
    ) -> ControlStructureResult:
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

            # Fast-loop heuristic: if per-iteration decomposition would be too
            # expensive relative to the computation, execute as a single unit.
            if self._should_execute_loop_as_single_unit(node, iterable, parent_context):
                if self.debug:
                    logger.debug("[CONTROL] Fast-loop: executing as single unit (overhead > benefit)")
                return self._execute_as_single_unit(node, ttl, silent)

            # all iterations.
            iterable_lineage = self._get_iterable_lineage(node.iter)
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
            self._update_lineage_after_execution(node, ast.unparse(node))

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
        context_hash = compute_context_hash(iteration_context)
        modified_code = f"# __iteration_context__: {context_hash}\n{code}"

        result = self.statement_processor.process_statement(modified_code, ttl, silent, render_badge=False)

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
        all_metrics: list[ProcessResult],
    ) -> bool:
        """Process a nested control structure inside a for loop body.

        Recurses into ``self.process``, injects the iteration context comment
        into nested metrics for correct badge grouping, and flushes output
        immediately for real-time streaming.

        Returns True if any nested iterations were computed (not cached).
        Raises on error, annotating the exception with the body node's line number.
        """
        result = self.process(body_node, ttl, silent, iteration_context)
        # Inject __iteration_context__ into nested metrics so the badge
        # renderer keeps them inside the loop group.
        for m in result.metrics:
            code = m.get('code', '')
            if '# __iteration_context__:' not in code:
                m['code'] = f"# __iteration_context__: {context_hash}\n{code}"
            if loop_vars and 'loop_vars' not in m:
                m['loop_vars'] = loop_vars
            if not m.get('_output_flushed'):
                self._flush_metrics_output(m)
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
        all_metrics: list[ProcessResult],
    ) -> bool:
        """Process a plain (non-control-structure) statement inside a for loop body.

        Unparsed the node, delegates to ``_process_body_statement``, flushes
        output immediately, and raises on error — annotating the exception with
        the body node's source line number.

        Returns True if the statement was freshly computed (status == 'COMPUTED').
        """
        stmt_code = ast.unparse(body_node)
        metrics = self._process_body_statement(stmt_code, iteration_context, ttl, silent)
        self._flush_metrics_output(metrics)
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
        1. The loop is NOT nested inside another loop (parent_context is None)
           — nested loops in convergence studies benefit from per-iteration caching.
        2. The loop has many iterations (> _MIN_ITERATIONS_FOR_SINGLE_UNIT)
           — small loops always benefit from per-iteration caching since the
           absolute overhead is small and granular invalidation is valuable.
        3. The estimated overhead (iterations × body_stmts × per_stmt_cost) is
           significant (> 1s).
        4. No body statement performs file I/O — file dependencies need
           per-iteration tracking.

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
        # Don't apply inside nested loops — the outer loop already controls
        # the iteration count, and inner loops in convergence studies should
        # still be cached per-iteration.
        if parent_context is not None:
            return False

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

    # ------------------------------------------------------------------
    # If/elif/else per-statement processing
    # ------------------------------------------------------------------

    def _execute_if_branch_stmt(
        self,
        body_node: ast.AST,
        branch_hash: str,
        branch_label: str,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
    ) -> bool:
        """Execute one statement from an if-branch; return True if computed (not cached)."""
        if is_control_structure(body_node):
            result = self.process(body_node, ttl, silent, None)
            self._tag_control_metrics(result, branch_hash, branch_label, all_metrics)
            if not result.success:
                err = result.error or RuntimeError("Error in nested control structure")
                if not hasattr(err, '_cash_error_lineno'):
                    with contextlib.suppress(AttributeError, TypeError):
                        err._cash_error_lineno = getattr(body_node, 'lineno', None)
                raise err
            return result.computed_iterations > 0
        stmt_code = ast.unparse(body_node)
        modified_code = f"# control_context: {branch_hash}\n{stmt_code}"
        metrics = self.statement_processor.process_statement(modified_code, ttl, silent, render_badge=False)
        metrics['control_context'] = branch_hash
        metrics['branch_label'] = branch_label
        self._flush_metrics_output(metrics)
        all_metrics.append(metrics)
        if metrics.get('status') == CacheStatus.ERROR:
            err = metrics.get('error', RuntimeError(f"Error executing: {stmt_code}"))
            with contextlib.suppress(AttributeError, TypeError):
                err._cash_error_lineno = getattr(body_node, 'lineno', None)
            raise err
        return metrics.get('status') == CacheStatus.COMPUTED

    def _execute_if_per_statement(
        self,
        node: ast.If,
        ttl: int | None,
        silent: bool,
    ) -> ControlStructureResult:
        """Execute an if/elif/else by processing each branch statement individually.

        1. Walk the if/elif/else chain evaluating conditions until one is True
           (or we fall through to else).
        2. Execute each statement in the taken branch individually through the
           statement processor, tagging metrics with ``control_context``
           so the badge can group them.
        3. Only the executed branch is shown in the badge.

        Falls back to ``_execute_as_single_unit`` if condition evaluation fails.
        """
        all_metrics: list[ProcessResult] = []
        cached_count = 0
        computed_count = 0

        try:
            # Walk if/elif/else chain to find the taken branch
            branch_body, branch_label = self._find_taken_branch(node)

            if self.debug:
                logger.debug("[CONTROL] If per-statement: taking branch '%s', %s statements",
                             branch_label, len(branch_body))

            # Build a context hash for cache key uniqueness (which branch)
            branch_hash = hashlib.sha256(branch_label.encode()).hexdigest()[:16]

            # Build the full body_statements for badge display (only executed branch)
            body_stmts = [f"{branch_label}:"]
            for body_node in branch_body:
                body_stmts.append(f"  {ast.unparse(body_node)}")

            for body_node in branch_body:
                was_computed = self._execute_if_branch_stmt(
                    body_node, branch_hash, branch_label, ttl, silent, all_metrics
                )
                if was_computed:
                    computed_count += 1
                else:
                    cached_count += 1

            # After execution, update lineage for mutated variables
            self._update_lineage_after_execution(node, ast.unparse(node))

            # Tag all metrics with body statements for the whole if block
            # (so the badge can show the header and which branch we took)
            for m in all_metrics:
                m['control_type'] = 'if'
                if 'body_statements' not in m:
                    m['body_statements'] = body_stmts

            return ControlStructureResult(
                success=True,
                metrics=all_metrics,
                total_iterations=1,
                cached_iterations=1 if computed_count == 0 and all_metrics else 0,
                computed_iterations=1 if computed_count > 0 else 0,
            )

        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user if-branch code
            logger.error("[CONTROL] Error in if per-statement execution: %s", e, exc_info=True)
            return ControlStructureResult(
                success=False,
                metrics=all_metrics,
                error=e,
            )

    def _find_taken_branch(
        self, node: ast.If
    ) -> tuple[list[ast.AST], str]:
        """Walk if/elif/else chain and return (body_nodes, label) of the taken branch.

        Evaluates each condition in the user namespace. Returns the first branch
        whose condition is truthy, or the else branch if no condition matches.

        Raises RuntimeError if no branch is taken (shouldn't happen with else).
        """
        current = node
        branch_idx = 0
        while True:
            test_code = ast.unparse(current.test)
            try:
                condition_result = eval(test_code, self.shell.user_ns, self.shell.user_ns)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to evaluate if condition: {test_code}"
                ) from e

            if condition_result:
                keyword = "elif" if branch_idx > 0 else "if"
                label = f"{keyword} {test_code}"
                return current.body, label

            # Move to elif/else
            if current.orelse:
                if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    # elif chain
                    current = current.orelse[0]
                    branch_idx += 1
                else:
                    # else branch
                    return current.orelse, "else"
            else:
                # No else and no condition matched → empty body
                return [], "if (no branch taken)"

    # ------------------------------------------------------------------
    # Try/except/else/finally per-statement processing
    # ------------------------------------------------------------------

    def _tag_control_metrics(
        self, result: Any, ctx_hash: str, ctx_label: str, all_metrics: list
    ) -> None:
        """Tag and flush metrics from a nested control-structure result."""
        for m in result.metrics:
            if 'control_context' not in m:
                m['control_context'] = ctx_hash
                m['branch_label'] = ctx_label
            if not m.get('_output_flushed'):
                self._flush_metrics_output(m)
        all_metrics.extend(result.metrics)

    def _execute_simple_branch(
        self,
        body_nodes: list,
        ctx_hash: str,
        ctx_label: str,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
    ) -> tuple[int, int]:
        """Execute body nodes under a context hash; return (cached_count, computed_count).

        Used for else and finally branches of try/except where no per-statement
        lineno tagging is needed.
        """
        cached = computed = 0
        for body_node in body_nodes:
            if is_control_structure(body_node):
                result = self.process(body_node, ttl, silent, None)
                self._tag_control_metrics(result, ctx_hash, ctx_label, all_metrics)
                if not result.success:
                    raise result.error or RuntimeError("Error in nested control structure")
                if result.computed_iterations > 0:
                    computed += 1
                else:
                    cached += 1
            else:
                stmt_code = ast.unparse(body_node)
                modified_code = f"# control_context: {ctx_hash}\n{stmt_code}"
                metrics = self.statement_processor.process_statement(modified_code, ttl, silent, render_badge=False)
                metrics['control_context'] = ctx_hash
                metrics['branch_label'] = ctx_label
                self._flush_metrics_output(metrics)
                all_metrics.append(metrics)
                if metrics.get('status') == CacheStatus.ERROR:
                    raise metrics.get('error', RuntimeError(f"Error executing: {stmt_code}"))
                if metrics.get('status') == CacheStatus.COMPUTED:
                    computed += 1
                elif metrics.get('status') in (CacheStatus.RESTORED, CacheStatus.SKIPPED):
                    cached += 1
        return cached, computed

    def _execute_try_body_stmts(
        self,
        node: ast.Try,
        branch_hash: str,
        branch_label: str,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
    ) -> tuple[bool, Exception | None, int, int]:
        """Execute the try-body statements; return (succeeded, caught_exc, cached, computed)."""
        cached = computed = 0
        try_body_succeeded = True
        caught_exception: Exception | None = None
        for body_node in node.body:
            if is_control_structure(body_node):
                try:
                    result = self.process(body_node, ttl, silent, None)
                    self._tag_control_metrics(result, branch_hash, branch_label, all_metrics)
                    if not result.success:
                        caught_exception = result.error or RuntimeError("Error in nested control structure")
                        try_body_succeeded = False
                        break
                    if result.computed_iterations > 0:
                        computed += 1
                    else:
                        cached += 1
                except Exception as e:  # noqa: BLE001 - catching user-raised exceptions from nested control structures
                    caught_exception = e
                    try_body_succeeded = False
                    break
            else:
                stmt_code = ast.unparse(body_node)
                modified_code = f"# control_context: {branch_hash}\n{stmt_code}"
                try:
                    metrics = self.statement_processor.process_statement(modified_code, ttl, silent, render_badge=False)
                except Exception as e:  # noqa: BLE001 - catching user-raised exceptions from statement execution
                    caught_exception = e
                    try_body_succeeded = False
                    break
                metrics['control_context'] = branch_hash
                metrics['branch_label'] = branch_label
                self._flush_metrics_output(metrics)
                all_metrics.append(metrics)
                if metrics.get('status') == CacheStatus.ERROR:
                    caught_exception = metrics.get('error', RuntimeError(f"Error executing: {stmt_code}"))
                    try_body_succeeded = False
                    break
                if metrics.get('status') == CacheStatus.COMPUTED:
                    computed += 1
                elif metrics.get('status') in (CacheStatus.RESTORED, CacheStatus.SKIPPED):
                    cached += 1
        return try_body_succeeded, caught_exception, cached, computed

    def _bind_exception_to_handler(
        self, matched_handler: ast.ExceptHandler, caught_exception: Exception
    ) -> None:
        """Bind the caught exception to the handler's variable and set its lineage."""
        if not matched_handler.name:
            return
        self.shell.user_ns[matched_handler.name] = caught_exception
        try:
            exc_class_name = type(caught_exception).__name__
            class_lineage = self.statement_processor.variable_lineage.get(exc_class_name, '')
            exc_lineage = hashlib.sha256(
                f"__exception__:{exc_class_name}:{class_lineage}:{caught_exception!s}:{caught_exception!r}".encode()
            ).hexdigest()
            self.statement_processor.variable_lineage[matched_handler.name] = exc_lineage
            with contextlib.suppress(AttributeError, TypeError):
                caught_exception._cash_lineage_hash = exc_lineage
        except (ValueError, AttributeError, TypeError) as exc:
            logger.debug("[CONTROL] Failed to compute exception lineage for handler variable: %s", exc)

    def _execute_try_per_statement(
        self,
        node: ast.Try,
        ttl: int | None,
        silent: bool,
    ) -> ControlStructureResult:
        """Execute a try/except/else/finally by processing each statement individually.

        1. Execute try body statements one by one.
        2. If a statement raises an exception, find the matching except handler
           and execute its body statements.
        3. If no exception, execute the else body (if present).
        4. Always execute the finally body (if present).
        5. Build ``body_stmts`` showing only the actually-executed parts.

        This mirrors ``_execute_if_per_statement`` so that each sub-statement
        gets its own cache key, storage info, and timing — and ``print()``
        calls are never suppressed by the SKIPPED optimisation.
        """
        all_metrics: list[ProcessResult] = []
        cached_count = 0
        computed_count = 0

        branch_label = "try"
        caught_exception = None
        matched_handler = None

        try:
            branch_hash = hashlib.sha256(branch_label.encode()).hexdigest()[:16]
            try_body_succeeded, caught_exception, c, d = self._execute_try_body_stmts(
                node, branch_hash, branch_label, ttl, silent, all_metrics
            )
            cached_count += c
            computed_count += d

            # If an exception occurred, find and execute the matching handler
            if caught_exception is not None and node.handlers:
                matched_handler = self._find_matching_handler(node.handlers, caught_exception)
                if matched_handler is not None:
                    handler_label = self._format_handler_label(matched_handler)
                    branch_label = handler_label
                    handler_hash = hashlib.sha256(handler_label.encode()).hexdigest()[:16]
                    self._bind_exception_to_handler(matched_handler, caught_exception)
                    c, d = self._execute_simple_branch(
                        matched_handler.body, handler_hash, handler_label, ttl, silent, all_metrics
                    )
                    cached_count += c
                    computed_count += d
                    caught_exception = None
                else:
                    raise caught_exception

            if try_body_succeeded and node.orelse:
                else_label = "else"
                else_hash = hashlib.sha256(else_label.encode()).hexdigest()[:16]
                c, d = self._execute_simple_branch(node.orelse, else_hash, else_label, ttl, silent, all_metrics)
                cached_count += c
                computed_count += d

            if getattr(node, 'finalbody', None):
                finally_label = "finally"
                finally_hash = hashlib.sha256(finally_label.encode()).hexdigest()[:16]
                c, d = self._execute_simple_branch(node.finalbody, finally_hash, finally_label, ttl, silent, all_metrics)
                cached_count += c
                computed_count += d

            if caught_exception is not None:
                raise caught_exception

            self._update_lineage_after_execution(node, ast.unparse(node))
            body_stmts = self._build_try_executed_body_stmts(node, try_body_succeeded, matched_handler)

            for m in all_metrics:
                m['control_type'] = 'try'
                if 'body_statements' not in m:
                    m['body_statements'] = body_stmts

            return ControlStructureResult(
                success=True,
                metrics=all_metrics,
                total_iterations=1,
                cached_iterations=1 if computed_count == 0 and all_metrics else 0,
                computed_iterations=1 if computed_count > 0 else 0,
            )

        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user try/except body code
            logger.error("[CONTROL] Error in try per-statement execution: %s", e, exc_info=True)
            return ControlStructureResult(
                success=False,
                metrics=all_metrics,
                error=e,
            )

    def _find_matching_handler(
        self, handlers: list[ast.ExceptHandler], exc: Exception
    ) -> ast.ExceptHandler | None:
        """Find the first except handler that matches the given exception.

        Returns None if no handler matches.
        """
        for handler in handlers:
            if handler.type is None:
                # Bare except: catches everything
                return handler
            try:
                exc_type_code = ast.unparse(handler.type)
                exc_type = eval(exc_type_code, self.shell.user_ns, self.shell.user_ns)
                if isinstance(exc, exc_type):
                    return handler
            except (NameError, AttributeError, TypeError, ValueError) as exc:
                # Can't evaluate handler type — skip
                logger.debug("[CONTROL] Failed to evaluate except type: %s: %s", ast.unparse(handler.type), exc)
                continue
        return None

    @staticmethod
    def _format_handler_label(handler: ast.ExceptHandler) -> str:
        """Format a human-readable label for an except handler."""
        if handler.type is None:
            label = "except"
        else:
            label = f"except {ast.unparse(handler.type)}"
            if handler.name:
                label += f" as {handler.name}"
        return label

    def _build_try_executed_body_stmts(
        self,
        node: ast.Try,
        try_body_succeeded: bool,
        matched_handler: ast.ExceptHandler | None,
    ) -> list[str]:
        """Build body_stmts list showing only the actually-executed branches.

        For badge display: shows the try body (always started), the handler
        if one was matched, else if try succeeded, and finally (always).
        """
        statements: list[str] = []

        # Always show the try body (it was at least partially executed)
        statements.append("try:")
        for stmt in node.body:
            statements.append(f"  {ast.unparse(stmt)}")

        # Show the matched handler (if any)
        if matched_handler is not None:
            label = self._format_handler_label(matched_handler)
            statements.append(f"{label}:")
            for stmt in matched_handler.body:
                statements.append(f"  {ast.unparse(stmt)}")

        # Show else if try succeeded and else exists
        if try_body_succeeded and node.orelse:
            statements.append("else:")
            for stmt in node.orelse:
                statements.append(f"  {ast.unparse(stmt)}")

        # Always show finally
        if getattr(node, 'finalbody', None):
            statements.append("finally:")
            for stmt in node.finalbody:
                statements.append(f"  {ast.unparse(stmt)}")

        return statements

    # ------------------------------------------------------------------
    # Single-unit execution (for while/with and break/continue loops)
    # ------------------------------------------------------------------

    def _execute_as_single_unit(
        self,
        node: ast.AST,
        ttl: int | None,
        silent: bool
    ) -> ControlStructureResult:
        """
        Execute an entire control structure as a single unit.

        Used for while loops, with statements, and for loops that contain
        break/continue. The whole code is passed to the statement processor,
        which handles caching decisions (mutation detection, side-effect
        checks, etc.).
        """
        try:
            code = ast.unparse(node)

            if self.debug:
                cs_type = get_control_structure_type(node)
                logger.debug("[CONTROL] Processing %s as single unit: %s...", cs_type, code[:80])

            metrics = self.statement_processor.process_statement(code, ttl, silent, stream_output=True)

            # After execution, update lineage for mutated variables
            if metrics.get('status') in (CacheStatus.COMPUTED, CacheStatus.RESTORED):
                self._update_lineage_after_execution(node, code)

            # Annotate metrics with control structure body statements
            # so the badge can show individual statements instead of the
            # entire block as one opaque line.
            cs_type = get_control_structure_type(node)
            metrics['control_type'] = cs_type
            body_stmts = self._extract_body_statements(node)
            if body_stmts:
                metrics['body_statements'] = body_stmts

            # Extract error and annotate with line info for clean traceback.
            # For single-unit control structures, the <cash> frame has a line
            # number relative to the unparsed code.  We need to offset it by
            # the node's starting line in the cell so _show_clean_error points
            # to the correct cell line.
            error = metrics.get('error') if metrics.get('status') == CacheStatus.ERROR else None
            if error is not None:
                # Try to extract the actual error line from the <cash> traceback
                cash_lineno = self._extract_cash_frame_lineno(error)
                if cash_lineno is not None:
                    # ast.unparse produces code starting at line 1;
                    # the node in the cell starts at node.lineno.
                    cell_lineno = getattr(node, 'lineno', 1) + cash_lineno - 1
                    with contextlib.suppress(AttributeError, TypeError):
                        error._cash_error_lineno = cell_lineno

            return ControlStructureResult(
                success=metrics.get('status') != CacheStatus.ERROR,
                metrics=[metrics],
                error=error,
                total_iterations=1,
                cached_iterations=1 if metrics.get('status') == CacheStatus.RESTORED else 0,
                computed_iterations=1 if metrics.get('status') == CacheStatus.COMPUTED else 0
            )
        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user code executed as a unit
            logger.error("[CONTROL] Error executing control structure as single unit: %s", e, exc_info=True)
            return ControlStructureResult(
                success=False,
                metrics=[],
                error=e
            )

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cash_frame_lineno(exc: Exception) -> int | None:
        """Extract the line number from the ``<cash>`` frame in the traceback.

        Returns None if no ``<cash>`` frame is found.
        """
        tb = getattr(exc, '__traceback__', None)
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == '<cash>':
                return tb.tb_lineno
            tb = tb.tb_next
        return None

    # ------------------------------------------------------------------
    # Lineage management
    # ------------------------------------------------------------------

    def _update_lineage_after_execution(self, node: ast.AST, code: str) -> None:
        """
        Update lineage for variables that may have been mutated inside the
        control structure body.

        This ensures downstream statements have correct cache keys even when
        variables were modified in-place (e.g., dict accumulation in loops).
        """
        body_nodes = self._get_body_nodes(node)
        if not body_nodes:
            return

        mutated_vars = self._find_potentially_mutated_variables(body_nodes)

        # Exclude loop target variables — they are not mutations
        if isinstance(node, ast.For):
            target_names = set(extract_target_names(node.target))
            mutated_vars -= target_names

        if mutated_vars:
            iterable_lineage = None
            if isinstance(node, ast.For):
                iterable_lineage = self._get_iterable_lineage(node.iter)

            self._update_mutated_variable_lineages(mutated_vars, iterable_lineage, code)

    def _get_body_nodes(self, node: ast.AST) -> list[ast.AST]:
        """Get all body nodes from a control structure."""
        body = []
        if hasattr(node, 'body'):
            body.extend(node.body)
        if hasattr(node, 'orelse'):
            body.extend(node.orelse)
        if hasattr(node, 'handlers'):
            for handler in node.handlers:
                body.extend(handler.body)
        if hasattr(node, 'finalbody'):
            body.extend(node.finalbody)
        return body

    def _get_expression_iterable_lineage(self, iter_node: ast.AST) -> str | None:
        """Compute lineage for a complex iterable expression by analyzing its inputs."""
        from .analysis import CodeAnalyzer
        iter_code = ast.unparse(iter_node)
        try:
            inputs, _ = CodeAnalyzer.analyze_code_block(iter_code)
            lineage_parts = []
            for var_name in sorted(inputs):
                if var_name in self.statement_processor.variable_lineage:
                    lineage_parts.append(self.statement_processor.variable_lineage[var_name])
                elif var_name in self.shell.user_ns:
                    try:
                        lineage_parts.append(
                            self.statement_processor.compute_hash(self.shell.user_ns[var_name])
                        )
                    except (TypeError, ValueError, AttributeError) as exc:
                        logger.debug("[CONTROL] Failed to hash input variable '%s' for iterable lineage: %s", var_name, exc)
            if lineage_parts:
                return hashlib.sha256(':'.join(lineage_parts).encode()).hexdigest()
        except (SyntaxError, ValueError, AttributeError, TypeError) as exc:
            logger.debug("[CONTROL] Failed to analyze iterable code for lineage: %s", exc)
        return None

    def _get_iterable_lineage(self, iter_node: ast.AST) -> str | None:
        """
        Get the lineage hash of the iterable expression.
        """
        if isinstance(iter_node, ast.Name):
            var_name = iter_node.id
            if var_name in self.statement_processor.variable_lineage:
                return self.statement_processor.variable_lineage[var_name]
            if var_name in self.shell.user_ns:
                try:
                    return self.statement_processor.compute_hash(self.shell.user_ns[var_name])
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.debug("[CONTROL] Failed to hash iterable variable '%s': %s", var_name, exc)
        else:
            return self._get_expression_iterable_lineage(iter_node)
        return None

    def _extract_while_stmts(self, node: ast.While) -> list[str]:
        """Extract while/else statements for badge display."""
        stmts = [f"while {ast.unparse(node.test)}:"]
        stmts.extend(f"  {ast.unparse(s)}" for s in node.body)
        if node.orelse:
            stmts.append("else:")
            stmts.extend(f"  {ast.unparse(s)}" for s in node.orelse)
        return stmts

    def _extract_with_stmts(self, node: ast.With) -> list[str]:
        """Extract with-block statements for badge display."""
        items_str = ', '.join(ast.unparse(item) for item in node.items)
        stmts = [f"with {items_str}:"]
        stmts.extend(f"  {ast.unparse(s)}" for s in node.body)
        return stmts

    def _extract_try_stmts(self, node: ast.Try) -> list[str]:
        """Extract try/except/else/finally statements for badge display."""
        stmts = ["try:"]
        stmts.extend(f"  {ast.unparse(s)}" for s in node.body)
        for handler in getattr(node, 'handlers', []):
            if handler.type:
                hdr = f"except {ast.unparse(handler.type)}"
                if handler.name:
                    hdr += f" as {handler.name}"
            else:
                hdr = "except"
            stmts.append(f"{hdr}:")
            stmts.extend(f"  {ast.unparse(s)}" for s in handler.body)
        if node.orelse:
            stmts.append("else:")
            stmts.extend(f"  {ast.unparse(s)}" for s in node.orelse)
        if getattr(node, 'finalbody', None):
            stmts.append("finally:")
            stmts.extend(f"  {ast.unparse(s)}" for s in node.finalbody)
        return stmts

    def _extract_body_statements(self, node: ast.AST) -> list[str]:
        """Extract individual body statements from a control structure for badge display.

        For if/else returns the statements from ALL branches grouped by branch.
        For while/with/try returns the body statements directly.
        The badge renderer can then show each statement as its own row.
        """
        statements: list[str] = []
        if isinstance(node, ast.If):
            self._extract_if_body_statements(node, statements)
        elif isinstance(node, ast.While):
            statements = self._extract_while_stmts(node)
        elif isinstance(node, ast.With):
            statements = self._extract_with_stmts(node)
        elif isinstance(node, ast.Try):
            statements = self._extract_try_stmts(node)
        return statements

    def _extract_if_body_statements(self, node: ast.If, statements: list[str],
                                     is_elif: bool = False) -> None:
        """Recursively extract if/elif/else branch statements."""
        keyword = "elif" if is_elif else "if"
        statements.append(f"{keyword} {ast.unparse(node.test)}:")
        for stmt in node.body:
            statements.append(f"  {ast.unparse(stmt)}")

        if node.orelse:
            # Check if the else is actually an elif (single If node in orelse)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self._extract_if_body_statements(node.orelse[0], statements, is_elif=True)
            else:
                statements.append("else:")
                for stmt in node.orelse:
                    statements.append(f"  {ast.unparse(stmt)}")

    def _find_potentially_mutated_variables(self, body_nodes: list) -> set[str]:
        """
        Find variables that are mutated inside the control structure body.

        Uses ``MutationDetector`` for precise detection of in-place mutations
        (subscript assignment, method calls like ``.append()``, augmented
        assigns, attribute assignments).
        """
        from .cacheability import analyze_statement

        mutated_vars: set = set()
        for body_node in body_nodes:
            if is_control_structure(body_node):
                nested_body = self._get_body_nodes(body_node)
                mutated_vars.update(self._find_potentially_mutated_variables(nested_body))
            else:
                stmt_code = ast.unparse(body_node)
                try:
                    detected = analyze_statement(stmt_code, None).all_mutated_vars
                    mutated_vars.update(detected)
                except (SyntaxError, ValueError, AttributeError, TypeError) as exc:
                    logger.debug("[CONTROL] Failed to detect mutations in: %s: %s", stmt_code[:60], exc)

        # Filter out built-ins
        built_ins = {'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
                     'sum', 'min', 'max', 'sorted', 'reversed', 'list', 'dict', 'set',
                     'str', 'int', 'float', 'bool', 'type', 'isinstance', 'hasattr',
                     'getattr', 'setattr', 'open', 'get_ipython', '__builtins__'}
        return mutated_vars - built_ins

    def _update_mutated_variable_lineages(self, mutated_vars: set[str],
                                           iterable_lineage: str | None,
                                           loop_code: str) -> None:
        """
        Update the lineage of variables that were mutated inside a control structure.

        The new lineage incorporates:
        1. The control structure code itself
        2. The variable's current value hash
        3. The iterable's lineage (for loops)

        This ensures downstream statements get fresh cache keys when the
        control structure produces different results.
        """
        for var_name in mutated_vars:
            if var_name not in self.shell.user_ns:
                continue

            val = self.shell.user_ns[var_name]

            # Skip immutable scalar types
            if isinstance(val, (int, float, complex, str, bytes, bool, type(None), frozenset, tuple)):
                continue

            try:
                loop_code_hash = hashlib.sha256(loop_code.encode()).hexdigest()
                value_hash = self.statement_processor.compute_hash(val)

                lineage_components = [loop_code_hash, value_hash]
                if iterable_lineage:
                    lineage_components.append(iterable_lineage)

                new_lineage = hashlib.sha256(':'.join(lineage_components).encode()).hexdigest()

                self.statement_processor.variable_lineage[var_name] = new_lineage

                if hasattr(self.statement_processor, 'vars_with_mutation_lineage'):
                    self.statement_processor.vars_with_mutation_lineage.add(var_name)

                with contextlib.suppress(AttributeError, TypeError):
                    val._cash_lineage_hash = new_lineage

                if self.debug:
                    logger.debug("[CONTROL] Updated lineage for mutated var '%s': %s...",
                                 var_name, new_lineage[:20])

            except (TypeError, ValueError, AttributeError) as e:
                if self.debug:
                    logger.warning("[CONTROL] Failed to update lineage for '%s': %s", var_name, e)

