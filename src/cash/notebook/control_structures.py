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
from cash.notebook import control_structure_helpers as _helpers

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
        # Per-strategy handlers — constructed once.  Each owns the
        # strategy-specific logic; the orchestrator stays thin.
        from .control_for_handler import ForLoopHandler
        self._for_handler = ForLoopHandler(shell, statement_processor, debug, dispatcher=self)

    @staticmethod
    def _flush_metrics_output(metrics: dict[str, Any]) -> None:
        """Thin wrapper around :func:`control_structure_helpers.flush_metrics_output`."""
        _helpers.flush_metrics_output(metrics)

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
            return self._for_handler.process(node, ttl, silent, parent_context)
        if isinstance(node, ast.If):
            return self._execute_if_per_statement(node, ttl, silent)
        if isinstance(node, ast.Try):
            return self._execute_try_per_statement(node, ttl, silent)
        return self._execute_as_single_unit(node, ttl, silent)

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
        """Thin wrapper around :func:`control_structure_helpers.tag_control_metrics`."""
        _helpers.tag_control_metrics(result, ctx_hash, ctx_label, all_metrics)

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
            self.statement_processor.lineage.record(
                matched_handler.name, exc_lineage, value=caught_exception,
            )
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
        """Thin wrapper around :func:`control_structure_helpers.extract_cash_frame_lineno`."""
        return _helpers.extract_cash_frame_lineno(exc)

    # ------------------------------------------------------------------
    # Lineage management
    # ------------------------------------------------------------------

    def _update_lineage_after_execution(self, node: ast.AST, code: str) -> None:
        """Thin wrapper around :func:`control_structure_helpers.update_lineage_after_execution`."""
        _helpers.update_lineage_after_execution(
            self.shell, self.statement_processor, node, code, debug=self.debug
        )

    def _get_body_nodes(self, node: ast.AST) -> list[ast.AST]:
        """Thin wrapper around :func:`control_structure_helpers.get_body_nodes`."""
        return _helpers.get_body_nodes(node)

    def _get_expression_iterable_lineage(self, iter_node: ast.AST) -> str | None:
        """Thin wrapper around :func:`control_structure_helpers.get_expression_iterable_lineage`."""
        return _helpers.get_expression_iterable_lineage(self.shell, self.statement_processor, iter_node)

    def _get_iterable_lineage(self, iter_node: ast.AST) -> str | None:
        """Thin wrapper around :func:`control_structure_helpers.get_iterable_lineage`."""
        return _helpers.get_iterable_lineage(self.shell, self.statement_processor, iter_node)

    def _extract_while_stmts(self, node: ast.While) -> list[str]:
        """Thin wrapper around :func:`control_structure_helpers.extract_while_stmts`."""
        return _helpers.extract_while_stmts(node)

    def _extract_with_stmts(self, node: ast.With) -> list[str]:
        """Thin wrapper around :func:`control_structure_helpers.extract_with_stmts`."""
        return _helpers.extract_with_stmts(node)

    def _extract_try_stmts(self, node: ast.Try) -> list[str]:
        """Thin wrapper around :func:`control_structure_helpers.extract_try_stmts`."""
        return _helpers.extract_try_stmts(node)

    def _extract_body_statements(self, node: ast.AST) -> list[str]:
        """Thin wrapper around :func:`control_structure_helpers.extract_body_statements`."""
        return _helpers.extract_body_statements(node)

    def _extract_if_body_statements(self, node: ast.If, statements: list[str],
                                     is_elif: bool = False) -> None:
        """Thin wrapper around :func:`control_structure_helpers.extract_if_body_statements`."""
        _helpers.extract_if_body_statements(node, statements, is_elif)

    def _find_potentially_mutated_variables(self, body_nodes: list) -> set[str]:
        """Thin wrapper around :func:`control_structure_helpers.find_potentially_mutated_variables`."""
        return _helpers.find_potentially_mutated_variables(body_nodes)

    def _update_mutated_variable_lineages(self, mutated_vars: set[str],
                                           iterable_lineage: str | None,
                                           loop_code: str) -> None:
        """Thin wrapper around :func:`control_structure_helpers.update_mutated_variable_lineages`."""
        _helpers.update_mutated_variable_lineages(
            self.shell, self.statement_processor, mutated_vars, iterable_lineage, loop_code, debug=self.debug
        )

