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
        from .control_if_handler import IfHandler
        from .control_try_handler import TryHandler
        self._for_handler = ForLoopHandler(shell, statement_processor, debug, dispatcher=self)
        self._if_handler = IfHandler(shell, statement_processor, debug, dispatcher=self)
        self._try_handler = TryHandler(shell, statement_processor, debug, dispatcher=self)

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
            return self._if_handler.process(node, ttl, silent)
        if isinstance(node, ast.Try):
            return self._try_handler.process(node, ttl, silent)
        return self._execute_as_single_unit(node, ttl, silent)

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
                _helpers.update_lineage_after_execution(
                    self.shell, self.statement_processor, node, code, debug=self.debug,
                )

            # Annotate metrics with control structure body statements
            # so the badge can show individual statements instead of the
            # entire block as one opaque line.
            cs_type = get_control_structure_type(node)
            metrics['control_type'] = cs_type
            body_stmts = _helpers.extract_body_statements(node)
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
                cash_lineno = _helpers.extract_cash_frame_lineno(error)
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

