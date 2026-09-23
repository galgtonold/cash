"""Try/except/else/finally per-statement processing strategy.

**Boundary rule:** ``TryHandler`` owns the per-statement decomposition of
``try`` blocks — executing the try body until either it succeeds or an
exception is caught, routing to the first matching ``except`` handler,
running ``else`` only on clean exit, and always running ``finally``.

Each executed sub-statement gets its own cache key, storage info, and
timing — and ``print()`` calls are never suppressed by the SKIPPED
optimisation.

It does NOT own:
- Shared lineage / mutation / badge helpers (in
  :mod:`control_structures.helpers`).
- Dispatch of nested control structures (delegated back through the
  orchestrator passed at construction time).
"""

from __future__ import annotations

import ast
import hashlib
import logging
from typing import TYPE_CHECKING

from cash.control_markers import mark_control

from ..cache_status import CacheStatus
from . import helpers as _helpers
from .common import ControlStructureResult, is_control_structure

if TYPE_CHECKING:
    from ..statement import ProcessResult

logger = logging.getLogger(__name__)


class TryHandler:
    """Per-statement caching for ``try`` / ``except`` / ``else`` / ``finally``.

    See module docstring for the boundary; tests can construct this
    handler with mock dependencies and exercise it directly.
    """

    def __init__(self, shell, statement_processor, dispatcher):
        self.shell = shell
        self.statement_processor = statement_processor
        self.dispatcher = dispatcher

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        node: ast.Try,
        ttl: int | None,
        silent: bool,
        raw_cell: str | None = None,
        inherited_annotation=None,
    ):
        """Execute a try/except/else/finally by processing each statement individually.

        1. Execute try body statements one by one.
        2. If a statement raises an exception, find the matching except handler
           and execute its body statements.
        3. If no exception, execute the else body (if present).
        4. Always execute the finally body (if present).
        5. Build ``body_stmts`` showing only the actually-executed parts.

        This mirrors the if per-statement path so that each sub-statement
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
            # A directive on the ``try`` header scopes to the whole construct and
            # flows down into every branch within it.
            try_annotation = _helpers.resolve_header_annotation(
                raw_cell,
                node,
                inherited_annotation,
            )
            try_body_succeeded, caught_exception, c, d = self._execute_try_body_stmts(
                node,
                branch_hash,
                branch_label,
                ttl,
                silent,
                all_metrics,
                raw_cell,
                try_annotation,
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
                        matched_handler.body,
                        handler_hash,
                        handler_label,
                        ttl,
                        silent,
                        all_metrics,
                        raw_cell,
                        try_annotation,
                    )
                    cached_count += c
                    computed_count += d
                    caught_exception = None
                else:
                    raise caught_exception

            if try_body_succeeded and node.orelse:
                else_label = "else"
                else_hash = hashlib.sha256(else_label.encode()).hexdigest()[:16]
                c, d = self._execute_simple_branch(
                    node.orelse,
                    else_hash,
                    else_label,
                    ttl,
                    silent,
                    all_metrics,
                    raw_cell,
                    try_annotation,
                )
                cached_count += c
                computed_count += d

            if getattr(node, "finalbody", None):
                finally_label = "finally"
                finally_hash = hashlib.sha256(finally_label.encode()).hexdigest()[:16]
                c, d = self._execute_simple_branch(
                    node.finalbody,
                    finally_hash,
                    finally_label,
                    ttl,
                    silent,
                    all_metrics,
                    raw_cell,
                    try_annotation,
                )
                cached_count += c
                computed_count += d

            if caught_exception is not None:
                raise caught_exception

            _helpers.update_lineage_after_execution(self.shell, self.statement_processor, node, ast.unparse(node))
            body_stmts = self._build_try_executed_body_stmts(node, try_body_succeeded, matched_handler)

            for m in all_metrics:
                m["control_type"] = "try"
                if "body_statements" not in m:
                    m["body_statements"] = body_stmts

            return ControlStructureResult(
                success=True,
                metrics=all_metrics,
                total_iterations=1,
                cached_iterations=1 if computed_count == 0 and all_metrics else 0,
                computed_iterations=1 if computed_count > 0 else 0,
            )

        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user try/except body code
            # Handed back to the cell, which raises it; logged above debug it
            # would print the traceback a second time.
            logger.debug("[CONTROL] Error in try per-statement execution: %s", e, exc_info=True)
            return ControlStructureResult(
                success=False,
                metrics=all_metrics,
                error=e,
            )

    # ------------------------------------------------------------------
    # Branch execution
    # ------------------------------------------------------------------

    def _execute_simple_branch(
        self,
        body_nodes: list,
        ctx_hash: str,
        ctx_label: str,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        raw_cell: str | None = None,
        branch_annotation=None,
    ) -> tuple[int, int]:
        """Execute body nodes under a context hash; return (cached_count, computed_count).

        Used for else and finally branches of try/except where no per-statement
        lineno tagging is needed.
        """
        cached = computed = 0
        for body_node in body_nodes:
            if self._run_body_node(
                body_node, ctx_hash, ctx_label, ttl, silent, all_metrics, raw_cell, branch_annotation
            ):
                computed += 1
            elif is_control_structure(body_node) or _helpers.counts_as_cached(all_metrics[-1]):
                cached += 1
        return cached, computed

    def _run_body_node(
        self,
        body_node: ast.AST,
        ctx_hash: str,
        ctx_label: str,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        raw_cell: str | None,
        branch_annotation,
    ) -> bool:
        """Run one statement of a branch; True if it (or a nested structure) computed."""
        if is_control_structure(body_node):
            result = _helpers.run_nested_structure(
                self.dispatcher,
                body_node,
                ttl,
                silent,
                None,
                raw_cell,
                branch_annotation,
                all_metrics,
                _helpers.branch_tag(ctx_hash, ctx_label),
            )
            return result.computed_iterations > 0
        metrics = _helpers.run_marked_statement(
            self.statement_processor,
            body_node,
            lambda code: mark_control(code, ctx_hash),
            ttl,
            silent,
            raw_cell,
            branch_annotation,
            all_metrics,
            {"control_context": ctx_hash, "branch_label": ctx_label},
        )
        return metrics.get("status") == CacheStatus.COMPUTED

    def _execute_try_body_stmts(
        self,
        node: ast.Try,
        branch_hash: str,
        branch_label: str,
        ttl: int | None,
        silent: bool,
        all_metrics: list,
        raw_cell: str | None = None,
        branch_annotation=None,
    ) -> tuple[bool, Exception | None, int, int]:
        """Execute the try-body statements; return (succeeded, caught_exc, cached, computed)."""
        cached = computed = 0
        for body_node in node.body:
            try:
                ran = self._run_body_node(
                    body_node, branch_hash, branch_label, ttl, silent, all_metrics, raw_cell, branch_annotation
                )
            except Exception as e:  # noqa: BLE001 - the try's own except clauses decide what the user's error means
                return False, e, cached, computed
            if ran:
                computed += 1
            elif is_control_structure(body_node) or _helpers.counts_as_cached(all_metrics[-1]):
                cached += 1
        return True, None, cached, computed

    # ------------------------------------------------------------------
    # Handler matching
    # ------------------------------------------------------------------

    def _bind_exception_to_handler(self, matched_handler: ast.ExceptHandler, caught_exception: Exception) -> None:
        """Bind the caught exception to the handler's variable and set its lineage."""
        if not matched_handler.name:
            return
        self.shell.user_ns[matched_handler.name] = caught_exception
        try:
            exc_class_name = type(caught_exception).__name__
            class_lineage = self.statement_processor.variable_lineage.get(exc_class_name, "")
            exc_lineage = hashlib.sha256(
                f"__exception__:{exc_class_name}:{class_lineage}:{caught_exception!s}:{caught_exception!r}".encode()
            ).hexdigest()
            self.statement_processor.lineage.record(
                matched_handler.name,
                exc_lineage,
                value=caught_exception,
            )
        except (ValueError, AttributeError, TypeError) as exc:
            logger.debug("[CONTROL] Failed to compute exception lineage for handler variable: %s", exc)

    def _find_matching_handler(self, handlers: list[ast.ExceptHandler], exc: Exception) -> ast.ExceptHandler | None:
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
        if getattr(node, "finalbody", None):
            statements.append("finally:")
            for stmt in node.finalbody:
                statements.append(f"  {ast.unparse(stmt)}")

        return statements
