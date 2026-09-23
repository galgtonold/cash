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

        The branches run inside a real ``try``/``finally`` in the same shape
        as the user's, so control flow is Python's:

        1. The try body runs statement by statement until one raises.
        2. An exception (any ``BaseException``) goes to the first matching
           ``except``; with none matching it propagates.
        3. The ``else`` body runs only when the try body finished, and its
           errors are not caught by the handlers.
        4. The ``finally`` body always runs, including when no handler
           matched or a handler or the ``else`` body raised.
        5. ``except ... as name`` unbinds ``name`` when the handler ends.

        This mirrors the if per-statement path so that each sub-statement
        gets its own cache key, storage info, and timing — and ``print()``
        calls are never suppressed by the SKIPPED optimisation.

        An ``Exception`` that leaves the construct is returned as the result's
        ``error``; any other ``BaseException`` propagates.
        """
        all_metrics: list[ProcessResult] = []
        try_body_succeeded = False
        matched_handler: ast.ExceptHandler | None = None
        computed_count = 0

        try:
            # A directive on the ``try`` header scopes to the whole construct and
            # flows down into every branch within it.
            try_annotation = _helpers.resolve_header_annotation(
                raw_cell,
                node,
                inherited_annotation,
            )

            def run_branch(body: list, label: str) -> None:
                nonlocal computed_count
                label_hash = hashlib.sha256(label.encode()).hexdigest()[:16]
                _cached, computed = self._execute_simple_branch(
                    body, label_hash, label, ttl, silent, all_metrics, raw_cell, try_annotation
                )
                computed_count += computed

            try:
                try:
                    run_branch(node.body, "try")
                except BaseException as caught:
                    handler = self._find_matching_handler(node.handlers, caught)
                    if handler is None:
                        raise
                    matched_handler = handler
                    self._bind_exception_to_handler(handler, caught)
                    try:
                        run_branch(handler.body, self._format_handler_label(handler))
                    finally:
                        self._unbind_handler_name(handler)
                else:
                    try_body_succeeded = True
                    if node.orelse:
                        run_branch(node.orelse, "else")
            finally:
                if node.finalbody:
                    run_branch(node.finalbody, "finally")

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

        except Exception as e:  # noqa: BLE001 - the user's error, after their own handlers and finally ran
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

        A statement that fails raises its error, marked with its cell line.
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

    # ------------------------------------------------------------------
    # Handler matching
    # ------------------------------------------------------------------

    def _unbind_handler_name(self, handler: ast.ExceptHandler) -> None:
        """Delete ``except ... as name`` once the handler ends, as Python does."""
        if not handler.name:
            return
        self.shell.user_ns.pop(handler.name, None)
        self.statement_processor.tracking_state.lineage.discard(handler.name)

    def _bind_exception_to_handler(self, matched_handler: ast.ExceptHandler, caught_exception: BaseException) -> None:
        """Bind the caught exception to the handler's variable and set its lineage."""
        if not matched_handler.name:
            return
        self.shell.user_ns[matched_handler.name] = caught_exception
        try:
            exc_class_name = type(caught_exception).__name__
            class_lineage = self.statement_processor.tracking_state.variable_lineage.get(exc_class_name, "")
            exc_lineage = hashlib.sha256(
                f"__exception__:{exc_class_name}:{class_lineage}:{caught_exception!s}:{caught_exception!r}".encode()
            ).hexdigest()
            self.statement_processor.tracking_state.lineage.record(
                matched_handler.name,
                exc_lineage,
                value=caught_exception,
            )
        except (ValueError, AttributeError, TypeError) as exc:
            logger.debug("[CONTROL] Failed to compute exception lineage for handler variable: %s", exc)

    def _find_matching_handler(self, handlers: list[ast.ExceptHandler], exc: BaseException) -> ast.ExceptHandler | None:
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
