from __future__ import annotations

"""If/elif/else per-statement processing strategy.

**Boundary rule:** ``IfHandler`` owns the per-statement decomposition of
``if`` chains — evaluating each condition in the user namespace, picking
the taken branch, and dispatching each body statement through the
statement processor with a ``# control_context:`` cache-key discriminator
so the badge groups them by branch.

It does NOT own:
- Shared lineage / mutation / badge helpers (in
  :mod:`control_structure_helpers`).
- Dispatch of nested control structures (delegated back through the
  orchestrator passed at construction time).
"""

import ast
import contextlib
import hashlib
import logging
from typing import TYPE_CHECKING

from cash.notebook import control_structure_helpers as _helpers
from cash.notebook.cache_status import CacheStatus

if TYPE_CHECKING:
    from .statement import ProcessResult

logger = logging.getLogger(__name__)


class IfHandler:
    """Per-statement caching for ``if`` / ``elif`` / ``else`` chains.

    See module docstring for the boundary; tests can construct this
    handler with mock dependencies and exercise it directly.
    """

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
        node: ast.If,
        ttl: int | None,
        silent: bool,
    ):
        """Execute an if/elif/else by processing each branch statement individually.

        1. Walk the if/elif/else chain evaluating conditions until one is True
           (or we fall through to else).
        2. Execute each statement in the taken branch individually through the
           statement processor, tagging metrics with ``control_context``
           so the badge can group them.
        3. Only the executed branch is shown in the badge.

        Falls back to ``_execute_as_single_unit`` if condition evaluation fails.
        """
        from .control_structures import ControlStructureResult

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
            _helpers.update_lineage_after_execution(
                self.shell, self.statement_processor, node, ast.unparse(node), debug=self.debug
            )

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

    # ------------------------------------------------------------------
    # Per-branch-statement processing
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
        from .control_structures import is_control_structure

        if is_control_structure(body_node):
            result = self.dispatcher.process(body_node, ttl, silent, None)
            _helpers.tag_control_metrics(result, branch_hash, branch_label, all_metrics)
            if not result.success:
                err = result.error or RuntimeError("Error in nested control structure")
                if not hasattr(err, '_cash_error_lineno'):
                    with contextlib.suppress(AttributeError, TypeError):
                        err._cash_error_lineno = getattr(body_node, 'lineno', None)
                raise err
            return result.computed_iterations > 0
        stmt_code = ast.unparse(body_node)
        modified_code = f"# control_context: {branch_hash}\n{stmt_code}"
        metrics = self.statement_processor.process_statement(modified_code, ttl, silent)
        metrics['control_context'] = branch_hash
        metrics['branch_label'] = branch_label
        _helpers.flush_metrics_output(metrics)
        all_metrics.append(metrics)
        if metrics.get('status') == CacheStatus.ERROR:
            err = metrics.get('error', RuntimeError(f"Error executing: {stmt_code}"))
            with contextlib.suppress(AttributeError, TypeError):
                err._cash_error_lineno = getattr(body_node, 'lineno', None)
            raise err
        return metrics.get('status') == CacheStatus.COMPUTED

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
