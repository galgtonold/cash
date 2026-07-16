from __future__ import annotations

"""Shared helpers for control-structure strategy handlers.

**Boundary rule:** this module owns the lineage / mutation / badge / error
helpers that all per-strategy handlers (for / if / try) plus the single-unit
fallback share.  It is import-only: handlers and the orchestrator import the
functions here; the helpers never call back into a handler.

Extracted from ``control_structures/processor.py`` so that ``ForLoopHandler``,
``IfHandler``, and ``TryHandler`` can stay focused on strategy-specific
logic without each carrying a copy of the lineage-update plumbing.
"""

import ast
import hashlib
import logging
import sys
from typing import Any

from ..annotations import (
    CacheAnnotation,
    get_statement_annotations,
    parse_annotations_in_range,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation resolution (CAS-135)
# ---------------------------------------------------------------------------
#
# ``@cash:`` directives were computed in ``cell_executor`` for each TOP-LEVEL
# node and then dropped for anything nested: ``ControlStructureProcessor.process``
# never took an annotation, so a ``# @cash:no-cache`` on a statement inside a
# loop body was silently ignored. The parser was never the problem — the
# directive binds to the body statement correctly — so these helpers only need
# to resolve it at the right scope, and the handlers need to pass ``raw_cell``.
#
# The governing rule is: **annotation granularity follows cache granularity.**
#
# * A ``for``/``if`` body is decomposed into per-statement cache entries, so each
#   body statement resolves its OWN annotation, and a directive on one statement
#   must not leak onto its siblings.
# * A ``while``/``with`` (and a ``for`` with break/continue, or one taking the
#   fast-loop path) executes as ONE cache unit, so a directive anywhere inside it
#   scopes to the whole unit — there is no finer entry for it to attach to.
# * A directive on a control structure's HEADER scopes to that structure, and is
#   therefore inherited by every statement within it.


def resolve_header_annotation(
    raw_cell: str | None,
    node: ast.AST,
    inherited: CacheAnnotation | None = None,
) -> CacheAnnotation | None:
    """The annotation attached to a control structure's *header*.

    Scans only the header line and the comment block immediately above it — NOT
    the node's whole line range. That separation is the point: the whole-range
    scan cannot tell ``# @cash:no-cache`` written above ``for`` (scopes to the
    loop) from one written on a single statement inside its body (scopes to that
    statement), and conflating them would disable caching for every sibling.

    Returns the merge with *inherited* so an enclosing structure's directive
    flows down into a nested one.
    """
    if raw_cell is None:
        return inherited
    lineno = getattr(node, 'lineno', None)
    if lineno is None:
        return inherited
    header = parse_annotations_in_range(raw_cell.splitlines(), lineno, lineno)
    if inherited is None:
        return header if header.has_directives() else None
    return inherited.merge(header)


def resolve_statement_annotation(
    raw_cell: str | None,
    node: ast.AST,
    inherited: CacheAnnotation | None = None,
) -> CacheAnnotation | None:
    """The effective annotation for one statement nested in a control structure.

    Its own directives (from the comment block above it and its own lines),
    merged under any inherited from the structures enclosing it.
    """
    if raw_cell is None:
        return inherited
    own = get_statement_annotations(raw_cell, node)
    if inherited is None:
        return own if own.has_directives() else None
    return inherited.merge(own)


def resolve_unit_annotation(
    raw_cell: str | None,
    node: ast.AST,
    inherited: CacheAnnotation | None = None,
) -> CacheAnnotation | None:
    """The effective annotation for a control structure executed as ONE unit.

    Deliberately the node's WHOLE range: the unit is a single cache entry, so a
    directive anywhere inside it applies to the entry. This is the one place the
    coarse whole-range scan is the correct reading rather than a conflation.
    """
    return resolve_statement_annotation(raw_cell, node, inherited)


# ---------------------------------------------------------------------------
# Output flushing
# ---------------------------------------------------------------------------

def flush_metrics_output(metrics: dict[str, Any]) -> None:
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


def tag_control_metrics(
    result: Any, ctx_hash: str, ctx_label: str, all_metrics: list
) -> None:
    """Tag and flush metrics from a nested control-structure result."""
    for m in result.metrics:
        if 'control_context' not in m:
            m['control_context'] = ctx_hash
            m['branch_label'] = ctx_label
        if not m.get('_output_flushed'):
            flush_metrics_output(m)
    all_metrics.extend(result.metrics)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def extract_cash_frame_lineno(exc: Exception) -> int | None:
    """Extract the line number from the ``<cash>`` frame in the traceback.

    Returns None if no ``<cash>`` frame is found.
    """
    tb = getattr(exc, '__traceback__', None)
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == '<cash>':
            return tb.tb_lineno
        tb = tb.tb_next
    return None


# ---------------------------------------------------------------------------
# Lineage management
# ---------------------------------------------------------------------------

def update_lineage_after_execution(
    shell, statement_processor, node: ast.AST, code: str, debug: bool = False
) -> None:
    """
    Update lineage for variables that may have been mutated inside the
    control structure body.

    This ensures downstream statements have correct cache keys even when
    variables were modified in-place (e.g., dict accumulation in loops).
    """
    body_nodes = get_body_nodes(node)
    if not body_nodes:
        return

    mutated_vars = find_potentially_mutated_variables(body_nodes)

    # Exclude loop target variables — they are not mutations
    if isinstance(node, ast.For):
        from .processor import extract_target_names
        target_names = set(extract_target_names(node.target))
        mutated_vars -= target_names

    if mutated_vars:
        iterable_lineage = None
        if isinstance(node, ast.For):
            iterable_lineage = get_iterable_lineage(shell, statement_processor, node.iter)

        update_mutated_variable_lineages(
            shell, statement_processor, mutated_vars, iterable_lineage, code, debug=debug
        )


def get_body_nodes(node: ast.AST) -> list[ast.AST]:
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


def get_expression_iterable_lineage(shell, statement_processor, iter_node: ast.AST) -> str | None:
    """Compute lineage for a complex iterable expression by analyzing its inputs."""
    from ..analysis import CodeAnalyzer
    iter_code = ast.unparse(iter_node)
    try:
        inputs, _ = CodeAnalyzer.analyze_code_block(iter_code)
        lineage_parts = []
        for var_name in sorted(inputs):
            if var_name in statement_processor.variable_lineage:
                lineage_parts.append(statement_processor.variable_lineage[var_name])
            elif var_name in shell.user_ns:
                try:
                    lineage_parts.append(
                        statement_processor.compute_hash(shell.user_ns[var_name])
                    )
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.debug("[CONTROL] Failed to hash input variable '%s' for iterable lineage: %s", var_name, exc)
        if lineage_parts:
            return hashlib.sha256(':'.join(lineage_parts).encode()).hexdigest()
    except (SyntaxError, ValueError, AttributeError, TypeError) as exc:
        logger.debug("[CONTROL] Failed to analyze iterable code for lineage: %s", exc)
    return None


def get_iterable_lineage(shell, statement_processor, iter_node: ast.AST) -> str | None:
    """
    Get the lineage hash of the iterable expression.
    """
    if isinstance(iter_node, ast.Name):
        var_name = iter_node.id
        if var_name in statement_processor.variable_lineage:
            return statement_processor.variable_lineage[var_name]
        if var_name in shell.user_ns:
            try:
                return statement_processor.compute_hash(shell.user_ns[var_name])
            except (TypeError, ValueError, AttributeError) as exc:
                logger.debug("[CONTROL] Failed to hash iterable variable '%s': %s", var_name, exc)
    else:
        return get_expression_iterable_lineage(shell, statement_processor, iter_node)
    return None


def find_potentially_mutated_variables(body_nodes: list) -> set[str]:
    """
    Find variables that are mutated inside the control structure body.

    Uses ``MutationDetector`` for precise detection of in-place mutations
    (subscript assignment, method calls like ``.append()``, augmented
    assigns, attribute assignments).
    """
    from .processor import is_control_structure
    from ..cacheability import analyze_statement, selfref_reassignment_targets

    mutated_vars: set = set()
    for body_node in body_nodes:
        if is_control_structure(body_node):
            nested_body = get_body_nodes(body_node)
            mutated_vars.update(find_potentially_mutated_variables(nested_body))
        else:
            stmt_code = ast.unparse(body_node)
            try:
                detected = analyze_statement(stmt_code, None).all_mutated_vars
                mutated_vars.update(detected)
            except (SyntaxError, ValueError, AttributeError, TypeError) as exc:
                logger.debug("[CONTROL] Failed to detect mutations in: %s: %s", stmt_code[:60], exc)
            # Self-referential reassignment accumulators (``total = total + b``,
            # ``total += b``) leave no in-place-mutation trace, so all_mutated_vars
            # misses them and the loop is wrongly re-executed on every downstream
            # read, re-draining one-shot iterables. Trust them like append.
            # Kept byte-identical with the simulation collector
            # (VirtualLineage._find_loop_mutated_vars) per the unified-key rule.
            # [CAS-120]
            mutated_vars.update(selfref_reassignment_targets(body_node))

    # Filter out built-ins
    built_ins = {'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
                 'sum', 'min', 'max', 'sorted', 'reversed', 'list', 'dict', 'set',
                 'str', 'int', 'float', 'bool', 'type', 'isinstance', 'hasattr',
                 'getattr', 'setattr', 'open', 'get_ipython', '__builtins__'}
    return mutated_vars - built_ins


def update_mutated_variable_lineages(
    shell, statement_processor, mutated_vars: set[str],
    iterable_lineage: str | None, loop_code: str, debug: bool = False,
) -> None:
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
        if var_name not in shell.user_ns:
            continue

        val = shell.user_ns[var_name]

        # Skip immutable scalar types
        if isinstance(val, (int, float, complex, str, bytes, bool, type(None), frozenset, tuple)):
            continue

        try:
            loop_code_hash = hashlib.sha256(loop_code.encode()).hexdigest()
            value_hash = statement_processor.compute_hash(val)

            lineage_components = [loop_code_hash, value_hash]
            if iterable_lineage:
                lineage_components.append(iterable_lineage)

            new_lineage = hashlib.sha256(':'.join(lineage_components).encode()).hexdigest()

            statement_processor.lineage.record(var_name, new_lineage, value=val)

            if hasattr(statement_processor, 'vars_with_mutation_lineage'):
                statement_processor.vars_with_mutation_lineage.add(var_name)

            if debug:
                logger.debug("[CONTROL] Updated lineage for mutated var '%s': %s...",
                             var_name, new_lineage[:20])

        except (TypeError, ValueError, AttributeError) as e:
            if debug:
                logger.warning("[CONTROL] Failed to update lineage for '%s': %s", var_name, e)


# ---------------------------------------------------------------------------
# Badge / body-statements extraction
# ---------------------------------------------------------------------------

def extract_while_stmts(node: ast.While) -> list[str]:
    """Extract while/else statements for badge display."""
    stmts = [f"while {ast.unparse(node.test)}:"]
    stmts.extend(f"  {ast.unparse(s)}" for s in node.body)
    if node.orelse:
        stmts.append("else:")
        stmts.extend(f"  {ast.unparse(s)}" for s in node.orelse)
    return stmts


def extract_with_stmts(node: ast.With) -> list[str]:
    """Extract with-block statements for badge display."""
    items_str = ', '.join(ast.unparse(item) for item in node.items)
    stmts = [f"with {items_str}:"]
    stmts.extend(f"  {ast.unparse(s)}" for s in node.body)
    return stmts


def extract_try_stmts(node: ast.Try) -> list[str]:
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


def extract_body_statements(node: ast.AST) -> list[str]:
    """Extract individual body statements from a control structure for badge display.

    For if/else returns the statements from ALL branches grouped by branch.
    For while/with/try returns the body statements directly.
    The badge renderer can then show each statement as its own row.
    """
    statements: list[str] = []
    if isinstance(node, ast.If):
        extract_if_body_statements(node, statements)
    elif isinstance(node, ast.While):
        statements = extract_while_stmts(node)
    elif isinstance(node, ast.With):
        statements = extract_with_stmts(node)
    elif isinstance(node, ast.Try):
        statements = extract_try_stmts(node)
    return statements


def extract_if_body_statements(node: ast.If, statements: list[str],
                               is_elif: bool = False) -> None:
    """Recursively extract if/elif/else branch statements."""
    keyword = "elif" if is_elif else "if"
    statements.append(f"{keyword} {ast.unparse(node.test)}:")
    for stmt in node.body:
        statements.append(f"  {ast.unparse(stmt)}")

    if node.orelse:
        # Check if the else is actually an elif (single If node in orelse)
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            extract_if_body_statements(node.orelse[0], statements, is_elif=True)
        else:
            statements.append("else:")
            for stmt in node.orelse:
                statements.append(f"  {ast.unparse(stmt)}")
