"""Shared helpers for control-structure strategy handlers.

**Boundary rule:** this module owns the lineage / mutation / badge / error
helpers that all per-strategy handlers (for / if / try) plus the single-unit
fallback share.  It is import-only: handlers and the orchestrator import the
functions here; the helpers never call back into a handler.

Extracted from ``control_structures/processor.py`` so that ``ForLoopHandler``,
``IfHandler``, and ``TryHandler`` can stay focused on strategy-specific
logic without each carrying a copy of the lineage-update plumbing.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import logging
import sys
from collections.abc import Callable
from typing import Any

from ...analysis.annotations import (
    CacheAnnotation,
    get_statement_annotations,
    parse_annotations_in_range,
)
from ...analysis.cacheability import analyze_statement
from ...analysis.code_analyzer import CodeAnalyzer
from ...analysis.mutations import selfref_reassignment_targets
from ..cache_status import CacheStatus
from ..compiled_source import is_cash_filename
from .common import extract_target_names, is_control_structure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation resolution
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
    lineno = getattr(node, "lineno", None)
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
    if metrics.get("stdout"):
        print(metrics["stdout"], end="", flush=True)
    if metrics.get("stderr"):
        print(metrics["stderr"], end="", file=sys.stderr, flush=True)
    metrics["_output_flushed"] = True


# ---------------------------------------------------------------------------
# Running a body statement
# ---------------------------------------------------------------------------


def run_marked_statement(
    statement_processor: Any,
    body_node: ast.AST,
    mark: Callable[[str], str],
    ttl: int | None,
    silent: bool,
    raw_cell: str | None,
    inherited_annotation: CacheAnnotation | None,
    all_metrics: list,
    tags: dict[str, Any],
) -> dict[str, Any]:
    """Run one body statement of a control structure as its own cache entry.

    *mark* adds the statement's cache-key discriminator (its iteration or
    branch) to the code; *tags* go on its metrics for the badge. Its own
    ``@cash:`` directive is resolved under *inherited_annotation*: body
    statements are separate entries, so one statement's directive must not
    leak onto its siblings. Output is flushed as the statement finishes. A
    statement that failed raises its error, carrying the body line.
    """
    code = ast.unparse(body_node)
    annotation = resolve_statement_annotation(raw_cell, body_node, inherited_annotation)
    # A body statement is never the cell's last expression: Jupyter shows
    # nothing for ``ax.text(...)`` inside a loop.
    metrics = statement_processor.process_statement(mark(code), ttl, silent, annotation=annotation, is_last=False)
    metrics.update(tags)
    flush_metrics_output(metrics)
    all_metrics.append(metrics)
    if metrics.get("status") == CacheStatus.ERROR:
        raise at_line(metrics.get("error", RuntimeError(f"Error executing: {code}")), body_node, overwrite=True)
    return metrics


def run_nested_structure(
    dispatcher: Any,
    body_node: ast.AST,
    ttl: int | None,
    silent: bool,
    parent_context: dict[str, Any] | None,
    raw_cell: str | None,
    inherited_annotation: CacheAnnotation | None,
    all_metrics: list,
    tag: Callable[[dict[str, Any]], None],
) -> Any:
    """Run a control structure nested in a body through the orchestrator.

    *tag* stamps each of its metrics for the enclosing structure's badge
    group; output not yet flushed is flushed. A failure raises the nested
    error, keeping the line it was raised at if it has one.
    """
    result = dispatcher.process(body_node, ttl, silent, parent_context, raw_cell, inherited_annotation)
    for m in result.metrics:
        tag(m)
        if not m.get("_output_flushed"):
            flush_metrics_output(m)
    all_metrics.extend(result.metrics)
    if not result.success:
        raise at_line(result.error or RuntimeError("Error in nested control structure"), body_node, overwrite=False)
    return result


def branch_tag(ctx_hash: str, ctx_label: str) -> Callable[[dict[str, Any]], None]:
    """A *tag* for :func:`run_nested_structure`: the enclosing branch, unless
    a branch nested deeper already claimed the metric."""

    def tag(m: dict[str, Any]) -> None:
        if "control_context" not in m:
            m["control_context"] = ctx_hash
            m["branch_label"] = ctx_label

    return tag


def counts_as_cached(metrics: dict[str, Any]) -> bool:
    return metrics.get("status") in (CacheStatus.RESTORED, CacheStatus.SKIPPED)


def at_line(err: BaseException, node: ast.AST, *, overwrite: bool) -> BaseException:
    """*err* marked with *node*'s cell line, for the clean traceback."""
    if overwrite or not hasattr(err, "_cash_error_lineno"):
        with contextlib.suppress(AttributeError, TypeError):
            err._cash_error_lineno = getattr(node, "lineno", None)  # type: ignore[attr-defined]
    return err


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def extract_cash_frame_lineno(exc: Exception) -> int | None:
    """Extract the line number from the cash-compiled frame in the traceback.

    Returns None if no cash frame is found. Matches on the ``<cash-`` prefix:
    each statement compiles under its own ``<cash-{digest}>`` name so its
    source resolves in linecache.
    """
    tb = getattr(exc, "__traceback__", None)
    while tb is not None:
        if is_cash_filename(tb.tb_frame.f_code.co_filename):
            return tb.tb_lineno
        tb = tb.tb_next
    return None


# ---------------------------------------------------------------------------
# Lineage management
# ---------------------------------------------------------------------------


def update_lineage_after_execution(
    shell,
    statement_processor,
    node: ast.AST,
    code: str,
    body_files: set[str] | None = None,
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
        target_names = set(extract_target_names(node.target))
        mutated_vars -= target_names

    if mutated_vars:
        inherit_body_file_deps(shell, statement_processor, body_nodes, mutated_vars, body_files)
        iterable_lineage = None
        target_names: set[str] = set()
        if isinstance(node, ast.For):
            iterable_lineage = get_iterable_lineage(shell, statement_processor, node.iter)

            target_names = set(extract_target_names(node.target))

        update_mutated_variable_lineages(
            shell,
            statement_processor,
            mutated_vars,
            iterable_lineage,
            code,
            input_lineages=collect_body_input_lineages(
                statement_processor,
                body_nodes,
                mutated_vars | target_names,
            ),
        )


def inherit_body_file_deps(
    shell, statement_processor, body_nodes: list, mutated_vars: set[str], body_files: set[str] | None = None
) -> None:
    """Give each variable the loop mutated the files its body read.

    ``for f in files: d = pd.read_csv(f); parts.append(d)`` recorded each file
    against ``d`` and none against ``parts``: ``parts.append(d)`` is a method
    call, and a loop body's calls are not classified. So ``raw =
    pd.concat(parts)`` inherited no file, the next statement's key had no file
    component, and after an existing file was rewritten ``sales`` was restored
    from the old content -- visible once the list passed 200 frames and its
    sampled hash stopped covering the middle. A
    statement that reads files already hands them to its outputs
    (``FileDepsTracker.inherit_from_inputs``); this is the same rule for the
    loop's accumulators.

    *body_files* is what a per-iteration loop gathered as it ran: a name the
    body rebinds holds only the last iteration's files by now.
    """
    executed_file_deps = statement_processor.tracking_state.executed_file_deps
    if executed_file_deps is None:
        return
    used = {sub.id for body_node in body_nodes for sub in ast.walk(body_node) if isinstance(sub, ast.Name)}
    files: set[str] = set(body_files or ())
    for name in used:
        files.update(executed_file_deps.get(name, ()))
    if not files:
        return
    for var_name in mutated_vars:
        value = shell.user_ns.get(var_name)
        if value is None or isinstance(value, (int, float, complex, str, bytes, bool)):
            continue
        executed_file_deps.setdefault(var_name, set()).update(files)


def collect_body_input_lineages(
    statement_processor,
    body_nodes: list,
    exclude: set[str],
) -> dict[str, str]:
    """Current lineage of every variable the control-structure body READS.

    The mutated variable's identity has to answer "did this come from the same
    upstream computation as last time?", and the body's inputs are most of that
    answer. Without them the only content signal is a sampled hash, which is
    not enough (see :func:`update_mutated_variable_lineages`).

    Excludes the mutated variables themselves (a loop body almost always reads
    what it mutates, and folding that in would just re-add the sampled hash by
    another route) and the loop targets (they are bindings the loop creates, not
    upstream inputs). Names with no recorded lineage are skipped rather than
    guessed at: absence is not a lineage, and inventing one would churn the key
    on every run.
    """
    reads: set[str] = set()
    for body_node in body_nodes:
        for sub in ast.walk(body_node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                reads.add(sub.id)
    lineages: dict[str, str] = {}
    for name in reads - exclude:
        lin = statement_processor.tracking_state.variable_lineage.get(name)
        if lin:
            lineages[name] = lin
    return lineages


def get_body_nodes(node: ast.AST) -> list[ast.AST]:
    """Get all body nodes from a control structure."""
    body = []
    if hasattr(node, "body"):
        body.extend(node.body)
    if hasattr(node, "orelse"):
        body.extend(node.orelse)
    if hasattr(node, "handlers"):
        for handler in node.handlers:
            body.extend(handler.body)
    if hasattr(node, "finalbody"):
        body.extend(node.finalbody)
    return body


def get_expression_iterable_lineage(shell, statement_processor, iter_node: ast.AST) -> str | None:
    """Compute lineage for a complex iterable expression by analyzing its inputs."""

    iter_code = ast.unparse(iter_node)
    try:
        inputs, _ = CodeAnalyzer.analyze_code_block(iter_code)
        lineage_parts = []
        for var_name in sorted(inputs):
            if var_name in statement_processor.tracking_state.variable_lineage:
                lineage_parts.append(statement_processor.tracking_state.variable_lineage[var_name])
            elif var_name in shell.user_ns:
                try:
                    lineage_parts.append(statement_processor.compute_hash(shell.user_ns[var_name]))
                except (TypeError, ValueError, AttributeError) as exc:
                    logger.debug("[CONTROL] Failed to hash input variable '%s' for iterable lineage: %s", var_name, exc)
        if lineage_parts:
            return hashlib.sha256(":".join(lineage_parts).encode()).hexdigest()
    except (SyntaxError, ValueError, AttributeError, TypeError) as exc:
        logger.debug("[CONTROL] Failed to analyze iterable code for lineage: %s", exc)
    return None


def get_iterable_lineage(shell, statement_processor, iter_node: ast.AST) -> str | None:
    """
    Get the lineage hash of the iterable expression.
    """
    if isinstance(iter_node, ast.Name):
        var_name = iter_node.id
        if var_name in statement_processor.tracking_state.variable_lineage:
            return statement_processor.tracking_state.variable_lineage[var_name]
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
            #
            mutated_vars.update(selfref_reassignment_targets(body_node))

    # Filter out built-ins
    built_ins = {
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sum",
        "min",
        "max",
        "sorted",
        "reversed",
        "list",
        "dict",
        "set",
        "str",
        "int",
        "float",
        "bool",
        "type",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "open",
        "get_ipython",
        "__builtins__",
    }
    return mutated_vars - built_ins


def update_mutated_variable_lineages(
    shell,
    statement_processor,
    mutated_vars: set[str],
    iterable_lineage: str | None,
    loop_code: str,
    input_lineages: dict[str, str] | None = None,
) -> None:
    """Give every variable the control structure mutated a new lineage.

    The lineage hashes, in order: the structure's code, the value's hash,
    the variable's lineage before the structure ran (``prev=``), the
    iterable's lineage (loops), and the lineage of every OTHER variable the
    body read (*input_lineages*).

    The value hash alone cannot carry this: ``compute_hash`` SAMPLES large
    objects, so two frames that differ past the sampled region hash equal.
    The lineage must come from provenance -- what went in -- or the next
    statement to read the variable restores a stale entry under an unchanged
    key. The ``prev=`` component is the only one left when the loop source
    matches and the sampled hash collides: a loop over a frame built two
    different ways upstream must leave with two different lineages.

    Re-running an unchanged mutation does not churn: the statement restore
    puts the receiver's pre-loop lineage back before the loop mints the next.
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

            # `prev=`: what this variable was before the loop touched it.
            # The only component left that discriminates when the loop's source
            # matches and the sampled value hash collides -- see the docstring.
            prior_lineage = statement_processor.tracking_state.variable_lineage.get(var_name)
            lineage_components = [loop_code_hash, value_hash]
            if prior_lineage:
                lineage_components.append(f"prev={prior_lineage}")
            if iterable_lineage:
                lineage_components.append(iterable_lineage)
            for name, lin in sorted((input_lineages or {}).items()):
                lineage_components.append(f"{name}={lin}")

            new_lineage = hashlib.sha256(":".join(lineage_components).encode()).hexdigest()

            statement_processor.tracking_state.lineage.record(var_name, new_lineage, value=val)

            statement_processor.tracking_state.vars_with_mutation_lineage.add(var_name)

            logger.debug("[CONTROL] Updated lineage for mutated var '%s': %s...", var_name, new_lineage[:20])

        except (TypeError, ValueError, AttributeError) as e:
            logger.debug("[CONTROL] Failed to update lineage for '%s': %s", var_name, e)


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
    items_str = ", ".join(ast.unparse(item) for item in node.items)
    stmts = [f"with {items_str}:"]
    stmts.extend(f"  {ast.unparse(s)}" for s in node.body)
    return stmts


def extract_try_stmts(node: ast.Try) -> list[str]:
    """Extract try/except/else/finally statements for badge display."""
    stmts = ["try:"]
    stmts.extend(f"  {ast.unparse(s)}" for s in node.body)
    for handler in getattr(node, "handlers", []):
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
    if getattr(node, "finalbody", None):
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


def extract_if_body_statements(node: ast.If, statements: list[str], is_elif: bool = False) -> None:
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
