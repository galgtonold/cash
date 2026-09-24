"""Running the statements the upstream check scheduled, and reporting a failure.

:class:`StatementReplay` re-runs them through the statement processor, and
when one fails it says whether the fault is the user's code or a gap in the
plan cash made.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cash.control_markers import strip_markers

from ...analysis.annotations import get_statement_annotations
from ...analysis.ast_util import parse_cached
from ...analysis.code_analyzer import clean_cell_source, parse_cell_source
from ...exceptions import UpstreamStateError
from ..control_structures import is_control_structure
from ..tracking_state import TrackingState

if TYPE_CHECKING:
    from ..statement import ProcessResult

__all__ = ["StatementReplay"]

logger = logging.getLogger(__name__)


class StatementReplay:
    """Re-runs scheduled upstream statements, in order, loudly on failure."""

    def __init__(self, tracking_state: TrackingState) -> None:
        self.tracking_state = tracking_state

    @staticmethod
    def in_notebook_order(metrics: list, notebook_cells: list[str] | None) -> list:
        """*metrics* in the order their statements stand in the notebook.

        The restores came first and the re-runs after them, so the badge's
        Upstream list read ``^CACHED: results[name] = evaluate(...)`` above
        ``^EXECUTED: results = {}`` -- an order nothing ran in.
        The restores carried a simulation-trace position and the
        re-runs none. Both are placed by their statement's place in the
        notebook; a loop's passes carry their loop's (``stmt_code``, stamped
        by ``reexecute``) and keep their order. A metric whose
        statement is not found keeps its place after the ones that are.
        """
        order: dict[str, int] = {}
        for cell in notebook_cells or ():
            tree = parse_cell_source(cell)
            if tree is None:
                continue
            for node in tree.body:
                at = len(order)
                order.setdefault(ast.unparse(node), at)
                # A restored loop pass is keyed by its body statement.
                for sub in ast.walk(node):
                    if sub is not node and isinstance(sub, ast.stmt):
                        order.setdefault(ast.unparse(sub), at)
        end = len(order)

        def place(item):
            i, m = item
            code = m.get("upstream_statement") or m.get("code") or ""
            code = strip_markers(code).strip()
            return (order.get(code, end), i)

        return [m for _, m in sorted(enumerate(metrics), key=place)]

    @staticmethod
    def statement_directives(notebook_cells: list[str] | None) -> dict[str, Any]:
        """``{statement code: its # @cash: directives}`` across the notebook.

        A statement re-run as an upstream repair ran with no annotation: one notebook
        put ``# @cash:no-cache-calls`` on a comprehension, and its calls were
        cached whenever a cell below repaired it. The repair has the
        statement's code, keyed as the simulator keys it; the directive is read
        from its cell as a direct run reads it. Only statements that carry one.
        """

        found: dict[str, Any] = {}
        for cell in notebook_cells or ():
            if "@cash:" not in cell:
                continue
            tree = parse_cell_source(cell)
            if tree is None:
                continue
            clean = clean_cell_source(cell)
            for node in tree.body:
                try:
                    annotation = get_statement_annotations(clean, node)
                    if annotation.has_directives():
                        found.setdefault(ast.unparse(node), annotation)
                except (ValueError, RecursionError):  # a directive lookup never breaks a repair
                    continue
        return found

    def sum_execution_times(self, executed_metrics: list) -> float:
        """Sum ``total_time`` from a list of metric dicts."""
        total = 0.0
        for metrics in executed_metrics:
            if metrics and "total_time" in metrics:
                total += metrics["total_time"]
        return total

    def reexecute(
        self,
        statements: list[str],
        process_callback: Callable[..., ProcessResult],
        global_ttl: int | None,
        progress_callback: Callable[..., None] | None = None,
        restored_info: list[ProcessResult] | None = None,
        control_structure_callback: Callable[..., Any] | None = None,
        annotations: dict[str, Any] | None = None,
        notebook_cells: list[str] | None = None,
    ) -> list[ProcessResult]:
        """Re-execute a list of statements and return their metrics.

        *annotations* maps a statement's code to the ``# @cash:`` directives
        written on it in its cell (see :meth:`statement_directives`).
        """
        executed_metrics = []
        total_upstream_steps = len(statements)

        for stmt_idx, stmt_code in enumerate(statements):
            # The badge's "Upstream" section is what tells the user; this line
            # is for ``%cash_debug`` (and the integration tests that read it).
            if logger.isEnabledFor(logging.DEBUG):
                stmt_short = stmt_code.split("\n")[0][:40]
                if len(stmt_code) > 40:
                    stmt_short += "..."
                logger.debug("Cash: Auto-executing upstream statement: %s", stmt_short)

            try:
                # Check if this is a control structure (for/if/try/with/while).
                # If so, delegate to the control structure callback which handles
                # per-iteration caching for loops instead of executing monolithically.
                ctrl_node = self._try_parse_control_structure(stmt_code)
                if ctrl_node is not None and control_structure_callback is not None:
                    logger.debug("[UPSTREAM] Delegating control structure to per-iteration processor")
                    ctrl_annotation = (annotations or {}).get(stmt_code)
                    ctrl_result = (
                        control_structure_callback(
                            ctrl_node, ttl=global_ttl, silent=True, inherited_annotation=ctrl_annotation
                        )
                        if ctrl_annotation is not None
                        else control_structure_callback(ctrl_node, ttl=global_ttl, silent=True)
                    )
                    for m in ctrl_result.metrics:
                        if m:
                            m["is_upstream"] = True
                            m.setdefault("upstream_statement", stmt_code)
                            executed_metrics.append(m)
                    if not ctrl_result.success:
                        raise ctrl_result.error or RuntimeError("Error in upstream control structure")
                else:
                    # Silent, like the control-structure branch above: this is
                    # ANOTHER cell's statement, and its output belongs to that
                    # cell -- a plain run of this cell never prints it. Re-running
                    # a figure's history (``print('panels', len(axes))`` among
                    # its fills) put that line into an unrelated cell's output.
                    # A failure still
                    # surfaces: the processor reports it in ``result['error']``.
                    stmt_annotation = (annotations or {}).get(stmt_code)
                    result = (
                        process_callback(stmt_code, global_ttl, silent=True, annotation=stmt_annotation)
                        if stmt_annotation is not None
                        else process_callback(stmt_code, global_ttl, silent=True)
                    )
                    logger.debug("[UPSTREAM] Callback result for '%s...': %s", stmt_code[:20], result)
                    if result:
                        result["is_upstream"] = True  # Mark as upstream so badge categorizes correctly
                        executed_metrics.append(result)
                        # The processor reports statement failures via the
                        # 'error' field instead of raising - surface those
                        # through the same loud path below.
                        # With its type: ``str(KeyError('f1'))`` is just ``'f1'``,
                        # and nothing below could tell a NameError from it --
                        # every repair failure took this path and got
                        # "fix the upstream cell" for a cell with nothing wrong.
                        error = result.get("error")
                        if error:
                            text = (
                                f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
                            )
                            raise UpstreamStateError(
                                self._format_upstream_failure(
                                    stmt_code,
                                    text,
                                    planning_gap=self._planning_gap_for(
                                        error, statements[:stmt_idx], stmt_code=stmt_code, notebook_cells=notebook_cells
                                    ),
                                )
                            )
            except UpstreamStateError:
                raise
            except Exception as e:  # noqa: BLE001 - see below
                # ANY exception, not a list of the ones user code was expected
                # to raise. The statement being re-run is the user's, so what it
                # raises is the user's failure to hear about. The list used to
                # be (RuntimeError, NameError, KeyError, TypeError, ValueError),
                # and an ImportError went straight past it into the executor's
                # catch-all for cash's own bugs: a notebook whose cell wrote an
                # .xlsx without openpyxl installed told the user "cash hit an
                # internal error ... Nothing in your code caused this ... Please
                # report it" on the FOLLOWING cell.
                # Swallowing here served the downstream cell a STALE value
                # while the upstream producer was silently broken -
                # the worst failure mode for a caching layer. Fail the user's
                # cell with the upstream failure instead, like a plain
                # top-to-bottom run would.
                logger.error("[ERROR] Failed to auto-execute statement: %s", e)
                raise UpstreamStateError(
                    self._format_upstream_failure(
                        stmt_code,
                        f"{type(e).__name__}: {e}",
                        planning_gap=self._planning_gap_for(
                            e, statements[:stmt_idx], stmt_code=stmt_code, notebook_cells=notebook_cells
                        ),
                    )
                ) from e

            # Report progress after each statement
            if progress_callback is not None:
                try:
                    all_so_far = (restored_info or []) + executed_metrics
                    progress_callback(all_so_far, stmt_code, stmt_idx + 1, total_upstream_steps)
                except (TypeError, ValueError, AttributeError):
                    pass  # Don't let progress reporting break execution

        return executed_metrics

    def _planning_gap_for(
        self,
        exc: object,
        already_scheduled: list[str],
        *,
        stmt_code: str | None = None,
        notebook_cells: list[str] | None = None,
    ) -> str | None:
        """Say so when the failure is a gap in cash's repair, not the user's code.

        A statement cash re-runs as a repair can fail for two very different
        reasons, which the user cannot tell apart from the error alone:

        * the code really fails -- a top-to-bottom run would fail too, and
          "fix the upstream cell" is right;
        * cash scheduled the statement that READS something without the one
          that WRITES it, or ran it against incomplete state -- cash's problem,
          and there is nothing in the cell to fix.

        ``ax.plot(sub[...])`` without ``sub = mm[...]`` four statements
        earlier. In four projects: ``name 'in_cents' is not defined`` with
        ``in_cents = ...`` above it in the same cell; ``KeyError: 'f1'`` right
        below ``results["f1"] = ...``; ``KeyError: 'logreg'`` for a dict whose
        filling loop was not re-run. Each user went looking for a bug in a
        correct cell.

        Evidence, strongest first: a statement above the failing one, not
        re-run first, that writes what is missing -- the name, or the key or
        attribute on a variable the failing statement reads; else this exact
        statement ran without error before. ``None`` when neither holds.
        """
        try:
            missing = self._what_is_missing(exc)
            if missing is None:
                return None
            kind, name = missing
            producer = self._unscheduled_producer(kind, name, stmt_code, already_scheduled, notebook_cells)
            if producer is not None:
                code, cell_no = producer
                first = code.split("\n")[0][:60]
                where = f" (cell {cell_no})" if cell_no else ""
                run_it = f"run cell {cell_no}" if cell_no else "run the cell holding it"
                return (
                    f"NOTE: {name!r} is set by {first!r}{where}, which cash did not "
                    f"re-run first. That is a gap in cash's re-execution plan, not "
                    f"something wrong with your code - to continue, {run_it} yourself "
                    f"and then this cell again (or Restart & Run All), and please report it"
                )
            ran_before = {c for c in (self.tracking_state.executed_cell_codes or {}).values() if isinstance(c, str)}
            if stmt_code and stmt_code in ran_before:
                return (
                    "NOTE: this exact statement ran without error before, so cash most "
                    "likely rebuilt it against incomplete state - your code is probably "
                    "fine unless a file it reads changed or a line it needs was removed. "
                    "To continue, run the cells "
                    "above it yourself (or Restart & Run All), and please report it"
                )
        except Exception:  # noqa: BLE001 - a diagnostic must never mask the error
            return None
        return None

    @staticmethod
    def _what_is_missing(exc: object) -> tuple[str, str] | None:
        """``('name', x)``, ``('key', k)`` or ``('attr', a)``: what the three
        errors a statement run against incomplete state raises are missing."""
        text = str(exc)
        if isinstance(exc, NameError) or "is not defined" in text:
            m = re.search(r"name '([^']+)' is not defined", text)
            return ("name", m.group(1)) if m else None
        if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], (str, int)):
            return ("key", str(exc.args[0]))
        if isinstance(exc, AttributeError):
            m = re.search(r"has no attribute '([^']+)'", text)
            return ("attr", m.group(1)) if m else None
        return None

    def _unscheduled_producer(
        self,
        kind: str,
        name: str,
        stmt_code: str | None,
        already_scheduled: list[str],
        notebook_cells: list[str] | None,
    ) -> tuple[str, int | None] | None:
        """The last statement above *stmt_code*, not re-run first, writing *name*."""
        scheduled = set(already_scheduled)
        reads: set[str] = set()
        if stmt_code:
            try:
                reads = {
                    n.id
                    for n in ast.walk(ast.parse(stmt_code))
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
            except SyntaxError:
                reads = set()
        if notebook_cells and stmt_code:
            found: tuple[str, int | None] | None = None
            for cell_idx, cell in enumerate(notebook_cells):
                tree = parse_cell_source(cell)
                if tree is None:
                    continue
                for node in tree.body:
                    code = ast.unparse(node)
                    if code == stmt_code:
                        return found
                    if code not in scheduled and self._writes(node, kind, name, reads):
                        found = (code, cell_idx + 1)
            return None
        if kind == "name":
            for code, outputs in self._known_producers():
                if name in outputs and code not in scheduled:
                    return code, None
        return None

    @staticmethod
    def _writes(node: ast.AST, kind: str, name: str, reads: set[str]) -> bool:
        """Whether *node* writes the missing name, or the missing key or
        attribute on a variable the failing statement reads."""
        for sub in ast.walk(node):
            if kind == "name":
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if sub.name == name:
                        return True
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    if any((a.asname or a.name.split(".")[0]) == name for a in sub.names):
                        return True
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store) and sub.id == name:
                    return True
                continue
            if not (isinstance(sub, (ast.Subscript, ast.Attribute)) and isinstance(sub.ctx, ast.Store)):
                continue
            if not (isinstance(sub.value, ast.Name) and sub.value.id in reads):
                continue
            if isinstance(sub, ast.Attribute):
                if sub.attr == name:
                    return True
            elif isinstance(sub.slice, ast.Constant):
                # ``df["week"] = ...`` is also what ``df.week`` reads.
                if str(sub.slice.value) == name:
                    return True
            elif kind == "key":
                return True  # ``results[name] = ...`` in the loop that fills it
        return False

    def _known_producers(self) -> list[tuple[str, set[str]]]:
        """``(statement code, names it assigns)`` for statements cash has seen.

        Read off ``executed_cell_codes``, which maps a variable to the statement
        that last produced it -- already maintained, so this costs nothing until
        something has actually failed.
        """
        pairs: dict[str, set[str]] = {}
        for var, code in (self.tracking_state.executed_cell_codes or {}).items():
            if isinstance(code, str):
                pairs.setdefault(code, set()).add(var)
        return list(pairs.items())

    @staticmethod
    def _format_upstream_failure(
        stmt_code: str,
        error_text: str,
        planning_gap: str | None = None,
    ) -> str:
        """One-line, embeddable message for an upstream statement failure.

        The executor re-raises this inside the user's cell via a generated
        ``raise ...('''<msg>''')`` statement, so the message must stay a
        single line and must not contain a triple quote.
        """
        stmt_short = stmt_code.split("\n")[0][:60]
        if len(stmt_code) > 60 or "\n" in stmt_code:
            stmt_short += "..."

        # "fix the upstream cell" is right for a statement that genuinely
        # raised, and actively misleading for a NameError. That one usually
        # means the cell simply has not run in THIS kernel -- there is nothing
        # to fix, and the remedy (run that cell) is the one thing the old text
        # never suggested. A user lost time to exactly that: five
        # cells blocked at once, the message pointing at a cell they had just
        # read through and found nothing wrong with.
        advice = "fix the upstream cell and re-run"
        missing = re.search(r"NameError: name '([^']+)' is not defined", error_text)
        if planning_gap:
            # The note says what to do; "fix the upstream cell" beside it would
            # send the user to a cell with nothing wrong in it.
            advice = ""
        elif missing:
            advice = (
                f"run the cell that defines '{missing.group(1)}' - it has not "
                f"run in this kernel yet, so there may be nothing to fix"
            )

        msg = (
            f"Upstream statement {stmt_short!r} failed during auto-"
            f"re-execution: {error_text}. Cash stopped instead of running "
            f"this cell against stale upstream state" + (f" - {advice}." if advice else ".")
        )
        if planning_gap:
            msg = f"{msg} {planning_gap}."
        return msg.replace("'''", '"""').replace("\n", " ")

    def _try_parse_control_structure(self, code: str) -> ast.AST | None:
        """Parse code and return the AST node if it's a single control structure."""
        tree = parse_cached(code)
        if tree and len(tree.body) == 1 and is_control_structure(tree.body[0]):
            return tree.body[0]
        return None
