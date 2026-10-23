"""A cell re-run above an in-place change a cell below made to its input.

Re-run on its own, a cell reads its inputs as the cells above leave them:
the upstream check rebuilds an input a cell below has moved on
(``MismatchClassifier._reject_read_only_input``). When the cell below
changed the value in place (``s.iloc[0] = 1e9``), that change is dropped
with the rebuild: the re-run computes without it, and the variable no longer
holds it until the cell below runs again. :func:`warn_changed_below` says so.
"""

from __future__ import annotations

import ast
import functools
import logging

from cash.control_markers import strip_markers

from ...analysis.ast_util import parse_cached
from ...analysis.cacheability import analyze_statement
from ...analysis.code_analyzer import clean_cell_source, parse_cell_source
from ...diagnostics import log_diagnostic, warn_diagnostic
from ...exceptions import CashWarning

__all__ = ["warn_changed_below"]

logger = logging.getLogger(__name__)


def _how_changed(code: str, name: str) -> str | None:
    """How the statement *code*, which last changed *name*, changed it
    without rebinding it by name: ``"in place"`` (``s.iloc[0] = 1e9``),
    ``"through what it calls"`` (a function it calls, or a thread it starts
    and joins, setting ``D['k']`` or a ``global``), or None."""
    tree = parse_cached(code)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return None
    try:
        if name in analyze_statement(code, None).top_level_mutated_vars:
            return "in place"
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    # Recorded as the statement that last changed *name*, it changed it
    # through a call: its own text neither binds nor changes the name.
    if any(isinstance(node, ast.Call) for node in ast.walk(tree)):
        return "through what it calls"
    return None


@functools.lru_cache(maxsize=512)
def _statements_of(cell: str) -> dict[str, str]:
    """Every statement of *cell*, nested ones too, as cash records a
    statement's code (unparsed: ``1e9`` reads ``1000000000.0``), to its
    text in the cell. Read it, never change it."""
    text = clean_cell_source(cell)
    tree = parse_cell_source(cell)
    if tree is None:
        return {}
    return {
        ast.unparse(node): ast.get_source_segment(text, node) or ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt)
    }


def _as_written(cell: str, code: str) -> str | None:
    """*code* as cell *cell* writes it, or None when the cell does not hold
    it as a statement: ``fill()`` is in the text ``def fill():`` too."""
    statements = _statements_of(cell)
    if not statements:
        return code if code in cell else None
    written = statements.get(code)
    if written is not None:
        return written
    return code if code in statements.values() else None


def _cell_below_holding(code: str, cells: list[str], cell_idx: int) -> tuple[int, str] | None:
    """The one cell below *cell_idx* that holds the statement *code*, with
    the statement as written there, when no cell at or above it holds it."""
    if any(_as_written(cell, code) is not None for cell in cells[: cell_idx + 1]):
        return None
    below = [(i, written) for i in range(cell_idx + 1, len(cells)) if (written := _as_written(cells[i], code))]
    return below[0] if len(below) == 1 else None


def warn_changed_below(
    name: str,
    last_code: str | None,
    cells: list[str],
    cell_idx: int,
    warned: set[tuple[int, str, str]],
    by_work: str | None = None,
) -> bool:
    """Warn NOTEBOOK-CHANGED-BELOW when *name*, an input of cell *cell_idx*
    that is about to be rebuilt as the cells above leave it, was last changed
    in place, or through what it calls, by *last_code*, a statement of a
    cell below; or by work in a thread *by_work*, a statement of a cell
    below, handed off (a Timer started, a pool's ``submit``, a Thread
    started), which is the statement named when both are below. Once per
    cell, name and statement (*warned*). Returns whether it warned."""
    if not 0 <= cell_idx < len(cells):
        return False
    found = how = None
    for candidate, judge in ((by_work, False), (last_code, True)):
        code = strip_markers(candidate or "").strip()
        if not code:
            continue
        found = _cell_below_holding(code, cells, cell_idx)
        how = (_how_changed(code, name) if judge else "through work it handed to a thread") if found else None
        if found is not None and how is not None:
            break
    if found is None or how is None:
        return False
    below, written = found
    key = (cell_idx, name, code)
    if key in warned:
        return False
    warned.add(key)
    first = written.strip().splitlines()[0]
    if len(first) > 60:
        first = first[:57] + "..."
    what = (
        f"cell {cell_idx + 1} reads '{name}', which cell {below + 1} below changed {how} "
        f"(`{first}`). Re-run on its own, cell {cell_idx + 1} gets '{name}' rebuilt as the cells "
        f"above leave it, so it computes without that change, and '{name}' no longer holds it "
        f"until cell {below + 1} runs again."
    )
    fix = (
        f"run cell {below + 1} again to apply the change; for cell {cell_idx + 1} to see it, move "
        f"the change into a cell above cell {cell_idx + 1}."
    )
    log_diagnostic(logger, "NOTEBOOK-CHANGED-BELOW", what, fix)
    warn_diagnostic(CashWarning, "NOTEBOOK-CHANGED-BELOW", what, fix, location=("<cash>", cell_idx + 1))
    return True
