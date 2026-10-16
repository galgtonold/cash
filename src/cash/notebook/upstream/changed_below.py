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


def _changes_in_place(code: str, name: str) -> bool:
    """Does the statement *code* change *name* in place without rebinding it?"""
    tree = parse_cached(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return False
    try:
        return name in analyze_statement(code, None).top_level_mutated_vars
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return False


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
    """*code* as cell *cell* writes it, or None when the cell does not hold it."""
    if code in cell:
        return code
    return _statements_of(cell).get(code)


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
) -> bool:
    """Warn NOTEBOOK-CHANGED-BELOW when *name*, an input of cell *cell_idx*
    that is about to be rebuilt as the cells above leave it, was last changed
    in place by *last_code*, a statement of a cell below. Once per cell,
    name and statement (*warned*). Returns whether it warned."""
    if not last_code or not 0 <= cell_idx < len(cells):
        return False
    code = strip_markers(last_code).strip()
    if not code:
        return False
    found = _cell_below_holding(code, cells, cell_idx)
    if found is None or not _changes_in_place(code, name):
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
        f"cell {cell_idx + 1} reads '{name}', which cell {below + 1} below changed in place "
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
