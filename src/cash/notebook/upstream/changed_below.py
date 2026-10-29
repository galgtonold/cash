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
import itertools
import logging
import types

from cash.control_markers import strip_markers

from ..._user_attrs import attribute_or, builtin_base, instance_of, own_dict
from ...analysis.ast_util import parse_cached
from ...analysis.cacheability import analyze_statement
from ...analysis.code_analyzer import clean_cell_source, parse_cell_source
from ...diagnostics import log_diagnostic, warn_diagnostic
from ...exceptions import CashWarning

__all__ = ["held_objects", "warn_changed_below"]

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


#: How deep, and over how many objects, `held_objects` looks: a default
#: list of 20,000 records is walked whole, in about 15 ms.
_HOLD_DEPTH = 4
_HOLD_ITEMS = 50_000

_ATOMS = (str, bytes, int, float, complex, bool, type(None), types.ModuleType)


def held_objects(holder: object) -> tuple[set[int], bool]:
    """The ids of the objects *holder* holds itself: as a default, a closure
    cell, an attribute, a ``partial`` argument, or inside a container one of
    those is, `_HOLD_DEPTH` deep and `_HOLD_ITEMS` objects at most. By
    identity: a list that only shares elements with ``BIG`` does not hold
    it. The flag says whether the walk saw everything; cut short by a
    limit, or by an object that cannot say what it holds, it may have
    missed one."""
    seen: set[int] = set()
    budget = [_HOLD_ITEMS]
    complete = [True]

    def walk(obj: object, depth: int) -> None:
        if id(obj) in seen or instance_of(obj, _ATOMS):
            # A module holds every global: ``__self__`` of ``math.sqrt``.
            return
        seen.add(id(obj))
        if depth <= 0 or budget[0] <= 0:
            complete[0] = False
            return
        budget[0] -= 1
        try:
            children, whole = _held_directly(obj)
        except Exception:  # noqa: BLE001 - a user object that cannot say what it holds
            complete[0] = False
            return
        if not whole:
            complete[0] = False
        for child in children:
            walk(child, depth - 1)

    walk(holder, _HOLD_DEPTH + 1)
    seen.discard(id(holder))
    return seen, complete[0]


def _held_directly(obj: object) -> tuple[list, bool]:
    """What *obj* holds one step down, `_HOLD_ITEMS` at most, and whether
    that is all of it."""
    # By the builtin type's own methods, of the real type: a subclass's
    # ``values()`` or ``__len__`` may print or count, where plain Python
    # never asked.
    base = builtin_base(obj)
    if base is not None:
        if base is dict:
            items, whole = _first_items(dict.values(obj), dict.__len__(obj))
        else:
            items, whole = _first_items(base.__iter__(obj), base.__len__(obj))
        attrs = own_dict(obj) if type(obj) is not base else None
        if attrs:
            # A subclass's own attributes: one keeping its items there
            # (``self._d``) held nothing by its builtin storage alone.
            more, all_of_them = _first_items(dict.values(attrs), dict.__len__(attrs))
            return items + more, whole and all_of_them
        return items, whole
    if instance_of(obj, functools.partial):
        return [obj.func, *obj.args, *obj.keywords.values()], True
    children = [
        *(attribute_or(obj, "__defaults__") or ()),
        *(attribute_or(obj, "__kwdefaults__") or {}).values(),
        *(_cell_contents(c) for c in attribute_or(obj, "__closure__") or ()),
        attribute_or(obj, "__self__"),
        attribute_or(obj, "__func__"),
    ]
    attrs = attribute_or(obj, "__dict__")
    if not isinstance(attrs, dict):
        return children, True
    items, whole = _first_items(attrs.values(), len(attrs))
    return children + items, whole


def _first_items(items, size: int) -> tuple[list, bool]:
    return list(itertools.islice(items, _HOLD_ITEMS)), size <= _HOLD_ITEMS


def _cell_contents(cell: object) -> object:
    try:
        return cell.cell_contents  # type: ignore[attr-defined]
    except ValueError:
        return None


def warn_changed_below(
    name: str,
    last_code: str | None,
    cells: list[str],
    cell_idx: int,
    warned: set[tuple[int, str, str]],
    through: str | None = None,
    maybe: bool = False,
) -> bool:
    """Warn NOTEBOOK-CHANGED-BELOW when *name*, an input of cell *cell_idx*
    that is about to be rebuilt as the cells above leave it, was last changed
    in place by *last_code*, a statement of a cell below. *through*: the
    input the cell reads, when *name* is a variable whose object that input
    holds (``def draw(k, rows=BIG)`` holds ``BIG``, which ``BIG.append(...)``
    changes); *maybe*: the walk over *through* was cut short, so it may
    hold *name*. Once per cell, name and statement (*warned*). Returns
    whether it warned."""
    if not last_code or not 0 <= cell_idx < len(cells):
        return False
    code = strip_markers(last_code).strip()
    if not code:
        return False
    found = _cell_below_holding(code, cells, cell_idx)
    if found is None or not _changes_in_place(code, name):
        return False
    below, written = found
    holds = "may hold" if maybe else "holds"
    reads = f"'{name}'" if through is None else f"'{through}', which {holds} '{name}'"
    key = (cell_idx, name, code)
    if key in warned:
        return False
    warned.add(key)
    first = written.strip().splitlines()[0]
    if len(first) > 60:
        first = first[:57] + "..."
    what = (
        f"cell {cell_idx + 1} reads {reads}, which cell {below + 1} below changed in place "
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
