"""The value of an expression a cell just showed, handed to the next
statement that computes it again.

A notebook is edited a step at a time: ``temp.unstack().reset_index(drop=True)``
is shown, then kept as ``imperfect_cell = temp.unstack().reset_index(drop=True)``;
``df_1[1].str.split("->", expand=True)`` is shown, then extended to
``df_1[1].str.split("->", expand=True)[0]``. Each new text is a new
statement, a cache miss, and computed the same expensive value again (12 s
for the ``unstack`` in a student's notebook).

When a cell's last line, a bare expression, took a while to compute a frame,
series or array, its value is kept here under the expression's text and the
lineage of each name it reads. A later statement that executes is searched
for that same expression, as its whole right-hand side or as the receiver of
the calls and subscripts chained onto it, and the kept value is put in its
place: a copy, so the statement can change what it is handed.

Only an expression that reads variables cash names by lineage, and modules,
is kept, and only when the statement that showed it was keyed, read no file
and drew no random number: the same text over the same lineages is then the
same value, as for the cache itself.
"""

from __future__ import annotations

import ast
import copy
import types
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

__all__ = ["ExpressionMemo", "MEMO_NAME_PREFIX"]

#: The names a kept value is bound to while the statement runs.
MEMO_NAME_PREFIX = "__cash_memo_"

#: A shown expression is kept when it took at least this long: copying the
#: value back out costs a fraction of a second for millions of rows.
_MIN_SECONDS = 0.5
#: How many shown values are kept: the latest few, which the next edits use.
_KEPT = 4

_KEPT_TYPES = {("pandas", "DataFrame"), ("pandas", "Series"), ("numpy", "ndarray")}


def _kept_type(value: Any) -> bool:
    t = type(value)
    return ((t.__module__ or "").split(".", 1)[0], t.__name__) in _KEPT_TYPES


def _names(node: ast.AST) -> set[str] | None:
    """The variable names *node* reads, or None when it binds any of its own
    (a comprehension, a lambda, a walrus): those are kept out."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Lambda, ast.comprehension, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
            return None
        if isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


class ExpressionMemo:
    """The latest values shown by a cell's last line, by text and lineage."""

    def __init__(self) -> None:
        self._kept: OrderedDict[tuple[str, tuple[tuple[str, str], ...]], tuple[Any, float]] = OrderedDict()

    def _key(
        self, node: ast.expr, lineages: Mapping[str, str], namespace: Mapping[str, Any]
    ) -> tuple[str, tuple[tuple[str, str], ...]] | None:
        names = _names(node)
        if not names:
            return None
        parts = []
        for name in sorted(names):
            if name in lineages:
                parts.append((name, lineages[name]))
            elif isinstance(namespace.get(name), types.ModuleType):
                parts.append((name, "module:" + namespace[name].__name__))
            else:
                return None
        return ast.unparse(node), tuple(parts)

    def keep(
        self,
        tree: ast.Module | None,
        value: Any,
        seconds: float,
        lineages: Mapping[str, str],
        namespace: Mapping[str, Any],
    ) -> None:
        """Keep *value*, which the bare expression *tree* computed in *seconds*."""
        if seconds < _MIN_SECONDS or not _kept_type(value):
            return
        if tree is None or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return
        key = self._key(tree.body[0].value, lineages, namespace)
        if key is None:
            return
        try:
            kept = copy.deepcopy(value) if not hasattr(value, "copy") else value.copy()
        except Exception:  # noqa: BLE001 - a value that cannot be copied is not kept
            return
        self._kept[key] = (kept, seconds)
        self._kept.move_to_end(key)
        while len(self._kept) > _KEPT:
            self._kept.popitem(last=False)

    def forget(self) -> None:
        self._kept.clear()

    def substitute(
        self, tree: ast.Module, lineages: Mapping[str, str], namespace: Mapping[str, Any]
    ) -> tuple[ast.Module, dict[str, Any], float] | None:
        """*tree* with each kept expression it computes replaced by a name,
        the values to bind those names to, and the seconds they took; None
        when it computes none of them."""
        if not self._kept or len(tree.body) != 1:
            return None
        stmt = tree.body[0]
        if not isinstance(stmt, (ast.Assign, ast.Expr, ast.AnnAssign)) or stmt.value is None:
            return None
        memo = self

        class _Put(ast.NodeTransformer):
            """Replaces each outermost kept expression; never inside a
            lambda or comprehension, which would run it once per item."""

            def __init__(self) -> None:
                self.bound: dict[str, Any] = {}
                self.saved = 0.0

            def visit(self, node: ast.AST) -> ast.AST:
                if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    return node
                if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp)) and isinstance(
                    getattr(node, "ctx", ast.Load()), ast.Load
                ):
                    key = memo._key(node, lineages, namespace)
                    hit = memo._kept.get(key) if key is not None else None
                    if hit is not None:
                        value, seconds = hit
                        name = f"{MEMO_NAME_PREFIX}{len(self.bound)}"
                        self.bound[name] = value.copy()
                        self.saved += seconds
                        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
                return self.generic_visit(node)

        new_tree = copy.deepcopy(tree)
        put = _Put()
        new_tree.body[0].value = put.visit(new_tree.body[0].value)
        bound, saved = put.bound, put.saved
        if not bound:
            return None
        ast.fix_missing_locations(new_tree)
        return new_tree, bound, saved
