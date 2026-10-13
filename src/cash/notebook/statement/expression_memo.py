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
is kept, and only when the statement that showed it was keyed, read no file,
drew no random number and printed nothing. It is handed out only while its
key as a statement of its own (`compute_cache_key`: the code it calls and
the globals that code reads, not just the names in it) is the one it had
when shown: the same value, as for the cache itself. A value that is, or
may share memory with, a variable is not kept (``f(df)`` returning ``df``
or a view of it: a copy in its place would not write through), nor one that
holds mutable objects (a column of lists, which a copy shares).
"""

from __future__ import annotations

import ast
import copy
import sys
import types
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from cash.analysis.ast_util import copy_tree

from ..shared_objects import output_history

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


#: What ``pandas.api.types.infer_dtype`` calls an object column whose items
#: are all immutable: a copy that shares them is as good as a deep one.
_IMMUTABLE_ITEMS = frozenset(
    {
        "empty",
        "string",
        "bytes",
        "integer",
        "floating",
        "mixed-integer-float",
        "boolean",
        "decimal",
        "complex",
        "datetime64",
        "datetime",
        "date",
        "time",
        "timedelta64",
        "timedelta",
        "period",
    }
)


def _object_parts(value: Any) -> list[Any] | None:
    """The object-dtype parts of a frame, series or array, or None when
    they cannot be listed."""
    dtype = getattr(value, "dtype", None)
    if dtype is not None:  # a series or an array
        return [value] if getattr(dtype, "kind", "") == "O" else []
    try:
        dtypes = value.dtypes
        return [value.iloc[:, i] for i, d in enumerate(dtypes) if getattr(d, "kind", "") == "O"]
    except Exception:  # noqa: BLE001 - not listed: not kept
        return None


def _holds_only_immutables(value: Any) -> bool:
    """Whether every object *value* holds in an object column is immutable
    (a string, a number, a date): ``copy()`` shares those objects, so a
    list in one would be changed for every statement handed a copy."""
    parts = _object_parts(value)
    if parts is None:
        return False
    if not parts:
        return True
    pandas = sys.modules.get("pandas")
    if pandas is None:
        return False
    try:
        infer = pandas.api.types.infer_dtype
        return all(
            infer(part.ravel() if hasattr(part, "ravel") and part.ndim > 1 else part, skipna=True) in _IMMUTABLE_ITEMS
            for part in parts
        )
    except Exception:  # noqa: BLE001 - not told: not kept
        return False


def _arrays(value: Any) -> list[Any] | None:
    """The numpy arrays holding *value*'s data, or None when they cannot
    be listed."""
    numpy = sys.modules.get("numpy")
    if numpy is None:
        return None
    if isinstance(value, numpy.ndarray):
        return [value]
    try:
        held = value._mgr.arrays  # a series' or frame's blocks
    except Exception:  # noqa: BLE001 - not listed: not kept
        return None
    arrays = []
    for array in held:
        if isinstance(array, numpy.ndarray):
            arrays.append(array)
            continue
        # An extension array: its numpy parts, where it has them.
        for attr in ("_ndarray", "_data", "_mask", "_codes"):
            part = getattr(array, attr, None)
            if isinstance(part, numpy.ndarray):
                arrays.append(part)
    return arrays


def _shares_with_a_variable(value: Any, namespace: Mapping[str, Any]) -> bool:
    """Whether *value* is, or may share memory with, a frame, series or
    array a variable holds: ``f(df)`` returning ``df`` or a view of it.
    A copy in its place would not write through to the variable.

    IPython's output history (``Out``, and ``_``, ``__``, ``_<n>`` bound to
    a value in it) is not such a variable: it holds the shown value itself.
    A variable of the user's whose name starts with ``_`` (``_arr``) is."""
    mine = _arrays(value)
    if mine is None:
        return True
    numpy = sys.modules["numpy"]
    _, held = output_history(namespace)
    history = {name for holder, name in held if holder is namespace}
    for name, other in list(namespace.items()):
        if name in history or not _kept_type(other):
            continue
        if other is value:
            return True
        theirs = _arrays(other)
        if theirs is None:
            return True
        for a in mine:
            for b in theirs:
                if numpy.may_share_memory(a, b):
                    return True
    return False


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
        self._kept: OrderedDict[tuple[str, tuple[tuple[str, str], ...]], tuple[Any, float, str]] = OrderedDict()

    def _key(
        self, node: ast.expr, lineages: Mapping[str, str], namespace: Mapping[str, Any]
    ) -> tuple[str, tuple[tuple[str, str], ...]] | None:
        names = _names(node)
        if not names:
            return None
        parts = []
        for name in sorted(names):
            if name not in namespace:
                return None  # ``del df``: the statement must raise as it would
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
        key_of: Callable[[str], str | None],
        computed_under: str | None = None,
        reads_a_stream: Callable[[], bool] | None = None,
    ) -> None:
        """Keep *value*, which the bare expression *tree* computed in *seconds*.

        *key_of* gives an expression's key as a statement of its own, now;
        the value is handed out only while that key stays the same. It is
        not kept when that key is no longer *computed_under*, the key the
        statement had before it ran: running it moved something its key
        folds in (``rng.random()`` on a generator a global holds), so
        computing it again gives another value. Nor when *reads_a_stream*
        says it reads an iterator or a generator its key does not follow
        (``next(counter)``)."""
        if seconds < _MIN_SECONDS or not _kept_type(value):
            return
        if tree is None or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return
        key = self._key(tree.body[0].value, lineages, namespace)
        if key is None:
            return
        if not _holds_only_immutables(value) or _shares_with_a_variable(value, namespace):
            return
        statement_key = key_of(key[0])
        if statement_key is None or (computed_under is not None and statement_key != computed_under):
            return
        if reads_a_stream is not None and reads_a_stream():
            return
        try:
            kept = copy.deepcopy(value) if not hasattr(value, "copy") else value.copy()
        except Exception:  # noqa: BLE001 - a value that cannot be copied is not kept
            return
        self._kept[key] = (kept, seconds, statement_key)
        self._kept.move_to_end(key)
        while len(self._kept) > _KEPT:
            self._kept.popitem(last=False)

    def forget(self) -> None:
        self._kept.clear()

    def substitute(
        self,
        tree: ast.Module,
        lineages: Mapping[str, str],
        namespace: Mapping[str, Any],
        key_of: Callable[[str], str | None],
    ) -> tuple[ast.Module, dict[str, Any], float] | None:
        """*tree* with each kept expression it computes replaced by a name,
        the values to bind those names to, and the seconds they took; None
        when it computes none of them. A kept expression whose key as a
        statement (*key_of*) moved on since it was shown (``f`` reads a
        global that changed) is computed again, and forgotten."""
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
                    if hit is not None and key_of(key[0]) != hit[2]:
                        del memo._kept[key]
                        hit = None
                    if hit is not None:
                        value, seconds, _ = hit
                        name = f"{MEMO_NAME_PREFIX}{len(self.bound)}"
                        self.bound[name] = value.copy()
                        self.saved += seconds
                        return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
                return self.generic_visit(node)

        # Not ``copy.deepcopy``: after a traceback the shared ``ast.Load()``
        # carries a ``parent`` link into a whole library file (`copy_tree`),
        # and every statement of every cell paid for copying it.
        new_tree = copy_tree(tree)
        put = _Put()
        new_tree.body[0].value = put.visit(new_tree.body[0].value)
        bound, saved = put.bound, put.saved
        if not bound:
            return None
        ast.fix_missing_locations(new_tree)
        return new_tree, bound, saved
