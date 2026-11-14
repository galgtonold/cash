"""Class attributes that are only what a ``for`` loop in the class body
left behind, iterating a set of strings: keyed by which set, not by which
of its elements the loop happened to end on (`set_loop_leftovers`)."""

from __future__ import annotations

import ast
import inspect
import textwrap
import weakref
from typing import Any

__all__ = ["leftover_value", "set_loop_leftovers"]

#: Per class: ``(size of its namespace, its data names, what was found)``.
_FOUND: weakref.WeakKeyDictionary[type, tuple[int, tuple[str, ...], dict[str, str]]] = weakref.WeakKeyDictionary()


#: What a class attribute that `set_loop_leftovers` found is keyed as, with
#: the name of the set it is left over from.
_LOOP_LEFTOVER = "cash:last element of the set"


def set_loop_leftovers(base: type, names: tuple[str, ...]) -> dict[str, str]:
    """``{name: set name}`` for the class attributes of *base* that are only
    what a ``for`` loop in its class body left behind, iterating a set of
    strings the class also holds::

        class EasyEnsembleClassifier(...):
            excluded_params = {"bootstrap", "oob_score", ...}
            for param in excluded_params:
                _parameter_constraints.pop(param, None)

    ``param`` is whichever string the set iterates last, and that order
    follows ``PYTHONHASHSEED``: keyed by its value, every process with
    another seed read the class as changed. It is keyed as "the last element
    of ``excluded_params``" instead, the set keyed beside it, whose order no
    key holds anywhere: a result built from ``list(some_set)`` is served to
    a process that iterates it in another order too.

    Only when the class body's source shows *name* bound nowhere but as the
    target of one ``for name in SET:`` at the class body's own level (no
    assignment, import, ``def``, ``with``, ``del`` or other loop binds it
    there), *SET* a name the class body binds above the loop and holds as
    data. Whether the attribute still holds what that loop leaves
    (`_is_loop_leftover`) is checked on every read: rebound to anything
    else, it is keyed by value. Without source, keyed by value.
    """
    size = len(vars(base))
    try:
        known = _FOUND.get(base)
    except TypeError:
        known = None
    if known is not None and known[0] == size and known[1] == names:
        return known[2]
    found = _find(base, names)
    try:
        _FOUND[base] = (size, names, found)
    except TypeError:
        pass
    return found


def _find(base: type, names: tuple[str, ...]) -> dict[str, str]:
    namespace = vars(base)
    sets = {
        n
        for n in names
        if isinstance(namespace.get(n), (set, frozenset)) and namespace[n] and all(type(v) is str for v in namespace[n])
    }
    if not sets:
        return {}
    wanted = {
        n
        for n in names
        if n not in sets and type(namespace.get(n)) is str and any(namespace[n] in namespace[s] for s in sets)
    }
    if not wanted:
        return {}
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(base)))
    except (OSError, TypeError, SyntaxError, ValueError, IndexError):
        return {}
    node = tree.body[0] if tree.body else None
    if not isinstance(node, ast.ClassDef) or node.name != base.__name__:
        return {}
    bound: dict[str, int] = {}
    loops: dict[str, str] = {}
    for stmt in node.body:
        if (
            isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and isinstance(stmt.iter, ast.Name)
            and stmt.iter.id in bound  # the class's own set, bound above it
        ):
            loops[stmt.target.id] = stmt.iter.id
        for sub in _class_scope_nodes(stmt):
            for n in _bound_names(sub):
                bound[n] = bound.get(n, 0) + 1
    return {n: loops[n] for n in wanted if n in loops and loops[n] in sets and bound.get(n) == 1}


def _class_scope_nodes(stmt: Any) -> list[Any]:
    """The nodes of a class-body statement that run in the class's own
    scope: not those inside a ``def``, ``lambda``, nested class or
    comprehension (whose names are their own), though their decorators,
    defaults and the comprehension's first iterable are."""
    out: list[Any] = []
    stack = [stmt]
    while stack:
        node = stack.pop()
        out.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(node.decorator_list)
            if isinstance(node, ast.ClassDef):
                stack.extend([*node.bases, *node.keywords])
            else:
                stack.extend([*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)])
            continue
        if isinstance(node, ast.Lambda):
            stack.extend([*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)])
            continue
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            stack.append(node.generators[0].iter)
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


def _bound_names(node: Any) -> list[str]:
    """The names *node* binds (or deletes) in the scope it runs in."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return [node.id]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return list(node.names)
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [node.name]
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return [node.name]
    if isinstance(node, ast.MatchMapping) and node.rest:
        return [node.rest]
    return []


def _is_loop_leftover(value: Any, over: Any) -> bool:
    """Whether *value* is what ``for value in over: pass`` leaves now: the
    last element *over* iterates, a set left as it is iterating in the same
    order (`set_loop_leftovers`)."""
    if type(value) is not str or not isinstance(over, (set, frozenset)) or not over:
        return False
    last = None
    for last in over:  # the leftover is the point
        pass
    return last == value


def leftover_value(namespace: Any, name: str, leftovers: dict[str, str]) -> Any:
    """What class attribute *name* of *namespace* is keyed as: its value,
    or, while it holds what its set loop left (*leftovers*,
    `set_loop_leftovers`), which set that is. Which element it is, is the
    set's iteration order; the set itself is keyed beside it."""
    value = namespace[name]
    over = leftovers.get(name)
    if over is not None and _is_loop_leftover(value, namespace.get(over)):
        return (_LOOP_LEFTOVER, over)
    return value
