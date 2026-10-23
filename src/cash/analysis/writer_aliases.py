"""The names a namespace binds to a writer, for `mutations.writer_of`.

``sa = setattr`` in a cell above, then ``sa(points[7], 'x', -3)``: the
statement's text alone does not say ``sa`` writes. The namespace does.
"""

from __future__ import annotations

import ast
import builtins
import functools
import operator
import types
from collections.abc import Iterable, Mapping
from typing import Any

from .mutations import A_HELD_WRITER, MUTATING_METHODS, THE_NAMESPACE, WRITER_ROOTS, WRITER_SPELLINGS, WRITES_INTO

__all__ = ["namespace_writer_aliases", "writer_names_aliases"]


def _writer_objects() -> dict[int, tuple[Any, str]]:
    """``id(object) -> (object, spelling)`` for each writer, each root a
    writer is read from, and the two modules."""
    found: dict[int, tuple[Any, str]] = {}
    for spelling in (*WRITER_SPELLINGS, *WRITER_ROOTS):
        root, *rest = spelling.split(".")
        value: Any = {"operator": operator, "builtins": builtins}.get(root) or getattr(builtins, root, None)
        for attr in rest:
            value = getattr(value, attr, None)
        if value is not None:
            found.setdefault(id(value), (value, spelling))
    return found


_WRITER_OBJECTS = _writer_objects()
_MISSING = object()


def writer_names_aliases(names: Iterable[str], namespace: Mapping[str, Any]) -> dict[str, str | None]:
    """``name -> spelling`` for each of *names* *namespace* binds to a writer
    or to what one is read from (``sa`` to ``setattr``, ``op`` to
    ``operator``); ``name -> None`` for a writer's root bound to something
    else (``object`` a class of the user's)."""
    found: dict[str, str | None] = {}
    for name in names:
        try:
            value = namespace.get(name, _MISSING)
        except Exception:  # noqa: BLE001 - a namespace that cannot be read binds nothing known
            continue
        if value is _MISSING:
            continue
        known = _WRITER_OBJECTS.get(id(value))
        if known is not None and known[0] is value:
            if known[1] != name:
                found[name] = known[1]
        elif _is_the_namespace(value, namespace):
            found[name] = THE_NAMESPACE  # ``ns = globals()``
        elif (held := _held_writer(value)) is not None:
            found[name] = _writes_into(held, namespace)
        elif name in WRITER_ROOTS:
            found[name] = None
    return found


def _is_the_namespace(value: Any, namespace: Mapping[str, Any]) -> bool:
    maps = getattr(namespace, "maps", None)
    return value is namespace or (isinstance(maps, list) and any(value is m for m in maps))


def _held_writer(value: Any) -> tuple[Any] | None:
    """``(the object it writes into,)`` for a writer bound to it: a builtin
    method that changes its object (``rows.append``, ``p.__setattr__``), a
    ``functools.partial`` of a writer and its object. None for anything else."""
    if isinstance(value, (types.BuiltinMethodType, types.MethodWrapperType)):
        owner = getattr(value, "__self__", None)
        name = getattr(value, "__name__", "")
        if owner is not None and not isinstance(owner, types.ModuleType):
            if name in MUTATING_METHODS or name in _WRITING_DUNDERS:
                return (owner,)
        return None
    if isinstance(value, functools.partial) and value.args:
        known = _WRITER_OBJECTS.get(id(value.func))
        if known is not None and known[0] is value.func and known[1] in WRITER_SPELLINGS:
            return (value.args[0],)
    return None


_WRITING_DUNDERS = frozenset({"__setattr__", "__delattr__", "__setitem__", "__delitem__"})


def _writes_into(held: tuple[Any], namespace: Mapping[str, Any]) -> str:
    """`WRITES_INTO` the variable holding the object a held writer writes
    into, or `A_HELD_WRITER` when no variable holds it itself (an item of a
    list: ``partial(setattr, points[7])``)."""
    target = held[0]
    try:
        for name, value in namespace.items():
            if value is target and not name.startswith("_"):
                return WRITES_INTO + name
    except Exception:  # noqa: BLE001 - a namespace that cannot be read: no name known
        pass
    return A_HELD_WRITER


def namespace_writer_aliases(tree: ast.AST, namespace: Mapping[str, Any]) -> dict[str, str | None]:
    """`writer_names_aliases` for the names *tree* reads."""
    return writer_names_aliases({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}, namespace)
