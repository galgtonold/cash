"""Randomness as an input: the seed epoch in the key, draws replayed on a
hit, and the warnings about unseeded draws."""

from __future__ import annotations

import ast
import inspect
import types
from typing import Any

#: Seeding calls, by the last segment of their dotted name. ``seed`` alone is
#: too common a method name, so it only counts under a ``random`` prefix.
_SEEDING_CALLS = frozenset(
    {
        "default_rng",
        "RandomState",
        "Random",
        "manual_seed",
        "SeedSequence",
        "PCG64",
        "PCG64DXSM",
        "MT19937",
        "Philox",
        "SFC64",
    }
)


def _seed_access_path(node: ast.AST) -> tuple[str, tuple[tuple[str, Any], ...]] | None:
    """``settings.sim.seed`` -> ``("settings", (("attr", "sim"), ("attr", "seed")))``;
    ``opts["seed"]`` -> ``("opts", (("item", "seed"),))``; None for anything
    else (a call, a computed key, an expression)."""
    path: list[tuple[str, Any]] = []
    while True:
        if isinstance(node, ast.Attribute):
            path.append(("attr", node.attr))
            node = node.value
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            path.append(("item", node.slice.value))
            node = node.value
        else:
            break
    if not isinstance(node, ast.Name):
        return None
    return node.id, tuple(reversed(path))


def seed_parameters(src: str) -> dict[str, tuple[str, str, bool, tuple]]:
    """``{seed expression: (call, root name, root is a parameter, path)}``.

    For seeding calls whose seed is a parameter, or an attribute or
    constant-key item reached from a parameter or a module global:
    ``default_rng(seed)``, ``default_rng(settings.seed)``,
    ``default_rng(opts["seed"])``, ``default_rng(CONFIG.seed)``. A root the
    function assigns itself is a local, which cannot be read before the call,
    and is skipped.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return {}
    a = fn.args
    params = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    assigned = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    found: dict[str, tuple[str, str, bool, tuple]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        dotted = ast.unparse(node.func)
        last = dotted.rsplit(".", 1)[-1]
        if not (last in _SEEDING_CALLS or (last == "seed" and "random" in dotted)):
            continue
        seed_arg = (
            node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in ("seed", "x", "a")), None)
        )
        if seed_arg is None:
            continue
        access = _seed_access_path(seed_arg)
        if access is None:
            continue
        root, path = access
        is_param = root in params
        if not is_param and root in assigned:
            continue
        expr = ast.unparse(seed_arg)
        found.setdefault(expr, (f"{dotted}({expr})", root, is_param, path))
    return found


_SEED_UNREADABLE = object()


def read_seed(value: Any, path: tuple) -> Any:
    """Follow *path* from *value* WITHOUT running user code, or `_SEED_UNREADABLE`.

    Attributes through ``inspect.getattr_static``: a plain instance or class
    attribute is read, a property or other descriptor is not evaluated. Items
    only from a plain mapping. This runs on every call, so it may not call
    anything the user wrote.
    """
    for kind, key in path:
        if kind == "attr":
            try:
                value = inspect.getattr_static(value, key)
            except AttributeError:
                return _SEED_UNREADABLE
            if hasattr(type(value), "__get__") and not isinstance(
                value, (types.FunctionType, types.BuiltinFunctionType)
            ):
                return _SEED_UNREADABLE  # a property or descriptor
        elif isinstance(value, (dict, types.MappingProxyType)):
            value = value.get(key, _SEED_UNREADABLE)
            if value is _SEED_UNREADABLE:
                return value
        else:
            return _SEED_UNREADABLE
    return value
