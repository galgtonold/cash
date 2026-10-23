"""Notebook functions and classes a statement may have stored as values.

``class Rec`` run again (each Run All) binds ``Rec`` to a new class under the
same lineage, as its code is the same. ``B['c'] = Rec`` below it then looks
current to the cell run plan and is skipped, so ``B`` keeps the class the
previous run made: ``B['c'] is Rec`` is False where a run from the top gives
True, and what was set on the new class (``Rec.it = ...``) is not where the
code reading ``B['c']`` looks. The processor notes, for each statement that
runs, which such objects it may have stored (`code_values_stored`); the plan
runs it again when one of those names holds another object now
(`stored_code_changed`).
"""

from __future__ import annotations

import ast
import types
from collections.abc import Iterable, Mapping
from typing import Any

from ..._user_attrs import attribute_or, instance_of

__all__ = ["code_values_stored", "stored_code_changed"]


def code_values_stored(tree: ast.AST | None, inputs: Iterable[str], user_ns: Mapping[str, Any]) -> dict[str, int]:
    """``{name: id(object)}`` for each of *inputs* that holds a function or
    class the notebook defines and that the statement *tree* uses as a value
    -- not only calls (``Rec()``) or reads an attribute of (``Rec.n``)."""
    if tree is None:
        return {}
    module = user_ns.get("__name__", "__main__")
    candidates = {name for name in inputs if _notebook_code(user_ns.get(name), module)}
    if not candidates:
        return {}
    as_value: set[str] = set()
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.Name)
                and child.id in candidates
                and isinstance(child.ctx, ast.Load)
                and not (isinstance(node, ast.Call) and node.func is child)
                and not (isinstance(node, ast.Attribute) and node.value is child)
            ):
                as_value.add(child.id)
    return {name: id(user_ns[name]) for name in sorted(as_value)}


def stored_code_changed(recorded: Mapping[str, int] | None, user_ns: Mapping[str, Any]) -> bool:
    """Does a name the statement stored as a value hold another object now?"""
    return bool(recorded) and any(name in user_ns and id(user_ns[name]) != oid for name, oid in recorded.items())


def _notebook_code(value: Any, module: str) -> bool:
    """A function or class a notebook cell defines (in *module*, the
    notebook's), which a cell run again makes anew."""
    if not instance_of(value, (types.FunctionType, type)):
        return False
    return attribute_or(value, "__module__") == module
