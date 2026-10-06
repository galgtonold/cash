"""What a statement that sets state on a local module left there, for a reload.

A reload of an edited module runs its top level again: ``mylib.K = k`` that a
cell ran is gone, and the cell is not run again. The kernel had ``K`` as the
cell set it, and so would Restart & Run All. ``CellExecutor`` puts it back:
it runs the statement again when what the statement reads still holds what
it held when it ran, and otherwise sets back the values it left (``mylib.K =
k`` with ``k`` rebound since, ``mylib.K = int(rng.integers(100))`` with the
generator moved on by the draw itself). Run again, either would set a value
the notebook never set.
"""

from __future__ import annotations

import ast
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..analysis.ast_util import parse_cached
from ..tracking.function_tracker import is_local_module

__all__ = ["ModuleStateWriter", "rebound_attributes"]


@dataclass
class ModuleStateWriter:
    """A statement that sets state on a local module (``TrackingState.module_state_writers``)."""

    code: str
    #: The lineage of each variable it reads, but a module, as it ran.
    input_lineages: Mapping[str, str | None]
    #: ``(module, attribute) -> value`` it left, for a statement that only
    #: binds attributes of local modules; None for any other.
    left: dict[tuple[str, str], Any] | None = None


def rebound_attributes(code: str, namespace: Mapping[str, Any]) -> list[tuple[str, str]] | None:
    """``(module, attribute)`` of each attribute of a local module *code*
    binds when that is all it does (``mylib.K = v``, ``mylib.A, mylib.B =
    pair``, ``setattr(mylib, "K", v)``); None otherwise."""
    tree = parse_cached(code)
    if tree is None or len(tree.body) != 1:
        return None
    node = tree.body[0]
    targets: list[ast.expr]
    if isinstance(node, ast.Assign):
        targets = []
        for target in node.targets:
            targets.extend(target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target])
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        targets = [node.target]
    elif (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "setattr"
        and len(node.value.args) == 3
        and isinstance(node.value.args[0], ast.Name)
        and isinstance(node.value.args[1], ast.Constant)
        and isinstance(node.value.args[1].value, str)
    ):
        module = _local_module(namespace.get(node.value.args[0].id))
        return [(module, node.value.args[1].value)] if module is not None else None
    else:
        return None
    found = []
    for target in targets:
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
            return None
        module = _local_module(namespace.get(target.value.id))
        if module is None:
            return None
        found.append((module, target.attr))
    return found


def _local_module(value: Any) -> str | None:
    if not isinstance(value, types.ModuleType):
        return None
    try:
        return value.__name__ if is_local_module(value) else None
    except (TypeError, AttributeError):
        return None
