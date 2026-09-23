"""What an import-only statement binds, and whether those bindings already hold.

The processor skips an import statement whose names already hold what it
would bind, and re-runs one whose names a restore could not bring back
(modules are not cacheable values). These helpers answer those questions from
the statement's AST, the user namespace and ``sys.modules``, without
importing anything.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Mapping
from typing import Any

__all__ = [
    "import_bindings_hold",
    "import_needs_reexecution",
    "import_source_modules",
    "redundant_import_names",
]


def import_source_modules(tree: ast.Module) -> set[str]:
    """Return the set of top-level module names referenced by import nodes in *tree*."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.add(node.module.split(".")[0])
    return names


def redundant_import_names(tree: ast.AST) -> set[str] | None:
    """The names an import-only *tree* binds, or ``None`` when *tree* holds
    anything but imports (and docstring-like constants), or a ``*`` import."""
    defined_names = set()

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        defined_names.add(alias.asname)
                    else:
                        defined_names.add(alias.name.split(".")[0])
            else:  # ImportFrom
                for alias in node.names:
                    if alias.asname:
                        defined_names.add(alias.asname)
                    else:
                        if alias.name == "*":
                            return None
                        defined_names.add(alias.name)

        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        else:
            return None

    return defined_names


def import_bindings_hold(tree: ast.AST, ns: Mapping[str, Any]) -> bool:
    """Does every name an import-only *tree* binds already hold the object
    that import would bind? Answered from ``sys.modules`` without importing
    anything; anything not already loaded, or relative, is "no"."""

    missing = object()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in sys.modules:
                    return False
                name = alias.asname or alias.name.split(".")[0]
                expected = sys.modules.get(alias.name if alias.asname else name)
                if expected is None or ns.get(name, missing) is not expected:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                return False
            module = sys.modules.get(node.module)
            if module is None:
                return False
            for alias in node.names:
                expected = getattr(module, alias.name, missing)
                if expected is missing:
                    expected = sys.modules.get(f"{node.module}.{alias.name}", missing)
                if expected is missing or ns.get(alias.asname or alias.name, missing) is not expected:
                    return False
    return True


def import_needs_reexecution(tree: ast.Module | None, ns: Mapping[str, Any]) -> bool:
    """True for a pure-import statement whose bound name(s) are absent from *ns*.

    A cache hit for an import would take the restore path, but restoring
    cannot rebind a *module* object (modules aren't cacheable values). On a
    fresh kernel (e.g. after a restart) the bound name is therefore missing,
    and a later statement in the same cell that uses it raises ``NameError``.
    Forcing re-execution re-imports and rebinds the name (cheap, idempotent)
    and re-stores the entry, so lineage tracking is preserved.
    """
    if tree is None:
        return False
    names = redundant_import_names(tree)
    if not names:
        return False
    return not all(name in ns for name in names)
