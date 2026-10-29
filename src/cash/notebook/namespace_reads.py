"""Code that may read any of the notebook's variables by name.

``globals()['bar']``, ``sys.modules['__main__'].bar``, ``import __main__``,
``%whos``: a scan of the names a cell reads finds none of these reads, so
a cell holding one counts as reading every variable.
"""

from __future__ import annotations

import ast

__all__ = ["NAMESPACE_READ", "NAMESPACE_REACHERS", "reaches_the_namespace"]

#: Read by a cell that may read any variable by name (`_cell_reads`).
NAMESPACE_READ = "<the namespace>"

#: Names through which code reaches the notebook's variables by name,
#: which no scan of the names it reads follows: ``globals()['bar']``,
#: ``sys.modules['__main__'].bar``, ``import __main__``, ``gc.get_objects()``.
NAMESPACE_REACHERS = frozenset(
    {
        "globals", "locals", "vars", "eval", "exec", "dir", "get_ipython", "__main__", "__name__", "sys",
        "__builtins__", "builtins", "__import__", "importlib", "gc", "inspect",
    }
)  # fmt: skip


def reaches_the_namespace(tree: ast.AST) -> bool:
    """Does *tree* name one of `NAMESPACE_REACHERS`, or import one?"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in NAMESPACE_REACHERS:
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.partition(".")[0] in NAMESPACE_REACHERS for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").partition(".")[0] in NAMESPACE_REACHERS:
            return True
    return False
