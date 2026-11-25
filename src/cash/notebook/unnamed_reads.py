"""Whether a cell may read a notebook variable it does not name.

The upstream check simulates a cell from the variables its code names. A
cell that reaches the namespace another way (``globals()[name]``,
``__main__.n``, a frame's globals, a function that looks a name up by a
string) reads variables that simulation never sees, so its check cannot
clear it.
"""

from __future__ import annotations

import ast
import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or, instance_of

from .cache_key import unkeyed_reach

__all__ = ["cell_reads_names_unseen"]

#: Names whose mere mention in a cell reaches the notebook's namespace
#: without naming the variable read (``__main__.n``, ``sys.modules``,
#: a frame's globals).
_NAMESPACE_MARKERS = (
    "globals",
    "locals",
    "vars",
    "eval",
    "exec",
    "__main__",
    "sys.modules",
    "modules",
    "_getframe",
    "f_globals",
    "f_locals",
    "f_back",
    "currentframe",
    "__dict__",
    "user_ns",
    "get_ipython",
)


def cell_reads_names_unseen(
    cell_code: str,
    inputs: Iterable[str],
    user_ns: Mapping[str, Any],
    variable_lineage: Mapping[str, str],
) -> bool:
    """Whether the cell *cell_code*, reading *inputs*, may read a notebook
    variable it does not name: its text mentions a way into the namespace
    (`_NAMESPACE_MARKERS`), or it or a function of the user's it reaches
    looks a name up by a string, or reaches a value too big to follow
    (`cache_key.unkeyed_reach`, as ``KEY-UNTRACKABLE-DEP`` finds it). True
    when it does not parse. False means every variable it reads is among
    *inputs* or what their functions name."""
    try:
        tree = ast.parse(cell_code)
    except (SyntaxError, ValueError):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _NAMESPACE_MARKERS:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _NAMESPACE_MARKERS:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.name in ("__main__", "sys", "inspect") for alias in node.names
        ):
            return True
    for name in inputs:
        # A name bound to one of those (``g = globals`` above).
        value = user_ns.get(name)
        if value is globals or value is vars or value is eval or value is exec or value is locals:
            return True
        if instance_of(value, types.ModuleType) and attribute_or(value, "__name__", None) in (
            "__main__",
            "sys",
            "inspect",
        ):
            return True
    for node in tree.body:
        try:
            code = ast.unparse(node)
        except (ValueError, TypeError, RecursionError):
            return True
        reach = unkeyed_reach(code, inputs, user_ns, variable_lineage)
        if reach.by_string or reach.unfollowed:
            return True
    return False
