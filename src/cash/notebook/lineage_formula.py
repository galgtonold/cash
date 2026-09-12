"""The one formula for a variable's lineage, shared by both engines.

The runtime (``statement/lineage.py``) records a variable's lineage when its
statement runs; the upstream simulator (``upstream/virtual_lineage.py``)
recomputes it from the code above a cell. Wherever the two disagree, a cell
reading the variable sees "changed" with nothing changed and re-runs other
cells' statements to repair it. They used to build the lineage separately, and
in round 21 they had drifted apart: a name imported from a local module carried
that module's source hash at runtime and not in the simulation, so every cell
downstream of ``from helpers import clean`` re-ran statements even in a plain
top-to-bottom run -- among them a model refit, which made an "accuracy before"
report use the refitted model.

Everything here is a pure function of its arguments. The callers decide which
ingredients they have; this module decides how they combine.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import os
import sys
import types
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


def read_module_source_hash(mod_file: str, dep_files: set[str] | None = None) -> str | None:
    # Imported on use: the ``statement`` package imports this module.
    from .statement.file_deps import read_module_source_hash as read
    return read(mod_file, dep_files)


def output_lineage(
    source_hash: str,
    input_lineages: Iterable[str],
    file_component: str = "",
    func_component: str = "",
    module_component: str = "",
) -> str:
    """The lineage hash of one output of a statement."""
    lineage_str = (f"{source_hash}:{':'.join(sorted(input_lineages))}"
                   f"{file_component}{func_component}{module_component}")
    return hashlib.sha256(lineage_str.encode("utf-8")).hexdigest()


def callable_source_component(function_tracker: Any, inputs: set[str], user_ns: dict) -> str:
    """``:name:hash`` for every user-defined callable among *inputs*."""
    if function_tracker is None:
        return ""
    hashes = function_tracker.get_callable_source_hashes(inputs, user_ns)
    if not hashes:
        return ""
    return ":" + ":".join(f"{k}:{v}" for k, v in sorted(hashes.items()))


def _tracked(function_tracker: Any) -> set[str]:
    return getattr(function_tracker, "_tracked_modules", None) or set()


def _from_module_hash(module_name: str, function_tracker: Any) -> str:
    if module_name not in _tracked(function_tracker):
        return ""
    mod_obj = sys.modules.get(module_name)
    mod_file = getattr(mod_obj, "__file__", None) if mod_obj else None
    if not (mod_file and os.path.isfile(mod_file)):
        return ""
    digest = read_module_source_hash(mod_file)
    return f":from_mod_src:{digest}" if digest else ""


def module_source_component(
    function_tracker: Any,
    value: Any,
    var_name: str,
    code: str,
    tree: ast.Module | None = None,
    note_from_import: Callable[[str, str], None] | None = None,
) -> str:
    """The source hash of the local module *var_name* came from, or ``""``.

    Three cases: ``import X`` of a tracked module (its source plus its tracked
    dependency files); a callable from a tracked module
    (``from X import func``); a non-callable imported from one
    (``from X import CONST``, found through the import statement itself).
    *note_from_import* is told ``(var_name, module)`` whenever *var_name* comes
    from another module through a ``from`` import -- the runtime keeps that map
    for module invalidation.
    """
    if function_tracker is None:
        return ""
    if isinstance(value, types.ModuleType):
        mod_file = getattr(value, "__file__", None)
        if not (mod_file and os.path.isfile(mod_file) and var_name in _tracked(function_tracker)):
            return ""
        parents = getattr(function_tracker, "_dep_file_to_parents", None) or {}
        dep_files = {dep for dep, owners in parents.items() if var_name in owners}
        digest = read_module_source_hash(mod_file, dep_files)
        return f":mod_src:{digest}" if digest else ""

    if callable(value):
        obj_module = getattr(value, "__module__", None)
        if not (obj_module and obj_module in _tracked(function_tracker)):
            return ""
        if note_from_import is not None:
            note_from_import(var_name, obj_module)
        return _from_module_hash(obj_module, function_tracker)

    try:
        parsed = tree if tree is not None else ast.parse(code.strip())
    except SyntaxError:
        return ""
    for node in parsed.body:
        if not (isinstance(node, ast.ImportFrom) and node.module):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) == var_name:
                if note_from_import is not None:
                    note_from_import(var_name, node.module)
                return _from_module_hash(node.module, function_tracker)
    return ""
