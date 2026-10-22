"""Which earlier statements set something on a module a re-run ``from M
import name`` binds from (`ReexecutionPlanner._complete_module_setting_writers`).

A fresh import binds what the module's file says. A statement above that
set ``helpers.CFG = 5`` -- by the module's name, an alias
(``import helpers as h``), a submodule (``from pkg import sub``) or an object
taken from it (``from helpers import TABLE``; ``TABLE['k'] = 5``) -- must run
again before it, unless it already ran in this kernel. Only a setting the
imported name can see counts: the name itself or what is inside it, or for
a function or a value built from one, a module global its code reads.
"""

from __future__ import annotations

import ast
import importlib.machinery
import os
import sys
from typing import Any

from ...analysis.ast_util import parse_cached

#: A method call that changes the object it is called on.
_CHANGING_METHODS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "update",
        "setdefault",
        "pop",
        "popitem",
        "remove",
        "clear",
        "add",
        "discard",
        "sort",
        "reverse",
        "set",
    }
)


def _tree(code: str) -> ast.Module | None:
    try:
        return parse_cached(code)
    except (SyntaxError, ValueError):
        return None


def imported_from(code: str) -> list[tuple[str, str]]:
    """``(module, name)`` for each name a ``from module import name`` in
    *code* binds (absolute imports only; ``*`` is left out)."""
    tree = _tree(code)
    if tree is None:
        return []
    return [
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and not node.level
        for alias in node.names
        if alias.name != "*"
    ]


def module_bindings(code: str) -> dict[str, str]:
    """The names an import in *code* binds, each to the dotted path of what
    it holds: ``import helpers as h`` binds ``h`` to ``helpers``, ``import
    pkg.sub`` binds ``pkg`` to ``pkg``, ``from pkg import sub as s`` binds
    ``s`` to ``pkg.sub`` (a submodule or an object of ``pkg``)."""
    tree = _tree(code)
    if tree is None:
        return {}
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    found[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if alias.name != "*":
                    found[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return found


def _path(node: ast.AST, bindings: dict[str, str]) -> str | None:
    """The dotted path an attribute or subscript chain ending at *node*
    reaches through a bound name, up to its first subscript: ``h.SCALE.k``
    is ``helpers.SCALE.k``, ``TABLE['k']`` is ``helpers.TABLE``."""
    attrs: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            attrs.clear()  # what a subscript holds is inside what holds it
            node = node.value
        elif isinstance(node, ast.Name):
            base = bindings.get(node.id)
            return None if base is None else ".".join([base, *reversed(attrs)])
        else:
            return None


def written_paths(code: str, bindings: dict[str, str]) -> set[str] | None:
    """The dotted paths *code* sets or changes through a name in
    *bindings*: an attribute or item it assigns, augments or deletes, or an
    object it calls a changing method of (``TABLE.update(...)``). None when
    it may write through one in a way not read here (``setattr(h, ...)``,
    ``vars(h)[...]``, a function it calls): every setting counts then."""
    tree = _tree(code)
    if tree is None:
        return None
    paths: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _CHANGING_METHODS:
                found = _path(node.func.value, bindings)
                if found is not None:
                    paths.add(found)
            continue
        for target in targets:
            for part in ast.walk(target):
                if isinstance(part, (ast.Attribute, ast.Subscript)) and isinstance(part.ctx, (ast.Store, ast.Del)):
                    found = _path(part, bindings)
                    if found is not None:
                        paths.add(found)
    return paths or None


def _source_of(module: str) -> ast.Module | None:
    """The parsed source of *module*, found the way an import finds it but
    without running it; None when it cannot be read."""
    loaded = sys.modules.get(module)
    path = getattr(loaded, "__file__", None) if loaded is not None else None
    if path is None:
        locations = None
        spec = None
        for i, part in enumerate(module.split(".")):
            try:
                spec = importlib.machinery.PathFinder.find_spec(part, locations)
            except (ImportError, ValueError):
                return None
            if spec is None:
                return None
            locations = spec.submodule_search_locations
            if i < module.count(".") and locations is None:
                return None
        path = spec.origin if spec is not None else None
    if not isinstance(path, str) or not path.endswith(".py") or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return ast.parse(handle.read())
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return None


def _defined(tree: ast.Module) -> dict[str, list[ast.AST]]:
    """What binds each top-level name of a module: its ``def``, ``class``
    or assignments."""
    found: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.Name):
                        found.setdefault(part.id, []).append(node)
    return found


def names_read_by(module: str, name: str) -> frozenset[str] | None:
    """The module globals what *module* binds to *name* can read: its own
    name, and the names its ``def``, ``class`` or assignment reads, through
    the module's other top-level names they read in turn. None when the
    module's source cannot be read or does not bind *name* plainly."""
    tree = _source_of(module)
    if tree is None:
        return None
    defined = _defined(tree)
    if name not in defined:
        return None
    seen = {name}
    pending = [name]
    while pending:
        for node in defined.get(pending.pop(), ()):
            for part in ast.walk(node):
                if isinstance(part, ast.Name) and part.id not in seen:
                    seen.add(part.id)
                    if part.id in defined:
                        pending.append(part.id)
    return frozenset(seen)


def sets_what_is_imported(paths: set[str] | None, imported: list[tuple[str, str]]) -> bool:
    """Can a setting of *paths* (None: not known) change what one of
    *imported* binds, or what it reads when it is called?"""
    if paths is None:
        return True
    for module, name in imported:
        target = f"{module}.{name}"
        reads: Any = False  # read on first need
        for path in paths:
            if path == target or path.startswith(target + ".") or target.startswith(path + "."):
                return True
            if not path.startswith(module + "."):
                continue
            if reads is False:
                reads = names_read_by(module, name)
            if reads is None or path[len(module) + 1 :].split(".")[0] in reads:
                return True
    return False
