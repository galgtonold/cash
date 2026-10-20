"""The code of the user's modules whose classes' objects a statement's inputs
hold, for its key.

``orders = pickle.load(f)`` holding objects of ``mylib.Order``, then
``t = [o.total() for o in orders]``: the statement names neither ``mylib`` nor
``Order``, and the lineage of ``orders`` is the pickle file's, so editing
``Order.total`` or the module's ``rate()`` that it calls (and reloading the
module, by cash's own check or ``%autoreload``) left the key where it was and
served the old result. The walk that finds the notebook functions a value
holds (``held_functions``) notes the classes of the user's modules it meets;
this names what they run, twice over:

- what the classes reach inside their module's file
  (``module_symbols.closure_digest``, the whole file when that cannot be
  bounded) and the files of the user's modules it imports, whole: what the
  source says, constants included;
- the code the module and those it imports have loaded
  (``source_norm.bytecode_identity`` of each function and method): what
  runs. A module edited since it was loaded and not reloaded (a module only
  a pickle imported is not tracked, so not reloaded), or loaded through an
  import hook that rewrites it, runs other code than its file says.

A module whose file cannot be read is named in the second part of the
answer: the statement runs uncached.

What the code reads is not all in the files: a cell setting ``mylib.K = 7``
(or ``mylib.Order.scale = 7``, ``mylib.TABLE["k"] = 7``) changes what
``total`` returns and leaves them as they were. `held_data_part` digests
the module data the classes' methods read, as the key of a statement naming
the module digests what it reads (``lineage_formula.module_data_component``).
"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or

from .._memo import CODE_OBJECTS, LruMemo
from ..code_digest import module_identity
from ..loaded_code import class_functions
from ..source_norm import bytecode_identity
from ..tracking.function_tracker import imports_of, is_local_module
from ..tracking.module_symbols import closure_digest
from .callee_reach import class_data
from .lineage_formula import module_data_digest

__all__ = ["held_data_part", "held_module_parts"]

#: Code object -> its `bytecode_identity`: a module's functions are digested
#: for every statement reading an object of one of its classes.
_IDENTITY: LruMemo[types.CodeType, str | None] = LruMemo(CODE_OBJECTS)


def held_module_parts(classes: Iterable[type]) -> tuple[list[str], set[str]]:
    """``"heldmod:<module>:<digest>"`` for each of the user's modules that
    defines one of *classes*, sorted, and the names of those modules whose
    code cannot be named (see the module docstring)."""
    by_module: dict[str, set[str]] = {}
    for cls in classes:
        name = attribute_or(cls, "__module__", None)
        if isinstance(name, str):
            by_module.setdefault(name, set()).add(attribute_or(cls, "__qualname__", "") or "")
    parts: list[str] = []
    unkeyed: set[str] = set()
    for name in sorted(by_module):
        digest = _module_digest(name, by_module[name])
        if digest is None:
            unkeyed.add(name)
        else:
            parts.append(f"heldmod:{name}:{digest}")
    return parts, unkeyed


def held_data_part(classes: Iterable[type], namespace: Mapping[str, Any]) -> str | None:
    """``"helddata:<digest>"`` of the local modules' data the methods of
    *classes* read (`callee_reach.class_data`), each piece digested as
    `module_data_digest` digests it: None when they read none."""
    data = class_data(classes, namespace)
    if not data:
        return None
    joined = ":".join(f"{label}={module_data_digest(label, value)}" for label, value in data)
    return "helddata:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _module_digest(name: str, qualnames: set[str]) -> str | None:
    module = sys.modules.get(name)
    path = attribute_or(module, "__file__", None)
    if module is None or not path or not os.path.isfile(path):
        return None
    roots = {qualname.split(".", 1)[0] for qualname in qualnames}
    own = None
    if all(qualname and "<locals>" not in qualname for qualname in qualnames):
        own = closure_digest(path, roots)
    if own is None:
        own = module_identity(path)
        if own is None:
            return None
        own = "whole:" + own
    hasher = hashlib.sha256(own.encode("utf-8"))
    for dep in sorted(_local_imports(name), key=lambda m: m.__name__):
        dep_digest = module_identity(dep.__file__)
        if dep_digest is None:
            return None
        hasher.update(f":{dep.__name__}={dep_digest}".encode("utf-8"))
        hasher.update(f":loaded={_loaded_identity(dep)}".encode("utf-8"))
    hasher.update(f":loaded={_loaded_identity(module)}".encode("utf-8"))
    return hasher.hexdigest()


def _local_imports(name: str) -> list[types.ModuleType]:
    """The user's modules *name*'s file imports, transitively."""
    found: dict[str, types.ModuleType] = {}
    pending = [name]
    while pending:
        imported = imports_of(pending.pop()) or frozenset()
        for other in sorted(imported):
            module = sys.modules.get(other)
            if other == name or other in found or module is None or not _local(module):
                continue
            found[other] = module
            pending.append(other)
    return list(found.values())


def _local(module: types.ModuleType) -> bool:
    try:
        return is_local_module(module)
    except (TypeError, AttributeError):
        return False


def _loaded_identity(module: types.ModuleType) -> str:
    """A digest of the code of every function and method *module* defines
    at its top, as loaded."""
    name = module.__name__
    parts: list[str] = []
    for attr, value in list(vars(module).items()):
        if isinstance(value, type) and attribute_or(value, "__module__", None) == name:
            functions = class_functions(value)
        elif isinstance(value, types.FunctionType) and value.__module__ == name:
            functions = [value]
        else:
            continue
        for fn in functions:
            parts.append(f"{attr}.{fn.__qualname__}={_identity(fn)}")
    return hashlib.sha256("|".join(sorted(parts)).encode("utf-8")).hexdigest()


def _identity(fn: types.FunctionType) -> str | None:
    code = fn.__code__
    try:
        found = _IDENTITY.get(code)
    except TypeError:
        return bytecode_identity(fn)
    if found is None:
        found = bytecode_identity(fn)
        _IDENTITY[code] = found
    return found
