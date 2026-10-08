"""The user code a piece of module data holds: functions and objects of the
user's classes kept in a registry dict, a handler set with a setter.

``mylib.REG['alpha'] = alpha`` in a cell, ``x = mylib.call_all()`` below:
the value hash of ``REG`` pickles ``alpha`` by its name, so editing
``alpha``'s body left the reader's key where it was and served the old
result. `held_code_digest` names that code by its `callable_identity`, the
digest the decorator keys a function's code on, and an object of a user's
class by its class's code (its attributes are in the value hash).
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import operator
import sys
import types
from typing import Any

from ..code_digest import callable_identity
from ..install_paths import is_user_code_file, is_user_code_module
from ..value_types import LEAF_TYPES

__all__ = ["held_code_digest"]

#: Types that hold no code and lead to none.
_LEAVES = frozenset(LEAF_TYPES)

#: How many objects one walk visits before it stops. Real module data holds
#: far fewer that are not plain values; a walk cut short says so in the
#: digest, the same way every time, rather than run on.
_WALK_LIMIT = 1_000_000

_CONTAINERS = (list, tuple, set, frozenset)
_PLAIN_CONTAINERS = frozenset((dict, list, tuple, set, frozenset))
_CODE_HOLDERS = frozenset((types.FunctionType, types.MethodType, functools.partial, staticmethod, classmethod))


def held_code_digest(value: Any) -> str:
    """A digest of the user code *value* holds, at any depth; empty when it
    holds none, so plain data keeps the digest it had.

    Followed: the items of a list, tuple, set or dict, the attributes of an
    object of a user's class, a bound method's function and object, a
    partial's function and arguments, a function's defaults and closure.
    An object of a library class is not entered: its code changes only with
    the library. The digest is over the sorted parts, so it does not depend
    on a set's order; which name holds which function is in the value hash.
    """
    parts: list[str] = []
    classes: dict[type, str | None] = {}
    seen: set[int] = {id(value)}
    level: list[Any] = [value]
    visited = 0
    while level:
        visited += len(level)
        if visited > _WALK_LIMIT:
            parts.append("walk-limit")
            break
        children: list[Any] = []
        # One step per TYPE in the level, not per object: the items of a
        # list of 100,000 records are gathered at C speed.
        kinds = set(map(type, level))
        for kind in kinds:
            group = level if len(kinds) == 1 else [obj for obj in level if type(obj) is kind]
            _visit_group(kind, group, parts, classes, children)
        if not children or set(map(type, children)) <= _LEAVES:
            break
        # Still at C speed: the children that are not plain values, by id,
        # less those met before (a cycle, an object held twice).
        others = list(itertools.compress(children, map(operator.not_, map(_LEAVES.__contains__, map(type, children)))))
        found = dict(zip(map(id, others), others))
        for key in seen & found.keys():
            del found[key]
        seen.update(found)
        level = list(found.values())
    if not parts:
        return ""
    return hashlib.sha256("|".join(sorted(parts)).encode("utf-8")).hexdigest()


def _visit_group(
    kind: type, group: list[Any], parts: list[str], classes: dict[type, str | None], children: list[Any]
) -> None:
    """Note the code the objects of type *kind* in *group* hold, and put
    what they lead to in *children*."""
    if issubclass(kind, dict):
        # Keys are nearly always strings: checked apart, at C speed, they
        # add nothing to the walk.
        if not set(map(type, itertools.chain.from_iterable(group))) <= _LEAVES:
            children.extend(itertools.chain.from_iterable(group))
        children.extend(itertools.chain.from_iterable(map(dict.values, group)))
    elif issubclass(kind, _CONTAINERS):
        children.extend(itertools.chain.from_iterable(group))
    elif issubclass(kind, types.ModuleType):
        return
    elif kind in _CODE_HOLDERS or issubclass(kind, type):
        for obj in group:
            _visit(obj, parts, classes, children)
        return
    if kind in _PLAIN_CONTAINERS:
        return
    # An object of a user's class, a dict or list subclass of theirs included.
    code = _class_code(kind, classes)
    if code is None:
        return
    parts.append(f"obj:{code}")
    for obj in group:
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            children.extend(attrs.values())


def _visit(obj: Any, parts: list[str], classes: dict[type, str | None], children: list[Any]) -> None:
    if isinstance(obj, types.FunctionType):
        if _user_function(obj):
            parts.append(f"fn:{callable_identity(obj)}")
            children.extend(obj.__defaults__ or ())
            children.extend((obj.__kwdefaults__ or {}).values())
            for cell in obj.__closure__ or ():
                try:
                    children.append(cell.cell_contents)
                except ValueError:  # an empty cell
                    pass
    elif isinstance(obj, types.MethodType):
        children.extend((obj.__func__, obj.__self__))
    elif isinstance(obj, functools.partial):
        children.append(obj.func)
        children.extend(obj.args)
        children.extend(obj.keywords.values())
    elif isinstance(obj, (staticmethod, classmethod)):
        children.append(obj.__func__)
    elif isinstance(obj, type):
        code = _class_code(obj, classes)
        if code is not None:
            parts.append(f"cls:{code}")


def _user_function(fn: types.FunctionType) -> bool:
    return is_user_code_file(getattr(fn.__code__, "co_filename", None))


def _user_class(cls: type) -> bool:
    """Is *cls* the user's: from a module of theirs (a notebook's
    ``__main__`` included), or, with its module not loaded, with methods
    written in their code?"""
    module = sys.modules.get(getattr(cls, "__module__", None) or "")
    if module is not None:
        return is_user_code_module(module)
    return any(_user_function(fn) for member in vars(cls).values() for fn in _member_functions(member))


def _class_code(cls: type, classes: dict[type, str | None]) -> str | None:
    """The code of the user's classes in *cls*'s MRO, or None when *cls* is
    not a class of the user's. Memoised for one walk only: a method can be
    assigned onto a class without making a new class object."""
    if cls in classes:
        return classes[cls]
    code: str | None = None
    if _user_class(cls):
        parts = []
        for klass in cls.__mro__:
            if klass is object or not _user_class(klass):
                continue
            for name, member in sorted(vars(klass).items(), key=lambda item: item[0]):
                for fn in _member_functions(member):
                    if _user_function(fn):
                        parts.append(f"{klass.__qualname__}.{name}:{callable_identity(fn)}")
        code = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    classes[cls] = code
    return code


def _member_functions(member: Any) -> list[types.FunctionType]:
    """The functions a class attribute runs: a method, a static or class
    method, a property's accessors."""
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        return [f for f in (member.fget, member.fset, member.fdel) if isinstance(f, types.FunctionType)]
    return [member] if isinstance(member, types.FunctionType) else []
