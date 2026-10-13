"""Code a notebook installs on a library class, in the key of a statement
that reaches it by attribute.

``@pd.api.extensions.register_dataframe_accessor("fx")`` on a class defined
in a cell, or ``pd.DataFrame.slow = _slow``: ``res = df.fx.slow()`` names
neither the class nor the function, so editing the accessor and running its
cell again served the old result. The decorator asks the same classes
(``code_args.loaded_accessor_classes``) for the names a cached function
reads; a statement asks them for the attribute names it reads, and those the
functions it calls read.

What is found is keyed by its code (`callable_identity` of each function, of
each method of a class) and the lineage of each global that code reads, so
the key moves with an edit to it or to a global it reads. A statement that
reaches no installed code keeps its key byte for byte.
"""

from __future__ import annotations

import ast
import builtins
import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash._memo import CODE_OBJECTS, NOTEBOOK_STATEMENTS, LruMemo
from cash.code_digest import callable_identity
from cash.decorator.closure_fold import iter_code_scopes
from cash.decorator.code_args import installed_user_code, loaded_accessor_classes
from cash.loaded_code import class_functions

__all__ = ["installed_code_parts"]

_MISSING = object()

#: Statement text -> the attribute names it reads.
_STATEMENT_ATTRIBUTES: LruMemo[str, frozenset[str]] = LruMemo(NOTEBOOK_STATEMENTS)
#: id(class attribute) -> (the attribute, the user code it stands for). A
#: plain dict: asked once per name a statement reads, and bounded by the
#: attributes the accessor libraries' classes have (dropped when it grows
#: past `CODE_OBJECTS`).
_USER_CODE: dict[int, tuple[Any, tuple[Any, ...]]] = {}


def _statement_attributes(code: str) -> frozenset[str]:
    found = _STATEMENT_ATTRIBUTES.get(code)
    if found is None:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            found = frozenset()
        else:
            found = frozenset(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        _STATEMENT_ATTRIBUTES[code] = found
    return found


def _code_names(names: Iterable[str], user_ns: Mapping[str, Any]) -> set[str]:
    """Every name the code of the functions *names* bind reads."""
    found: set[str] = set()
    for name in names:
        fn = user_ns.get(name)
        if isinstance(fn, types.FunctionType):
            for scope in iter_code_scopes(fn.__code__):
                found.update(scope.co_names)
    return found


def _user_code(attr: Any) -> tuple[Any, ...]:
    entry = _USER_CODE.get(id(attr))
    if entry is not None and entry[0] is attr:
        return entry[1]
    found = tuple(installed_user_code(attr))
    if len(_USER_CODE) >= CODE_OBJECTS:
        _USER_CODE.clear()
    _USER_CODE[id(attr)] = (attr, found)
    return found


def _functions(carrier: Any) -> list[types.FunctionType]:
    if isinstance(carrier, type):
        return class_functions(carrier)
    return [carrier] if isinstance(carrier, types.FunctionType) else []


def installed_code_parts(
    code: str,
    inputs: Iterable[str],
    user_ns: Mapping[str, Any],
    variable_lineage: Mapping[str, str],
) -> list[str]:
    """``"installed:<name>:<identity>"`` for the user code a loaded library
    class carries under an attribute name that *code* reads, or the functions
    it calls (*inputs*, and the globals those reach), and
    ``"installed-reads:<global>:<lineage>"`` for each global that code reads
    (and those the functions it calls read). Empty when no accessor library
    is loaded or nothing is found."""
    family = loaded_accessor_classes()
    if not family:
        return []
    from .cache_key import called_function_dependencies, called_function_globals

    inputs = set(inputs)
    names = set(_statement_attributes(code))
    if any(isinstance(user_ns.get(name), types.FunctionType) for name in inputs):
        names |= _code_names(inputs, user_ns)
        try:
            names |= _code_names(called_function_globals(inputs, user_ns, variable_lineage=variable_lineage), user_ns)
        except (TypeError, ValueError, AttributeError, RecursionError):
            pass
    parts: set[str] = set()
    reads: set[str] = set()
    for name in sorted(names):
        for cls in family:
            for klass in cls.__mro__:
                attr = klass.__dict__.get(name, _MISSING)
                if attr is _MISSING:
                    continue
                for carrier in _user_code(attr):
                    functions = _functions(carrier)
                    identity = ",".join(sorted(callable_identity(fn) for fn in functions)) or callable_identity(carrier)
                    parts.add(f"installed:{name}:{getattr(carrier, '__qualname__', '?')}:{identity}")
                    for fn in functions:
                        for scope in iter_code_scopes(fn.__code__):
                            reads.update(scope.co_names)
                break
    if not parts:
        return []
    refs = sorted(
        ref
        for ref in reads
        if not hasattr(builtins, ref) and ref in user_ns and not isinstance(user_ns.get(ref), types.ModuleType)
    )
    parts.update(f"installed-reads:{ref}:{variable_lineage.get(ref, 'ABSENT')}" for ref in refs)
    parts.update(f"installed-reads:{dep}" for dep in called_function_dependencies(refs, user_ns, variable_lineage))
    return sorted(parts)
