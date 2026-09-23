"""AST helpers shared by the decorator's analyzer, the notebook's
cacheability scan and the upstream simulation.

:func:`resolve_callee` names the object a call expression refers to without
running the user's code: an attribute is read with ``getattr`` only on a
module; on anything else it is looked up statically, or not at all.
:func:`called_names` lists the bare names a tree calls. :func:`parse_cached`
is the one bounded parse memo for statement and cell text.
"""

from __future__ import annotations

import ast
import builtins
import functools
import inspect
import sys
import types
from collections.abc import Mapping
from typing import Any, Literal

__all__ = ["CallScope", "called_names", "parse_cached", "resolve_callee"]

#: Descriptors implemented in C that bind a method and run nothing else.
_C_METHOD_DESCRIPTORS = (
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
    types.BuiltinFunctionType,
)


@functools.lru_cache(maxsize=1024)
def parse_cached(code: str) -> ast.Module | None:
    """``ast.parse(code)``, memoised; None when *code* does not parse.

    The tree is shared between callers: read it, never change it.
    """
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):  # ValueError: a null byte in the source
        return None


def resolve_callee(
    func: ast.AST,
    namespace: Mapping[str, Any],
    *,
    modules_only: bool = True,
    builtins_fallback: bool = False,
) -> Any | None:
    """The object a call's callee expression names, or None.

    ``name`` is looked up in *namespace* (then in ``builtins`` when
    *builtins_fallback*); ``a.b.c`` resolves ``a`` the same way and reads each
    attribute in turn. Anything else -- a subscript, a call, a literal -- is
    None: there is nothing to look up without evaluating it.

    An attribute of a MODULE is read with ``getattr``, and so is one of a
    ``unittest.mock`` object standing in for a module (``mock.patch.object(app,
    "json", MagicMock())``): the caller must see the mock to refuse caching
    its answer, and reading it runs nothing of the user's. With *modules_only*,
    that is the only attribute followed. Without it, an attribute of any other
    object is looked up with ``inspect.getattr_static``, which never runs a
    property, a descriptor or ``__getattr__``: a function found on the class
    comes back bound to the object, a ``staticmethod`` or ``classmethod``
    unwrapped or bound as Python would, a C method bound (which runs nothing
    else), and a value stored on the object as itself. A property or any
    other descriptor is None.
    """
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    if node.id in namespace:
        obj = namespace[node.id]
    elif builtins_fallback and hasattr(builtins, node.id):
        obj = getattr(builtins, node.id)
    else:
        return None
    for attr in reversed(parts):
        if obj is None:
            return None
        if isinstance(obj, types.ModuleType) or _is_mock(obj):
            obj = getattr(obj, attr, None)
        elif modules_only:
            return None
        else:
            obj = _static_attribute(obj, attr)
    return obj


def _static_attribute(obj: Any, attr: str) -> Any | None:
    """``obj.attr`` as Python would bind it, when that runs no user code."""
    try:
        raw = inspect.getattr_static(obj, attr)
    except AttributeError:
        return None
    is_class = isinstance(obj, type)
    if isinstance(raw, staticmethod):
        return raw.__func__
    if isinstance(raw, classmethod):
        return types.MethodType(raw.__func__, obj if is_class else type(obj))
    if isinstance(raw, types.FunctionType):
        if is_class or _in_instance_dict(obj, attr):
            return raw
        return types.MethodType(raw, obj)
    if isinstance(raw, _C_METHOD_DESCRIPTORS):
        return getattr(obj, attr, None)
    if hasattr(type(raw), "__get__") and not _in_instance_dict(obj, attr):
        return None  # a property or another descriptor: reading it runs code
    return raw


def _in_instance_dict(obj: Any, attr: str) -> bool:
    """Was *attr* found in the object's own ``__dict__``, where Python does not bind?"""
    try:
        own = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return False
    return isinstance(own, dict) and attr in own


def _is_mock(obj: Any) -> bool:
    """A ``unittest.mock`` object; never imports ``unittest.mock`` itself."""
    module = sys.modules.get("unittest.mock")
    return module is not None and isinstance(obj, module.NonCallableMock)


#: Which calls :func:`called_names` reports.
#:
#: * ``"all"`` -- every call in the tree.
#: * ``"top_level"`` -- skips the tree's own ``def`` / ``class`` statements (but
#:   not a function defined inside an ``if``), the same scope as the
#:   top-level mutation visitor in ``cacheability.analyze_statement``.
#: * ``"eager"`` -- skips every function, class and lambda body: only the calls
#:   that happen while the tree itself runs.
#: * ``"no_control_bodies"`` -- skips the bodies of ``for`` / ``while`` / ``if`` /
#:   ``with`` / ``try`` (their headers still count). A control structure's body
#:   belongs to the control-structure handler, not to the statement around it.
CallScope = Literal["all", "top_level", "eager", "no_control_bodies"]

_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_DEFERRED = (*_DEFINITIONS, ast.Lambda)
_CONTROL_STATEMENTS = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try)
_CONTROL_BODY_FIELDS = frozenset({"body", "orelse", "finalbody", "handlers"})


def called_names(tree: ast.AST | None, scope: CallScope = "all") -> frozenset[str]:
    """The bare names called as ``name(...)`` in *tree*, within *scope*."""
    if tree is None:
        return frozenset()
    if scope == "all":
        return frozenset(n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))
    out: set[str] = set()

    def visit(node: ast.AST, top: bool) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out.add(node.func.id)
        for field, value in ast.iter_fields(node):
            if scope == "no_control_bodies" and isinstance(node, _CONTROL_STATEMENTS) and field in _CONTROL_BODY_FIELDS:
                continue
            for child in value if isinstance(value, list) else (value,):
                if not isinstance(child, ast.AST):
                    continue
                if scope == "top_level" and top and isinstance(child, _DEFINITIONS):
                    continue
                if scope == "eager" and isinstance(child, _DEFERRED):
                    continue
                visit(child, False)

    visit(tree, True)
    return frozenset(out)
