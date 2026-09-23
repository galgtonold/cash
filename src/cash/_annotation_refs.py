"""The classes and functions a class's or function's type annotations name.

An annotation is not always inert. Pydantic validates a field ``b: B`` by
running ``B``'s validators; anything built on ``typing.get_type_hints``
(cattrs, dacite, FastAPI, a builder of your own) constructs ``B`` from ``A``'s
hints. ``B`` is then named nowhere in the code that runs, and editing it served
a result from before the edit.

So what an annotation names is followed like code a body loads: a hint that
never runs costs a recompute when its class is edited, never a stale value.
Resolution is best effort and never raises: a string annotation is resolved by
name in the owner's module, generics are walked through ``typing.get_args``,
and anything that cannot be resolved is skipped.
"""

from __future__ import annotations

import re
import sys
import types
import typing
from typing import Any

_MAX_DEPTH = 8
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def _raw_annotations(owner: Any) -> dict:
    """*owner*'s own annotations, unevaluated where the runtime allows it."""
    try:
        import annotationlib  # Python 3.14+

        return dict(annotationlib.get_annotations(owner, format=annotationlib.Format.FORWARDREF))
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - a broken __annotate__ is not ours to raise
        return {}
    try:
        if isinstance(owner, type):
            found = owner.__dict__.get("__annotations__", {})
        else:
            found = getattr(owner, "__annotations__", {})
        return dict(found) if isinstance(found, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _module_namespace(owner: Any) -> dict:
    module = sys.modules.get(getattr(owner, "__module__", None) or "")
    return vars(module) if module is not None else (getattr(owner, "__globals__", None) or {})


def _resolve_string(text: str, namespace: dict) -> list[Any]:
    found = []
    for dotted in _DOTTED.findall(text):
        head, *rest = dotted.split(".")
        value = namespace.get(head)
        for attr in rest:
            if value is None:
                break
            value = getattr(value, attr, None)
        if value is not None:
            found.append(value)
    return found


def _walk(value: Any, namespace: dict, out: list, depth: int) -> None:
    if depth > _MAX_DEPTH or value is None:
        return
    if isinstance(value, str):
        for resolved in _resolve_string(value, namespace):
            _walk(resolved, namespace, out, depth + 1)
        return
    forward = getattr(value, "__forward_arg__", None)
    if isinstance(forward, str):
        _walk(forward, namespace, out, depth + 1)
        return
    if isinstance(value, (type, types.FunctionType)):
        out.append(value)
        # A generic alias of a user class (`Box[int]`) is not a type; its
        # origin is. Plain classes have no args to walk.
        return
    try:
        origin = typing.get_origin(value)
        args = typing.get_args(value)
    except Exception:  # noqa: BLE001
        return
    if origin is not None:
        _walk(origin, namespace, out, depth + 1)
    for arg in args:
        _walk(arg, namespace, out, depth + 1)
    # `Annotated[int, AfterValidator(check)]`: the validator's function runs.
    func = getattr(value, "func", None)
    if isinstance(func, types.FunctionType):
        out.append(func)


def annotation_referents(obj: Any, is_user: Any = None) -> list[Any]:
    """Classes and functions named by *obj*'s annotations (see module doc).

    For a class: every class in its MRO, the parameter and return
    annotations of the methods they define, and a pydantic model's field
    annotations. For a function: its parameter and return annotations.
    *is_user*, when given, limits the classes walked to user code: a library
    base (pydantic's ``BaseModel``) annotates hundreds of its own methods. What
    is returned is unfiltered; the caller decides what is user code.
    """
    out: list[Any] = []
    try:
        if isinstance(obj, type):
            owners = []
            for base in obj.__mro__:
                if base is object or (is_user is not None and not is_user(base)):
                    continue
                owners.append(base)
                for member in vars(base).values():
                    if isinstance(member, property):
                        owners.extend(a for a in (member.fget, member.fset) if a is not None)
                        continue
                    member = getattr(member, "__func__", member)
                    if isinstance(member, types.FunctionType):
                        owners.append(getattr(member, "__wrapped__", member))
        else:
            obj = getattr(obj, "__func__", obj)
            owners = [getattr(obj, "__wrapped__", obj)]
        for owner in owners:
            namespace = _module_namespace(owner)
            for value in _raw_annotations(owner).values():
                _walk(value, namespace, out, 0)
        if isinstance(obj, type):
            fields = getattr(obj, "model_fields", None)
            if isinstance(fields, dict):
                namespace = _module_namespace(obj)
                for info in fields.values():
                    _walk(getattr(info, "annotation", None), namespace, out, 0)
                    for meta in getattr(info, "metadata", ()) or ():
                        _walk(meta, namespace, out, 0)
    except Exception:  # noqa: BLE001 - following annotations must never break a call
        return out
    return out
