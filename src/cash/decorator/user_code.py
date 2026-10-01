"""Which code counts as the user's: the predicates every key channel asks
before it keys a function, a class or a module, and cash's own wrapper,
which is never keyed."""

from __future__ import annotations

import functools
import sys
import types
from typing import Any

from ..install_paths import in_own_package, is_cash_path, is_user_code_module, top_package
from ..source_norm import unwrap_partials


def is_cash_wrapper(value: Any) -> bool:
    """Is *value* the wrapper ``@cash.cache`` returns, or a method bound
    from one? A mock answers every attribute, never with True."""
    try:
        return getattr(value, "_cash_cached", False) is True
    except Exception:  # noqa: BLE001 - an object's __getattr__ may raise anything
        return False


def cached_function_in(value: Any) -> Any:
    """The ``@cash.cache`` wrapper *value* is, or a partial wraps; else None."""
    value = unwrap_partials(value)
    return value if is_cash_wrapper(value) else None


def cash_wrapped(value: Any) -> Any:
    """The user function under a ``@cash.cache`` wrapper, else *value*.

    A wrapper's code is cash's, never part of a key: a code channel that
    meets one keys the function it wraps, and a data channel keys its state
    (`GlobalsFold.data_callable_identity`).
    """
    return getattr(value, "__wrapped__", value) if is_cash_wrapper(value) else value


def own_package(func: Any) -> str | None:
    """The top-level package of the module that defines *func*."""
    return top_package(getattr(func, "__module__", None))


def is_user_class(cls: Any, own_pkg: str | None = None) -> bool:
    """Is *cls* a class of the user's: inside *own_pkg*, or in a module
    `is_user_code_module` accepts (a notebook cell's class included)?

    Used to fold what a class holds (``ClassName.CONSTANT``, an instance's
    class): editing a class-level constant invalidates, while a library
    class's attributes do not churn the key.
    """
    name = getattr(cls, "__module__", None)
    if in_own_package(name, own_pkg):
        return True
    mod = sys.modules.get(name or "")
    return mod is not None and is_user_code_module(mod)


_HEAPTYPE = 1 << 9
_IMMUTABLETYPE = 1 << 8


#: Callables implemented in C: builtin functions and bound methods, and the
#: slot and method descriptors of C types.
_C_CALLABLES = (
    types.BuiltinFunctionType,
    types.MethodWrapperType,
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
)


def _is_c_type(cls: type) -> bool:
    """Was *cls* made by C code rather than a ``class`` statement? A static
    type, or a heap type an extension created immutable; Python cannot make
    either."""
    flags = getattr(cls, "__flags__", 0)
    return not flags & _HEAPTYPE or bool(flags & _IMMUTABLETYPE)


def _c_code_verdict(obj: Any) -> bool | None:
    """`is_user_code_object` for C code whose ``__module__`` cannot be
    followed; ``None`` when *obj* is not C code.

    A C type or callable is never the exec'd or notebook code that the
    fallback is for. ``sys.stdout.write`` has no ``__module__`` and
    ``_thread.lock`` is not reachable as ``_thread.lock``, and both counted as
    user code whose code could not be hashed: a logger, a queue or a stream
    reached through an argument warned about ``lock.acquire``. A bound method
    is judged by the class that defines it, so one of an extension built in
    the project still counts and ``write`` of your ``io.StringIO`` subclass
    does not.
    """
    if isinstance(obj, type):
        return False if _is_c_type(obj) else None
    if not isinstance(obj, _C_CALLABLES):
        return None
    owner = getattr(obj, "__self__", None)
    if owner is None:
        owner = getattr(obj, "__objclass__", None)
    if isinstance(owner, types.ModuleType):
        return is_user_code_module(owner)
    if owner is None:
        return False
    cls = owner if isinstance(owner, type) else type(owner)
    name = getattr(obj, "__name__", None)
    try:
        cls = next((klass for klass in cls.__mro__ if name in vars(klass)), cls)
    except Exception:  # noqa: BLE001 - an odd class must not break the verdict
        pass
    return is_user_code_object(cls)


def is_user_code_object(obj: Any) -> bool:
    """True when *obj* -- a class OR a function -- is defined in code the user
    plausibly edits. Both carry ``__module__``, so one predicate serves both.

    ``__module__`` alone is not trustworthy. A class or function built by
    ``exec(body, ns)`` where *ns* lacks a ``__name__`` key (a bare ``{}``,
    unlike a real notebook's globals, which start with ``__name__ ==
    '__main__'``) gets a fallback ``__module__`` from CPython's implicit
    ``__module__ = __name__`` lookup at definition time: ``None`` for a
    function, and -- because that lookup falls all the way through to the
    REAL ``builtins`` module's own ``__name__`` attribute -- literally
    ``'builtins'`` for a class. Neither reflects where the code actually
    lives. Confirm *obj* is actually reachable through the module it
    claims before trusting that module's verdict; otherwise this is the
    exec()/notebook case the predicate exists to catch, so it counts as
    user code (mirroring ``is_user_code_module``'s fileless-module
    handling) -- unless it is C code (`_c_code_verdict`).
    """
    code = getattr(obj, "__code__", None)
    if isinstance(code, types.CodeType) and is_cash_path(code.co_filename):
        # cash's own code -- the wrapper `@cash.cache` returns, whose
        # ``__module__`` and ``__qualname__`` are the user function's.
        return False
    mod_name = getattr(obj, "__module__", None)
    mod = sys.modules.get(mod_name) if mod_name else None
    if mod is None or not qualname_resolves_in(mod, obj):
        verdict = _c_code_verdict(obj)
        return True if verdict is None else verdict
    return is_user_code_module(mod)


def qualname_resolves_in(mod: Any, obj: Any) -> bool:
    """True if *obj* is actually reachable by walking its ``__qualname__``
    from *mod*, not merely claiming *mod* via ``__module__``.

    ``getattr(x, name, default)`` only swallows ``AttributeError`` -- a
    module implementing PEP 562 ``__getattr__`` (a real pattern for
    deprecation shims: raise a custom error for an old name instead of
    just returning it) can make this walk raise something else entirely,
    and ``is_user_code_object`` must never raise.
    """
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if not qualname:
        return False
    cur = mod
    try:
        for part in qualname.split("."):
            if part == "<locals>":
                return False  # nested in a function body - not module-reachable
            cur = getattr(cur, part, None)
            if cur is None:
                return False
        return cur is obj
    except Exception:  # noqa: BLE001 - a module __getattr__ may raise anything
        return False  # could not confirm reachability - do not trust it


def wraps_code(value: Any) -> bool:
    """Is *value* a descriptor around a function (classmethod, staticmethod,
    property, cached_property, partialmethod...)?"""
    if isinstance(value, (classmethod, staticmethod, property, functools.cached_property, functools.partialmethod)):
        return True
    return hasattr(type(value), "__get__") and any(
        callable(getattr(value, name, None)) for name in ("__func__", "fget", "func")
    )
