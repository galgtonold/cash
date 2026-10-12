"""Functions a JIT compiler wraps, seen as the Python function they compile.

numba's ``@njit``, ``@jit``, ``@vectorize``, ``@guvectorize`` and ``@cfunc``
return an object (a ``CPUDispatcher``, a ``DUFunc``, a ``GUFunc``, a
``CFunc``) that runs machine code compiled from one Python function. To the
key that object is code: the function it compiles (`jit_python_function`),
walked like any helper, plus the options it was compiled with
(`jit_options`), which change what the machine code computes (``fastmath``,
``error_model``, ``parallel``). Its own attributes -- type contexts, overload
tables, the on-disk cache -- are the compiler's state, not data a result
depends on.

Detected by shape, never by importing numba: numba stays optional, and a
process that never imported it pays one type lookup per callable.
"""

from __future__ import annotations

import hashlib
import types
from typing import Any

__all__ = ["jit_identity", "jit_options", "jit_python_function", "is_jit_type"]


#: Options that change where compiled code is kept or how it is found, not
#: what it computes.
_NOT_RESULT_OPTIONS = frozenset({"cache", "no_cfunc_wrapper", "_dbg_extend_lifetimes", "_dbg_optnone"})


def is_jit_type(t: type) -> bool:
    """Is *t* a class numba defines (its dispatchers, ufunc builders, cfuncs)?"""
    module = getattr(t, "__module__", None)
    return isinstance(module, str) and (module == "numba" or module.startswith("numba."))


def _own_dict(obj: Any) -> dict | None:
    """*obj*'s instance ``__dict__``, read without running its ``__getattr__``."""
    try:
        state = object.__getattribute__(obj, "__dict__")
    except Exception:  # noqa: BLE001 - no dict, or one that refuses to be read
        return None
    return state if isinstance(state, dict) else None


def jit_python_function(obj: Any) -> types.FunctionType | None:
    """The Python function the JIT-compiled callable *obj* compiles, or None
    when *obj* is not one.

    A numba dispatcher has it as ``py_func`` beside its ``targetoptions``;
    any object holding both is taken to be one. numba's other wrappers
    (``DUFunc``, ``GUFunc``, ``CFunc``) keep it as ``_pyfunc`` or, through
    ``functools.update_wrapper``, as ``__wrapped__``.
    """
    if isinstance(obj, (types.FunctionType, type, types.ModuleType, types.MethodType)):
        return None
    state = _own_dict(obj)
    if state is None:
        return None
    fn = state.get("py_func")
    if isinstance(fn, types.FunctionType) and "targetoptions" in state:
        return fn
    if not is_jit_type(type(obj)):
        return None
    for name in ("py_func", "_pyfunc", "__wrapped__"):
        fn = state.get(name)
        if isinstance(fn, types.FunctionType):
            return fn
    return None


def _attr(obj: Any, name: str) -> Any:
    """``getattr(obj, name, None)``, None for a None *obj* or a failing read."""
    if obj is None:
        return None
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - a property that raises
        return None


def _stable(value: Any) -> str:
    """A repr of an option value that is the same in every process."""
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(f"{_stable(k)}:{_stable(v)}" for k, v in sorted(value.items(), key=lambda kv: repr(kv[0])))
            + "}"
        )
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_stable(v) for v in value)) + "}"
    if isinstance(value, (list, tuple)):
        return "(" + ",".join(_stable(v) for v in value) + ")"
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return repr(value)
    if isinstance(value, (type, types.FunctionType)):
        return f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', '?')}"
    # A numba type (``float64``, ``array(float64, 1d, C)``) prints as itself.
    return f"{type(value).__qualname__}:{value}"


def jit_options(obj: Any) -> str:
    """The compile options of the JIT-compiled callable *obj*, as a stable
    string: what kind of wrapper it is, its ``targetoptions`` (sorted, minus
    those that only say where the compiled code is kept, like ``cache``), the
    ``locals`` types it was given, and the signature a ``@guvectorize`` layout
    or a ``@cfunc`` was declared with."""
    t = type(obj)
    parts = [f"{t.__module__}.{t.__qualname__}"]
    holders = [obj]
    inner = _attr(obj, "_dispatcher")
    if inner is not None:
        holders.append(inner)
    for holder in holders:
        options = _attr(holder, "targetoptions")
        if isinstance(options, dict):
            kept = {k: v for k, v in options.items() if k not in _NOT_RESULT_OPTIONS}
            parts.append("options=" + _stable(kept))
        local_types = _attr(holder, "locals")
        if isinstance(local_types, dict) and local_types:
            parts.append("locals=" + _stable(local_types))
    layout = _attr(_attr(obj, "gufunc_builder"), "signature")
    if layout is not None:
        parts.append("layout=" + _stable(layout))
    sig = _attr(obj, "_sig")
    if sig is not None:
        parts.append("sig=" + _stable(sig))
    return ";".join(dict.fromkeys(parts))


def jit_identity(function_identity: str, obj: Any) -> str:
    """The identity of the JIT-compiled callable *obj*, from
    *function_identity*, the identity of the function it compiles."""
    return hashlib.sha256(f"{function_identity}:jit:{jit_options(obj)}".encode("utf-8")).hexdigest()
