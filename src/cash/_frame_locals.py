"""One local of a running frame, read without copying the others.

Before Python 3.13 (PEP 667) ``frame.f_locals`` of a function frame copies
every local into a dict the frame keeps until it returns: a value the code
then lets go of (``del big``) stays alive through that copy. A trace that
reads one local as a call runs (a method's receiver, the name a
``getattr`` looks up) would keep all of them alive. There `frame_local`
reads the one local through ``PyFrame_GetVar`` (3.12), which copies
nothing; elsewhere ``f_locals`` is a view and is read as it is.
"""

from __future__ import annotations

import sys
from typing import Any

#: ``inspect.CO_OPTIMIZED``: a function frame, its locals in slots.
_CO_OPTIMIZED = 0x1


def _get_var() -> Any:
    if sys.version_info >= (3, 13) or sys.implementation.name != "cpython":
        return None
    try:
        import ctypes

        get_var = ctypes.pythonapi.PyFrame_GetVar
    except (ImportError, AttributeError):
        return None
    get_var.restype = ctypes.py_object
    get_var.argtypes = (ctypes.py_object, ctypes.py_object)
    return get_var


_GET_VAR = _get_var()


def frame_local(frame: Any, name: str, default: Any = None) -> Any:
    """The local *name* of *frame* (a cell's value for a cell variable);
    *default* when it is unbound or not one of its locals."""
    if _GET_VAR is None or not frame.f_code.co_flags & _CO_OPTIMIZED:
        return frame.f_locals.get(name, default)
    try:
        return _GET_VAR(frame, name)
    except NameError:
        return default
