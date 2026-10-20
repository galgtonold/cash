"""Reading an attribute of a notebook's object without its errors becoming cash's."""

from __future__ import annotations

from typing import Any


def attribute_or(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr(obj, name, default)``, with any error the lookup raises taken
    as the attribute missing.

    Python's own default covers AttributeError only. A class whose
    ``__getattr__`` raises KeyError for a name it does not hold (a record
    reading its fields from a dict) failed cash's own question about it --
    ``fit``, ``__code__`` -- and the cell with it, where plain Python never
    asks. The question is cash's, so the answer is: not there.
    """
    try:
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return default


_MISSING = object()


def has_attribute(obj: Any, name: str) -> bool:
    """``hasattr(obj, name)``, with any error the lookup raises taken as the
    attribute missing (`attribute_or`)."""
    return attribute_or(obj, name, _MISSING) is not _MISSING


def instance_of(obj: Any, kinds: type | tuple[type, ...]) -> bool:
    """``isinstance(obj, kinds)``, with any error the object's own lookups
    raise answered by its real type.

    When the type does not match, ``isinstance`` asks the object for its
    ``__class__`` (a proxy's way to pass for what it wraps), and so does an
    ABC's check. An object whose ``__getattribute__`` raises for it failed
    cash's own question and the cell with it, where plain Python never asks.
    """
    try:
        return isinstance(obj, kinds)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return issubclass(type(obj), kinds)


def is_routine(obj: Any) -> bool:
    """``inspect.isroutine(obj)``, with any error the object's own lookups
    raise (`instance_of`) taken as not a routine: a function, method or
    builtin is not an object whose lookups raise."""
    import inspect

    try:
        return inspect.isroutine(obj)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return False
