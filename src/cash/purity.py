"""@pure and @stateful decorator system for caching decisions."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

__all__ = [
    "pure",
    "stateful",
    "is_pure",
    "is_stateful",
    "is_known_pure",
    "KNOWN_PURE_BUILTINS",
]

F = TypeVar("F", bound=Callable[..., Any])

# Attribute names used to mark functions
_PURE_ATTR = "_cash_pure"
_STATEFUL_ATTR = "_cash_stateful"


def pure(func: F) -> F:
    """Promise that a function has no side effects cash needs to know about.

    With ``@cash.cache``, cash trusts it: a cached function that calls it
    is not warned about it. Its code is still part of that function's key,
    so editing it recomputes the function, as editing any helper does. In a
    notebook, a statement that calls a helper that writes a file caches only
    when the helper is marked pure. It is a promise, not a check.

    Args:
        func: The function to mark.

    Returns:
        ``func`` itself, marked.

    Examples:
        ```python
        @pure
        def compute(x, y):
            return x + y
        ```
    """
    setattr(func, _PURE_ATTR, True)
    return func


def stateful(func: F) -> F:
    """Declare that a function's effects matter beyond its return value.

    With ``@cash.cache``, the function (or a cached function that calls it)
    is reported with `CashImpurityWarning`, and with ``strict=True`` it is
    refused. Its code is part of a caller's key, as any helper's is. In a
    notebook, a statement that calls it is never cached, so it runs every
    time.

    Args:
        func: The function to mark.

    Returns:
        ``func`` itself, marked.

    Examples:
        ```python
        @stateful
        def train_model(data):
            model.fit(data)
            return model.score(data)
        ```
    """
    setattr(func, _STATEFUL_ATTR, True)
    return func


def is_pure(func: Any) -> bool:
    """Return whether ``func`` was marked with `pure`."""
    return getattr(func, _PURE_ATTR, False) is True


def is_stateful(func: Any) -> bool:
    """Return whether ``func`` was marked with `stateful`."""
    return getattr(func, _STATEFUL_ATTR, False) is True


# ============================================================================
# Known-pure built-in and stdlib functions
# ============================================================================

#: Builtins that touch nothing but their arguments: no file, no global, no
#: output. ``next`` advances the iterator it is handed; the rest only read.
#: Not "same output for the same inputs": ``id`` and ``hash`` answer per
#: object and per process.
#: Matched by name, so a caller that can see the user's bindings checks that
#: the name still means the builtin (`is_known_pure`).
KNOWN_PURE_BUILTINS: frozenset[str] = frozenset(
    {
        # Type constructors / conversions
        "int",
        "float",
        "str",
        "bool",
        "bytes",
        "complex",
        "list",
        "tuple",
        "set",
        "frozenset",
        "dict",
        # Numeric / math
        "abs",
        "round",
        "pow",
        "divmod",
        "min",
        "max",
        "sum",
        # Sequence / iteration
        "len",
        "sorted",
        "reversed",
        "enumerate",
        "zip",
        "range",
        "map",
        "filter",
        "all",
        "any",
        # Object introspection
        "type",
        "isinstance",
        "issubclass",
        "id",
        "hash",
        "callable",
        "hasattr",
        "getattr",
        "repr",
        "ascii",
        "format",
        "chr",
        "ord",
        "hex",
        "oct",
        "bin",
        # Containers
        "iter",
        "next",
        "slice",
    }
)


def is_known_pure(name: str, namespace: Mapping[str, Any] | None = None) -> bool:
    """Is *name* one of `KNOWN_PURE_BUILTINS`, and, given the *namespace* it
    is looked up in, still bound to that builtin there?

    A user's own ``sorted`` or ``next`` marked `stateful` is not the builtin,
    and must not be waved through on its name.
    """
    if name not in KNOWN_PURE_BUILTINS:
        return False
    if namespace is None or name not in namespace:
        return True
    return namespace[name] is getattr(builtins, name, None)
