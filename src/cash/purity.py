"""@pure and @stateful decorator system for caching decisions."""

from __future__ import annotations

from collections.abc import Callable
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

# These built-in functions have no side effects and always return
# the same output for the same inputs.
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


def is_known_pure(name: str) -> bool:
    """Check if a function name is a known-pure built-in.

    This allows the statement processor to skip mutation detection
    for statements that only call known-pure built-in functions,
    even without explicit ``@pure`` annotations.

    Args:
        name: The function name to check.

    Returns:
        True if the name is in the known-pure builtins list.
    """
    return name in KNOWN_PURE_BUILTINS
