"""@pure and @stateful decorator system for caching decisions."""

from __future__ import annotations

import functools
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
    """Mark a function as pure (no side effects) for the purity analyzer.

    This is a promise to the analyzer, not a caching switch. Its load-bearing
    effect is on the ``@cash.cache`` decorator: a callee marked pure is trusted,
    so the ``CashImpurityWarning`` that would otherwise fire for it is
    suppressed (see :mod:`cash.purity_analyzer`). In the notebook *statement*
    path it changes one verdict: a call to a helper that writes a file (a
    chart, an export) runs every time, as the write itself would, unless the
    helper is marked pure. Otherwise an unmarked helper's statements already
    cache. Use :func:`stateful` when you need to stop a statement from caching.

    Args:
        func: The function to mark as pure.

    Returns:
        The same function with a ``_cash_pure`` attribute set to True.

    Example::

        @pure
        def compute(x, y):
            return x + y
    """
    setattr(func, _PURE_ATTR, True)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, _PURE_ATTR, True)
    return wrapper  # type: ignore[return-value]  # wrapper preserves F's signature via @wraps


def stateful(func: F) -> F:
    """Mark a function as stateful (has side effects).

    Stateful functions should never be cached because their return value
    alone does not capture their full effect. The notebook caching system
    will skip caching for statements that call stateful functions.

    Args:
        func: The function to mark as stateful.

    Returns:
        The same function with a ``_cash_stateful`` attribute set to True.

    Example::

        @stateful
        def train_model(data):
            model.fit(data)
            return model.score(data)
    """
    setattr(func, _STATEFUL_ATTR, True)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, _STATEFUL_ATTR, True)
    return wrapper  # type: ignore[return-value]  # wrapper preserves F's signature via @wraps


def is_pure(func: Any) -> bool:
    """Check if a function is marked as pure.

    Args:
        func: The function or callable to check.

    Returns:
        True if the function has the ``_cash_pure`` attribute set to True.
    """
    return getattr(func, _PURE_ATTR, False) is True


def is_stateful(func: Any) -> bool:
    """Check if a function is marked as stateful.

    Args:
        func: The function or callable to check.

    Returns:
        True if the function has the ``_cash_stateful`` attribute set to True.
    """
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
