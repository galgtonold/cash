"""The type of what ``@cash.cache`` returns, for type checkers and editors.

At run time the decorated function is a plain function with three attributes
(`Wrappers` attaches them). This protocol tells a type checker the same: the
function keeps its signature, and ``cache_info``, ``cache_clear`` and
``explain`` exist.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, Protocol, TypeVar, overload

if TYPE_CHECKING:
    from .explain import CacheExplanation

__all__ = ["CachedCallable"]

P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S")
Q = ParamSpec("Q")
R = TypeVar("R")


class CachedCallable(Protocol[P, T_co]):
    """A function decorated with ``@cash.cache``: callable like the original,
    plus ``cache_info()``, ``cache_clear()`` and ``explain(...)``."""

    __wrapped__: Callable[P, T_co]
    __name__: str
    __qualname__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co: ...

    # A decorated method binds like any function: ``obj.method(x)`` passes
    # ``obj`` as the first argument.
    @overload
    def __get__(self, instance: None, owner: type | None = None) -> CachedCallable[P, T_co]: ...

    @overload
    def __get__(
        self: CachedCallable[Concatenate[S, Q], R], instance: S, owner: type | None = None
    ) -> CachedCallable[Q, R]: ...

    def cache_info(self) -> dict[str, Any]:
        """Hits, misses, hit rate, time saved and the recent warnings."""
        ...

    def cache_clear(self) -> None:
        """Delete this function's entries and reset its statistics."""
        ...

    def explain(self, *args: Any, **kwargs: Any) -> CacheExplanation:
        """Whether a call with these arguments would hit, and why. Runs nothing."""
        ...
