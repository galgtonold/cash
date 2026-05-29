"""Deferred-resolution proxy for cache values.

A :class:`LazyProxy` wraps a zero-argument *loader* and the entry's
metadata, so callers can inspect metadata (size, ttl, type info) and
decide whether they actually want the value *before* paying the
deserialization cost. The value is loaded at most once, on first access.

Pair it with :func:`make_lazy_loader`, which builds a proxy from a backend
and a key: it reads metadata up front via ``backend.get_metadata`` (cheap,
no value deserialization) and defers the full ``backend.get`` until the
proxy is resolved.
"""

from __future__ import annotations

__all__ = ["LazyProxy", "make_lazy_loader"]

from typing import Any
from collections.abc import Callable

from ._base import CacheBackend, MetadataDict


class LazyProxy:
    """A value that is loaded on first access, not on construction.

    Args:
        loader: Zero-argument callable returning the wrapped value. Invoked
            at most once, on the first :meth:`resolve` / :attr:`value` access.
        metadata: The entry's metadata, available without resolving the
            value. Defaults to an empty dict.
        cache_key: Optional key the proxy stands in for, used in ``repr``.
    """

    def __init__(
        self,
        loader: Callable[[], Any],
        metadata: MetadataDict | None = None,
        cache_key: str | None = None,
    ) -> None:
        self._loader = loader
        self.metadata: MetadataDict = metadata if metadata is not None else {}
        self.cache_key = cache_key
        self._resolved = False
        self._value: Any = None

    @property
    def is_resolved(self) -> bool:
        """Whether the value has been loaded yet."""
        return self._resolved

    def resolve(self) -> Any:
        """Load the value (once) and return it."""
        if not self._resolved:
            self._value = self._loader()
            self._resolved = True
        return self._value

    @property
    def value(self) -> Any:
        """The wrapped value, loading it on first access."""
        return self.resolve()

    def __repr__(self) -> str:
        if self._resolved:
            return f"<LazyProxy resolved value={self._value!r}>"
        return f"<LazyProxy pending cache_key={self.cache_key!r}>"


def make_lazy_loader(backend: CacheBackend, key: str) -> LazyProxy | None:
    """Build a :class:`LazyProxy` for *key* in *backend*, or ``None`` if absent.

    Reads metadata via ``backend.get_metadata`` so existence and metadata are
    known without deserializing the value; the value itself is fetched via
    ``backend.get`` only when the proxy is resolved.
    """
    metadata = backend.get_metadata(key)
    if metadata is None:
        return None
    return LazyProxy(lambda: backend.get(key)[1], metadata=metadata, cache_key=key)
