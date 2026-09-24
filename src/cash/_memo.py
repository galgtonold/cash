"""A memo that keeps at most a fixed number of entries.

For caches whose keys keep arriving for as long as a session runs -- one per
code object, per statement, per cache entry -- so that a long notebook with
many redefined cells does not grow them without limit.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

__all__ = ["LruMemo"]

K = TypeVar("K")
V = TypeVar("V")


class LruMemo(Generic[K, V]):
    """A mapping of at most *maxsize* entries that drops the least recently
    used one to make room. Reading an entry (``get``, ``in`` does not count)
    marks it used."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None) -> V | None:
        try:
            value = self._data[key]
        except KeyError:
            return default
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def pop(self, key: K, default: V | None = None) -> V | None:
        return self._data.pop(key, default)

    def clear(self) -> None:
        self._data.clear()
