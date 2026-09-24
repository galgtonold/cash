"""The one bounded memo, and the size of every memo that uses it.

For caches whose keys keep arriving for as long as a process runs -- one per
code object, per statement, per file -- so that a long session with many
redefined cells does not grow them without limit.

Full, a memo drops the entry used least recently. A memo that stops taking
entries once full becomes a permanent slow path for every key after the cap,
and one that is cleared when full throws the entries in use away with the
rest; dropping the least recently used keeps what the session is working with.
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
    marks it used.

    Safe to share between threads the way a dict is: an entry another thread
    drops between two steps of an operation is not an error.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None) -> V | None:
        data = self._data
        try:
            value = data[key]
        except KeyError:
            return default
        try:
            data.move_to_end(key)
        except KeyError:
            pass  # dropped by another thread meanwhile; the value still holds
        return value

    def __setitem__(self, key: K, value: V) -> None:
        data = self._data
        data[key] = value
        try:
            data.move_to_end(key)
        except KeyError:
            pass
        while len(data) > self.maxsize:
            try:
                data.popitem(last=False)
            except KeyError:
                break  # another thread emptied it

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[K]:
        """The keys, least recently used first (a copy)."""
        return list(self._data)

    def pop(self, key: K, default: V | None = None) -> V | None:
        return self._data.pop(key, default)

    def clear(self) -> None:
        self._data.clear()
