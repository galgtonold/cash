"""The bounded memo the file tracker's jobs remember their answers in."""

from __future__ import annotations

from typing import Any

__all__ = ["Memo"]


class Memo:
    """A dict that stops growing at *limit* entries.

    Past the limit a new key is either not remembered, or (``reset=True``,
    for a memo whose old entries go stale anyway) the memo starts over.
    """

    __slots__ = ("_data", "_limit", "_reset")

    def __init__(self, limit: int, *, reset: bool = False) -> None:
        self._data: dict[Any, Any] = {}
        self._limit = limit
        self._reset = reset

    def get(self, key: Any, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def put(self, key: Any, value: Any) -> bool:
        """Remember *value* under *key*; False if the memo is full and keeps it out."""
        if key not in self._data and len(self._data) >= self._limit:
            if not self._reset:
                return False
            self._data.clear()
        self._data[key] = value
        return True
