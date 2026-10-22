"""A per-class answer cash remembers without keeping the class alive.

A walk asks the same question of every object of a class (is it an
estimator, is it from the user's own module, what slots does it have), and
remembers the answer by class. Kept in a plain dict, the class was its key:
every class a function makes per call (a factory returning an instance of a
class defined inside it) stayed alive, with its methods and what their
closures held, for the rest of the session. Here an answer is kept by the
class's id, beside a weak reference to it: when the class goes, so does its
answer, and an id reused by a later class does not find it.
"""

from __future__ import annotations

import weakref
from typing import Any, Generic, TypeVar

V = TypeVar("V")


class TypeMemo(Generic[V]):
    """``{class: answer}``, held weakly. ``get`` and item assignment, as a
    dict's; a class that cannot be weakly referenced is not remembered."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[int, tuple[weakref.ref, V]] = {}

    def get(self, cls: type, default: Any = None) -> V | Any:
        entry = self._entries.get(id(cls))
        if entry is not None and entry[0]() is cls:
            return entry[1]
        return default

    def __contains__(self, cls: type) -> bool:
        return self.get(cls, _ABSENT) is not _ABSENT

    def __getitem__(self, cls: type) -> V:
        found = self.get(cls, _ABSENT)
        if found is _ABSENT:
            raise KeyError(cls)
        return found

    def __setitem__(self, cls: type, value: V) -> None:
        key = id(cls)
        entries = self._entries

        def gone(ref: weakref.ref, key: int = key, entries: dict = entries) -> None:
            entry = entries.get(key)
            if entry is not None and entry[0] is ref:
                del entries[key]

        try:
            ref = weakref.ref(cls, gone)
        except TypeError:
            return
        entries[key] = (ref, value)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


_ABSENT = object()
