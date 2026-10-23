"""A per-class answer cash remembers without keeping the class alive.

A walk asks the same question of every object of a class (is it an
estimator, is it from the user's own module, what slots does it have), and
remembers the answer by class. Kept in a plain dict, the class was its key:
every class a function makes per call (a factory returning an instance of a
class defined inside it) stayed alive, with its methods and what their
closures held, for the rest of the session.

Here the answer is kept in a plain dict by the class's address
(`type_key`), beside a weak reference to the class whose callback drops the
entry. Python calls it as the class goes, before its memory can hold another
object, so an entry found by address is always the live class's: the walks
read ``memo.get(type_key(cls))`` at C speed, with no check of their own. An
answer must not refer to its class (a slot's descriptor does), or it would
keep the class alive itself.

The address is read as ``object.__hash__`` gives it (the address, rotated:
one per live object), not by ``id()``: Python raises an audit event for
every ``id()``, and each audit hook written in Python runs for it.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable
from typing import Any, Generic, TypeVar

V = TypeVar("V")


#: A class's address, as `TypeMemo` keys it.
type_key: Callable[[type], int] = object.__hash__


class TypeMemo(dict, Generic[V]):
    """``{type_key(class): answer}`` for live classes. Read with
    ``memo.get(type_key(cls))``; written with `put`, never by item
    assignment."""

    __slots__ = ("_refs",)

    def __init__(self) -> None:
        super().__init__()
        self._refs: dict[int, weakref.ref] = {}

    def put(self, cls: type, value: V) -> V:
        """Remember *value* for *cls* while *cls* lives, and return it."""
        key = type_key(cls)
        refs = self._refs
        ref = refs.get(key)
        if ref is None or ref() is not cls:

            def gone(ref: weakref.ref, key: int = key, memo: TypeMemo = self) -> None:
                if memo._refs.get(key) is ref:
                    del memo._refs[key]
                    memo.pop(key, None)

            try:
                refs[key] = weakref.ref(cls, gone)
            except TypeError:
                return value  # a class that cannot be weakly referenced is not remembered
        dict.__setitem__(self, key, value)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("TypeMemo is written with put(cls, value)")

    def clear(self) -> None:
        dict.clear(self)
        self._refs.clear()


_ABSENT = object()


def per_type(fn: Callable[[type], V]) -> Callable[[type], V]:
    """``functools.lru_cache`` for a function of a class alone, held
    weakly: the class is not kept alive by the answer it was asked for."""
    memo: TypeMemo[V] = TypeMemo()
    get, put = memo.get, memo.put

    @functools.wraps(fn)
    def remembered(kind: type) -> V:
        found = get(type_key(kind), _ABSENT)
        if found is _ABSENT:
            found = put(kind, fn(kind))
        return found

    remembered.memo = memo  # type: ignore[attr-defined]
    return remembered
