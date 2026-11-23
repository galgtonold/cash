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
import itertools
import operator
import weakref
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from . import _weak

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
                refs[key] = _weak.ref(cls, gone)
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


#: The ``__mro__`` of a class as ``type`` reads it: no metaclass of the
#: user's runs.
mro_of: Callable[[type], tuple] = type.__dict__["__mro__"].__get__


def _class_reader(name: str) -> Callable[[Any], Any]:
    """``obj.<name>`` read by ``type``'s own descriptor for a class, so a
    ``__getattribute__`` or ``__getattr__`` of its metaclass's (code of the
    user's, which plain Python never asks for these) does not run. An
    object that only passes for a class (a ``__class__`` property, which
    ``isinstance(obj, type)`` believes) is read as ``obj.<name>`` reads it,
    as before."""
    read = type.__dict__[name].__get__

    def class_reader(obj: Any) -> Any:
        if type(obj) is type:
            return read(obj)
        try:
            # A metaclass may define the name itself, in C (Cython 3.3's
            # ``_common_types_metatype`` gives ``__module__``, where
            # ``type``'s descriptor reads the instances' member descriptor
            # out of the class's namespace): ``type``'s lookup finds that,
            # and still runs no ``__getattribute__`` or ``__getattr__``.
            return type.__getattribute__(obj, name)
        except TypeError:
            if issubclass(type(obj), type):
                raise
            return getattr(obj, name)

    class_reader.__name__ = class_reader.__qualname__ = f"class_reader_{name.strip('_')}"
    return class_reader


# A class's own facts as ``type`` reads them, for any class cash meets.
#: Its namespace (``vars(cls)``, ``cls.__dict__``).
class_dict: Callable[[type], Any] = _class_reader("__dict__")
#: ``cls.__name__``.
class_name: Callable[[type], str] = _class_reader("__name__")
#: ``cls.__qualname__``.
class_qualname: Callable[[type], str] = _class_reader("__qualname__")
#: ``cls.__module__``; raises AttributeError for a class made with none.
class_module: Callable[[type], Any] = _class_reader("__module__")
#: ``cls.__flags__``.
class_flags: Callable[[type], int] = _class_reader("__flags__")
#: ``cls.__dictoffset__``.
class_dictoffset: Callable[[type], int] = _class_reader("__dictoffset__")
#: ``cls.__bases__``.
class_bases: Callable[[type], tuple] = _class_reader("__bases__")
#: ``cls.__mro__`` for a value that may only pass for a class (`mro_of`
#: takes a real one).
class_mro: Callable[[type], tuple] = _class_reader("__mro__")
#: ``getattr(cls, name)`` by ``type``'s own lookup (the metaclass's
#: descriptors, then the MRO's namespaces), never by the metaclass's
#: ``__getattribute__`` or ``__getattr__``.
type_lookup: Callable[[type, str], Any] = type.__getattribute__


def module_of_class(cls: type, default: Any = None) -> Any:
    """*cls*'s ``__module__`` (`class_module`), *default* for a class made
    with none."""
    try:
        return class_module(cls)
    except AttributeError:
        return default


class MroMemo(TypeMemo):
    """A `TypeMemo` for an answer read along a class's MRO: a class keeps its
    address when its ``__bases__`` are assigned, and its MRO changes. Read
    with `get_for`; an answer for another MRO is not found."""

    __slots__ = ()

    def get_for(self, cls: type, default: Any = None) -> Any:
        found = dict.get(self, type_key(cls))
        if found is None or not same_mro(found[0], cls):
            return default
        return found[1]

    def put(self, cls: type, value: V) -> V:  # type: ignore[override]
        TypeMemo.put(self, cls, (mro_mark(cls), value))
        return value


def mro_mark(cls: type) -> tuple:
    """What `same_mro` tells *cls*'s MRO by: its classes after *cls*.

    Not the MRO tuple itself, which refers to the class, and not its
    address: three ``__bases__`` assignments in a row gave the new MRO the
    first one's address (the tuple's memory reused), and a memo answered for
    bases the class no longer had. The bases are held only while the class's
    entry is."""
    return mro_of(cls)[1:]


def same_mro(mark: tuple, cls: type) -> bool:
    """Is *cls*'s MRO still the one *mark* (`mro_mark`) was taken of? By
    identity: no metaclass ``__eq__`` of the user's runs."""
    mro = mro_of(cls)
    return len(mark) == len(mro) - 1 and all(map(operator.is_, mark, itertools.islice(mro, 1, None)))


def per_mro(fn: Callable[[type], V]) -> Callable[[type], V]:
    """`per_type` for a function reading the class's MRO (`MroMemo`)."""
    memo: MroMemo = MroMemo()
    get, put = memo.get_for, memo.put

    @functools.wraps(fn)
    def remembered(kind: type) -> V:
        found = get(kind, _ABSENT)
        if found is _ABSENT:
            found = put(kind, fn(kind))
        return found

    remembered.memo = memo  # type: ignore[attr-defined]
    return remembered
