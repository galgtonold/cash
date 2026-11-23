"""A class of the user's whose metaclass has a ``__getattribute__`` of its
own is not asked anything by cash while it keys, runs or serves a cached
call that reaches the class.

The key of a call reaches the classes its body names and its arguments
hold: their namespaces, MRO, module, name, fields, annotations and source.
cash read each through Python's attribute lookup (``vars(cls)``,
``cls.__mro__``, ``getattr(cls, "__module__")``, ``isinstance(x, dict)``,
which asks a class that is not one for its ``__class__``), and so ran the
metaclass's hook, where plain Python never does: here its write
(``RAN.append``) changed the user's state on every call. Each is read as
``type`` reads it now; the instances' own lookups, which the user's code
makes, are not the metaclass's.
"""

from __future__ import annotations

import dataclasses

RAN: list[str] = []


class Watching(type):
    def __getattribute__(cls, name):
        RAN.append(name)
        return super().__getattribute__(name)


class Counted(metaclass=Watching):
    step = 2

    def __init__(self, base=40):
        self.base = base

    def total(self):
        return self.base + self.step


class Keyed(metaclass=Watching):
    """Keyed by what ``__cash_key__`` gives: cash finds it on the class."""

    def __init__(self, n):
        self.n = n

    def __cash_key__(self):
        return self.n


@dataclasses.dataclass
class Point(metaclass=Watching):
    x: int = 1
    y: int = 2


def _counted_total(a):
    return Counted().total() + a


def _built_by(kind, a):
    return kind(a).total()


def _keyed_value(k):
    return k.n * 2


def _norm(a):
    p = Point(a, 4)
    return p.x * p.x + p.y * p.y


def _twice(cached, *args):
    """*cached* called twice with *args*, the second a hit, and what the
    metaclass's hook was asked meanwhile."""
    RAN.clear()
    first = cached(*args)
    again = cached(*args)
    assert first == again
    assert cached.cache_info()["hits"] == 1, cached.cache_info()
    return list(RAN)


def test_a_body_naming_such_a_class_is_keyed_without_asking_it(cash_instance):
    assert _twice(cash_instance.cache(_counted_total), 1) == []


def test_such_a_class_passed_as_an_argument_is_keyed_without_asking_it(cash_instance):
    assert _twice(cash_instance.cache(_built_by), Counted, 5) == []


def test_an_argument_keyed_by_its_cash_key_is_keyed_without_asking_its_class(cash_instance):
    assert _twice(cash_instance.cache(_keyed_value), Keyed(4)) == []


def test_a_dataclass_with_such_a_metaclass_is_keyed_without_asking_it(cash_instance):
    assert _twice(cash_instance.cache(_norm), 3) == []


def test_explain_reads_such_a_class_without_asking_it(cash_instance):
    cached = cash_instance.cache(_built_by)
    cached(Counted, 6)
    RAN.clear()
    cached.explain(Counted, 6)
    assert RAN == []
