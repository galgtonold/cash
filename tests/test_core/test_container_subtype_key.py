"""Container SUBCLASSES must not collide onto their base type's cache key.

`@cash.cache` canonicalised tuple/dict/list subclasses (namedtuple, OrderedDict,
defaultdict) into their base type before hashing the argument, so two genuinely
different inputs shared one cache entry and the second call was served the
first's result -- a silent wrong-HIT. `f(P(1,2))` returned `f(Q(1,2))`'s value
for two distinct namedtuple types with equal values.

Found by an adversarial round-16 tester against the 0.1.1 build and reproduced
independently. The counterpart guarantee matters too: an EXACT plain tuple/dict
must keep its old key (no cache invalidation) and still HIT on a repeat call.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict, namedtuple

import pytest

import cash

P = namedtuple("P", "x y")
Q = namedtuple("Q", "x y")


def _fresh():
    from cash import Cash
    from cash.backends import InMemoryBackend
    return Cash(backend=InMemoryBackend(), register_magic=False)


def test_two_namedtuple_types_do_not_collide():
    inst = _fresh()
    calls = []

    @inst.cache(assume_safe=True)
    def describe(t):
        calls.append(type(t).__name__)
        return f"{type(t).__name__}:{tuple(t)}"

    a = describe(P(1, 2))
    b = describe(Q(1, 2))  # different type, equal values
    assert a == "P:(1, 2)"
    assert b == "Q:(1, 2)", f"Q call served the P entry: {b!r}"
    assert calls == ["P", "Q"], "the second, differently-typed call was a wrong HIT"


def test_namedtuple_does_not_collide_with_plain_tuple():
    inst = _fresh()

    @inst.cache(assume_safe=True)
    def kind(t):
        return type(t).__name__

    assert kind((1, 2)) == "tuple"
    assert kind(P(1, 2)) == "P", "namedtuple served the plain-tuple entry"


def test_ordereddict_does_not_collide_with_dict():
    inst = _fresh()

    @inst.cache(assume_safe=True)
    def kind(d):
        return type(d).__name__

    assert kind({"a": 1}) == "dict"
    assert kind(OrderedDict([("a", 1)])) == "OrderedDict"


def test_defaultdict_does_not_collide_with_dict():
    inst = _fresh()

    @inst.cache(assume_safe=True)
    def kind(d):
        return type(d).__name__

    assert kind({"a": 1}) == "dict"
    dd = defaultdict(int)
    dd["a"] = 1
    assert kind(dd) == "defaultdict"


def test_plain_containers_still_hit_and_key_is_unchanged():
    """Exact base types must be unaffected -- same key, still cached."""
    inst = _fresh()
    calls = []

    @inst.cache
    def total(t):
        calls.append(1)
        return sum(t)

    assert total((1, 2, 3)) == 6
    assert total((1, 2, 3)) == 6        # HIT
    assert len(calls) == 1, "plain tuple stopped hitting -- key changed for the common case"

    # A plain dict equal but for insertion order must still share a key (CAS-108).
    @inst.cache
    def keys(d):
        calls.append(1)
        return sorted(d)

    keys({"a": 1, "b": 2})
    before = len(calls)
    keys({"b": 2, "a": 1})              # reordered -> must HIT
    assert len(calls) == before, "insertion-order-only dict difference wrongly missed"


def test_equal_namedtuples_of_the_same_type_still_hit():
    """Same type + same values must remain a HIT (no over-invalidation)."""
    inst = _fresh()
    calls = []

    @inst.cache
    def f(t):
        calls.append(1)
        return tuple(t)

    f(P(1, 2))
    f(P(1, 2))
    assert len(calls) == 1, "two equal same-type namedtuples should share a key"


def test_ordereddict_order_is_significant():
    inst = _fresh()

    @inst.cache(assume_safe=True)
    def first_key(d):
        return next(iter(d))

    assert first_key(OrderedDict([("a", 1), ("b", 2)])) == "a"
    # Different order, same items: an OrderedDict distinguishes these, so cash
    # must not serve the first entry for the second.
    assert first_key(OrderedDict([("b", 2), ("a", 1)])) == "b"
