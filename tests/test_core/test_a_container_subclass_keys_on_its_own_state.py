"""A dict/list subclass's own state is part of the key, not just its items.

Found while attacking the decorator before round 26: canonicalising a container
rebuilds it from its items and tags the type name, so state that is not an item
never reached the key -- ``defaultdict(list)`` and ``defaultdict(set)`` shared
one entry, and a ``dict`` subclass carrying ``self.source`` served the first
caller's answer for every source. Pickle carries both, so this was signal cash
had and dropped.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from cash import Cash
from cash.backends import InMemoryBackend


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


def test_a_defaultdict_factory_is_part_of_the_key(cash):
    @cash.cache
    def bucket(mapping, key):
        missing = mapping[key] if key in mapping else mapping.default_factory()
        return f"{type(missing).__name__}: {missing!r}"

    assert bucket(defaultdict(list, {"a": 1}), "new") == "list: []"
    assert bucket(defaultdict(set, {"a": 1}), "new") == "set: set()"


def test_a_dict_subclasss_attributes_are_part_of_the_key(cash):
    class Config(dict):
        def __init__(self, source, **kw):
            super().__init__(**kw)
            self.source = source

    @cash.cache
    def where_from(cfg):
        return f"{cfg.source} {dict(cfg)!r}"

    assert where_from(Config("prod", x=1)) == "prod {'x': 1}"
    assert where_from(Config("staging", x=1)) == "staging {'x': 1}"


def test_a_list_subclasss_attributes_are_part_of_the_key(cash):
    class Rows(list):
        def __init__(self, label, items):
            super().__init__(items)
            self.label = label

    @cash.cache
    def describe(rows):
        return f"{rows.label}:{len(rows)}"

    assert describe(Rows("a", [1, 2])) == "a:2"
    assert describe(Rows("b", [1, 2])) == "b:2"


def test_equal_plain_containers_still_share_an_entry(cash):
    ran = []

    @cash.cache
    def total(values):
        ran.append(1)
        return sum(values.values()) if isinstance(values, dict) else sum(values)

    assert total({"a": 1, "b": 2}) == 3
    assert total({"b": 2, "a": 1}) == 3
    assert total([1, 2]) == 3
    assert total([1, 2]) == 3
    assert len(ran) == 2, "an equal plain dict/list must keep sharing its entry"


def test_a_defaultdict_with_the_same_factory_still_hits(cash):
    ran = []

    @cash.cache
    def size(mapping):
        ran.append(1)
        return len(mapping)

    assert size(defaultdict(list, {"a": 1})) == 1
    assert size(defaultdict(list, {"a": 1})) == 1
    assert len(ran) == 1
