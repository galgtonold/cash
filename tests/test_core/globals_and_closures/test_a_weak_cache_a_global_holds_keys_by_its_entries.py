"""A weak mapping a global holds keys by its entries, not its machinery.

A ``WeakValueDictionary`` keeps a ``_remove`` callback whose default is a
weak reference to the mapping itself. Walking an object that held one (a
unit registry indexing its contexts) reached that callback and failed on
its default: the call ran uncached with KEY-UNHASHABLE-DEFAULT naming
``WeakValueDictionary.__init__.<locals>.remove``. The entries are what the
program reads, and are kept: cash cannot tell a cache of derived entries
from a table that changes what its holder does.
"""

from __future__ import annotations

import warnings
import weakref

from cash import Cash, FileBackend


class _Entry:
    def __init__(self, factor):
        self.factor = factor


class _Holder:
    def __init__(self, scale):
        self.scale = scale
        self.kept = [_Entry(2)]
        self.index = weakref.WeakValueDictionary({"double": self.kept[0]})
        self.handler = lambda x: x * self.scale


TABLE = _Holder(10)


def _scaled(x):
    return x * TABLE.scale * TABLE.index["double"].factor


def test_a_global_holding_a_weak_mapping_is_keyed_and_hits(tmp_path):
    global TABLE
    f = Cash(backend=FileBackend(cache_dir=str(tmp_path))).cache(_scaled)
    try:
        TABLE = _Holder(10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert f(1) == 20
            assert f(1) == 20
        assert [str(w.message) for w in caught if "KEY-" in str(w.message)] == []
        assert f.cache_info()["hits"] == 1

        TABLE.scale = 11
        assert f(1) == 22
        TABLE.kept[0].factor = 3
        assert f(1) == 33
        assert f.cache_info()["hits"] == 1
    finally:
        TABLE = _Holder(10)
