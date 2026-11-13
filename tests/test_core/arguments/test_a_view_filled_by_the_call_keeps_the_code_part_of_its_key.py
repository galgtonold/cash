"""A view the body fills on an argument of the user's class leaves the code part of the key alone.

networkx (installed from a checkout, so its classes are user code) fills
``G.adj`` with an ``AdjacencyView`` the first time a body reads it. The walk
for the code an argument carries then found one more class in ``G`` after the
first call than before it, so the identical second call missed ("a function
or class passed as an argument changed") and only the third hit. The view
holds nothing but links to its graph, and its class is still keyed: the
getter that makes it names it, and the graph's class folds what its code
reaches.
"""

from __future__ import annotations

import functools


class ItemsView:
    __slots__ = ("_items",)

    def __init__(self, items):
        self._items = items

    def total(self):
        return sum(self._items.values())


class Basket:
    def __init__(self, items):
        self._items = dict(items)

    @functools.cached_property
    def view(self):
        return ItemsView(self._items)


def test_the_second_identical_call_hits(disk_cash):
    runs = []

    @disk_cash.cache
    def total(basket):
        runs.append(1)
        return basket.view.total()  # fills basket.view

    basket = Basket({"a": 1, "b": 2})
    assert "view" not in vars(basket)
    assert total(basket) == 3
    assert "view" in vars(basket)
    assert total(basket) == 3
    assert total(basket) == 3
    assert len(runs) == 1


def test_an_edit_to_the_view_class_still_misses(disk_cash, monkeypatch):
    runs = []

    @disk_cash.cache
    def total(basket):
        runs.append(1)
        return basket.view.total()

    basket = Basket({"a": 1, "b": 2})
    assert total(basket) == 3
    assert total(basket) == 3
    monkeypatch.setattr(ItemsView, "total", lambda self: 10 * sum(self._items.values()))
    assert total(basket) == 30
    assert len(runs) == 2
