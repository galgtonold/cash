"""The check for other holders of a statement's output answers a flat list
or dict of values (``d = [f(i) for i in range(N)]``) at C speed, without the
tree check, which sizes every item for the facts it keeps: 20,000 ints made
each such statement about a millisecond slower. Pins the work, not the
behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

from cash import _plain_data
from cash.notebook import shared_objects


def _tree_checks(monkeypatch) -> list[object]:
    asked: list[object] = []
    real = _plain_data.held_only_by_parents
    monkeypatch.setattr(
        _plain_data, "held_only_by_parents", lambda value, leaves: asked.append(value) or real(value, leaves)
    )
    return asked


def test_a_flat_list_of_ints_skips_the_tree_check(monkeypatch):
    asked = _tree_checks(monkeypatch)
    namespace = {"d": [i * 2 for i in range(20_000)], "m": {f"k{i}": i for i in range(100)}}
    assert not shared_objects.shared_names(dict(namespace), (namespace,))
    assert asked == []


def test_records_still_take_the_tree_check(monkeypatch):
    asked = _tree_checks(monkeypatch)
    records = [{"id": i} for i in range(100)]
    namespace = {"records": records}
    shared_objects.shared_names({"records": records}, (namespace,))
    assert asked == [records]


def test_a_flat_list_two_names_bind_is_still_shared():
    values = [i * 2 for i in range(20_000)]
    namespace = {"d": values, "e": values}
    assert shared_objects.shared_names({"d": values}, (namespace,))
