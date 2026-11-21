"""The pickled-size estimate of JSON-like data a `one_look` walked already
is 0 without reading it again: dicts, lists and tuples over strings,
numbers and dates hold no array and no object, the only things the
estimate counts. The store's two estimates of a statement's million
records read them twice more (0.13 s).
"""

from __future__ import annotations

import datetime
import types

from cash import _plain_data, sizing


def _records(n: int = 5_000) -> list:
    return [{"i": i, "v": [i, 2.5], "at": datetime.date(2020, 1, 1)} for i in range(n)]


def _refusing(monkeypatch) -> None:
    def refused(*args, **kwargs):
        raise AssertionError("the records were read for their size")

    monkeypatch.setattr(sizing, "_K", types.SimpleNamespace(items_estimate=refused))
    monkeypatch.setattr(sizing, "_items_estimate", refused)


def test_records_a_look_walked_are_not_read_for_their_size(monkeypatch):
    records = _records()
    expected = sizing.pickled_size_estimate(records)
    assert expected == 0
    with _plain_data.one_look({"records": records}):
        assert _plain_data.tree_facts(records) is not None
        _refusing(monkeypatch)
        assert sizing.pickled_size_estimate(records) == expected


def test_records_no_look_walked_are_still_read(monkeypatch):
    records = _records()
    _refusing(monkeypatch)
    with _plain_data.one_look({"other": [1]}):
        try:
            sizing.pickled_size_estimate(records)
        except AssertionError:
            return
    raise AssertionError("records no look walked were not read")
