"""A big list or dict of Python objects is priced by a sample of its items.

The RAM tier copies such a value by a pickle round trip, an object at a
time: 1.5 million ``SimpleNamespace`` records built in 1.5 s took 8 s to
copy, against 0.07 s the byte model predicted, and a hit copied them again.
`many_objects_cost` pickles, loads and copies an evenly spread sample of
the items as the tier's store copies them, and scales by their count.
JSON-like data, which the tier copies a container at a time, and small
containers stay priced by their bytes.
"""

from __future__ import annotations

import types

from cash.backends.sampled_cost import many_objects_cost


class Record:
    def __init__(self, i: int) -> None:
        self.i = i
        self.name = f"n{i}"


def _namespaces(count: int) -> list:
    return [types.SimpleNamespace(i=i, x=i * 0.5, name=f"n{i}") for i in range(count)]


def test_a_big_list_of_records_is_priced_by_its_count():
    cost = many_objects_cost({"records": _namespaces(30_000)})

    assert cost is not None
    assert cost.items == 30_000 and cost.unit == "objects"
    assert cost.copy > 0 and cost.dump > 0 and cost.load > 0


def test_instances_of_a_class_and_dicts_holding_sets_are_priced_too():
    by_class = many_objects_cost({"records": [Record(i) for i in range(30_000)]})
    with_sets = many_objects_cost({"records": [{"i": i, "tags": {i % 7, i % 11}} for i in range(30_000)]})

    assert by_class is not None and by_class.items == 30_000
    assert with_sets is not None and with_sets.items == 30_000


def test_a_big_dict_of_records_is_priced_by_its_count():
    cost = many_objects_cost({"by_id": {i: Record(i) for i in range(30_000)}})

    assert cost is not None and cost.items == 30_000


def test_json_like_data_and_small_lists_are_left_to_the_byte_model():
    """The control: the same count of JSON-like records, of plain numbers,
    or a few thousand records, is not sampled."""
    assert many_objects_cost({"records": [{"i": i, "v": [i, i + 1]} for i in range(30_000)]}) is None
    assert many_objects_cost({"numbers": list(range(30_000))}) is None
    assert many_objects_cost({"records": _namespaces(5_000)}) is None
