"""Records as a parser returns them -- a list of dicts holding lists and
dicts -- have no holder but the variable, unless another name reaches into
them.

The check that tells the two apart reads a whole level of containers at a
time, so it must give the same answers as asking each container: a dict,
a list or a tuple inside the records that another variable holds keeps the
records shared, and records nothing else reaches into are not.
"""

from __future__ import annotations

from cash.notebook.shared_objects import shared_names


def _records(n: int = 50) -> list:
    return [{"id": i, "tags": ["a", "b"], "meta": {"k": i}, "pair": (i, [i])} for i in range(n)]


def _shared(namespace: dict) -> set[str]:
    return shared_names({"records": namespace["records"]}, (namespace,))


def test_records_nothing_else_reaches_into_are_not_shared():
    assert _shared({"records": _records()}) == set()


def test_records_in_a_dict_of_columns_are_not_shared():
    namespace = {"records": {"train": _records(), "test": _records(5)}}

    assert _shared(namespace) == set()


def test_a_record_another_variable_holds_keeps_them_shared():
    namespace = {"records": _records()}
    namespace["first"] = namespace["records"][0]

    assert _shared(namespace) == {"records"}


def test_a_list_inside_a_record_another_variable_holds_keeps_them_shared():
    namespace = {"records": _records()}
    namespace["tags"] = namespace["records"][30]["tags"]

    assert _shared(namespace) == {"records"}


def test_a_dict_two_levels_down_another_variable_holds_keeps_them_shared():
    namespace = {"records": {"train": _records()}}
    namespace["meta"] = namespace["records"]["train"][7]["meta"]

    assert _shared(namespace) == {"records"}


def test_a_tuple_another_variable_holds_keeps_them_shared():
    namespace = {"records": _records()}
    namespace["pair"] = namespace["records"][3]["pair"]

    assert _shared(namespace) == {"records"}


def test_a_container_held_by_something_other_than_a_variable_keeps_them_shared():
    namespace = {"records": _records()}
    registry = [namespace["records"][10]["meta"]]

    assert _shared(namespace) == {"records"}
    assert registry


def test_one_record_twice_in_the_list_is_not_another_holder():
    namespace = {"records": _records()}
    namespace["records"].append(namespace["records"][0])

    assert _shared(namespace) == set()
