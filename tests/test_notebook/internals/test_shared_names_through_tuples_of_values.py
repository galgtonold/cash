"""A record made of tuples of plain values has no holder to find, and a tuple
that carries something mutable still does.

``logs = [(user, [(action, when), ...]), ...]`` is a list of millions of small
tuples; the walk does not stop at each one. These cases pin what the shortcut
for a tuple of values must not change.
"""

from __future__ import annotations

import datetime

from cash.notebook.shared_objects import shared_names

WHEN = datetime.datetime(2019, 10, 15, 18, 8, 2)


def _shared(namespace: dict) -> set[str]:
    return shared_names({"rows": namespace["rows"]}, (namespace,))


def test_a_list_of_tuples_of_values_is_not_shared():
    namespace = {"rows": [("u1", [("a", WHEN), ("b", WHEN)]), ("u2", [("c", WHEN)])]}

    assert _shared(namespace) == set()


def test_a_tuple_of_tuples_of_values_is_not_shared():
    namespace = {"rows": [(("a", 1), (("b", 2.5), frozenset({"c"})))]}

    assert _shared(namespace) == set()


def test_a_list_inside_a_tuple_that_another_variable_holds_is_shared():
    namespace = {"rows": [("u1", [("a", WHEN)])]}
    namespace["alias"] = namespace["rows"][0][1]

    assert _shared(namespace) == {"rows"}


def test_a_list_nested_two_tuples_deep_that_another_variable_holds_is_shared():
    namespace = {"rows": [("u1", (("a", WHEN), (1, [2])))]}
    namespace["alias"] = namespace["rows"][0][1][1][1]

    assert _shared(namespace) == {"rows"}
