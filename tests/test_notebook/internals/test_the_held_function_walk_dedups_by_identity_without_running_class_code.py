"""The walk for held notebook functions tells objects apart by identity
without asking each for its ``id``, and without running their class's code.

An openpyxl sheet's 400,000 cells were deduped by ``id`` twice a walk: a
new int each, and an audit event the interpreter hands to every audit hook
(cash's ``io_watch`` one included, 800,000 calls a walk). Objects whose
class hashes and compares as ``object`` does are now their own key. A class
that defines ``__eq__`` or ``__hash__`` is still told apart by ``id``: two
of its objects that call themselves equal are both walked. And the class's
methods are read from its dict, not looked up: looking ``__hash__`` up on a
``MagicMock``'s class made a new mock of a new class, which the walk met,
without end.
"""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import pytest

from cash.notebook import held_functions


class Cell:
    __slots__ = ("parent", "value")

    def __init__(self, value, parent):
        self.value = value
        self.parent = parent


class Sheet:
    def __init__(self, n: int):
        self.cells = {(i, 1): Cell(float(i), self) for i in range(n)}


class AllEqual:
    """Every object of it says it equals every other."""

    def __init__(self, held):
        self.held = held

    def __eq__(self, other):
        return isinstance(other, AllEqual)

    def __hash__(self):
        return 1


def _notebook(source: str) -> dict:
    namespace: dict = {"__name__": "__main__"}
    exec(source, namespace)
    return namespace


@pytest.fixture
def ids(monkeypatch):
    """How many objects the walk asked for their ``id``."""
    asked = [0]

    def counting(obj):
        asked[0] += 1
        return builtins.id(obj)

    monkeypatch.setattr(held_functions, "id", counting, raising=False)
    return asked


def test_cells_of_a_sheet_are_not_asked_for_their_id(ids):
    namespace = _notebook("def make():\n    return 1")
    sheet = Sheet(5000)
    sheet.cells[(4000, 1)].value = namespace["make"]
    held = held_functions._walk(sheet, namespace)
    assert held[0] == (namespace["make"],)
    assert ids[0] < 100, ids[0]


def test_objects_that_call_themselves_equal_are_each_walked():
    namespace = _notebook("def first():\n    return 1\ndef second():\n    return 2")
    value = [AllEqual(namespace["first"]), AllEqual(namespace["second"])]
    functions, _classes, complete = held_functions._walk(value, namespace)
    assert complete
    assert set(functions) == {namespace["first"], namespace["second"]}


def test_a_mock_in_a_value_is_walked_without_making_mocks():
    namespace = _notebook("def make():\n    return 1")
    mock = MagicMock()
    mock.hook = namespace["make"]
    before = len(mock.mock_calls), len(mock._mock_children)
    functions, _classes, complete = held_functions._walk([mock], namespace)
    assert complete
    assert namespace["make"] in functions
    assert (len(mock.mock_calls), len(mock._mock_children)) == before


@pytest.mark.parametrize(
    ("kind", "by_identity"),
    [(Cell, True), (object, True), (type, True), (AllEqual, False), (list, False), (int, False)],
)
def test_only_a_class_that_hashes_and_compares_as_object_does_is_its_own_key(kind, by_identity):
    assert held_functions._identity_hashed(kind) is by_identity
