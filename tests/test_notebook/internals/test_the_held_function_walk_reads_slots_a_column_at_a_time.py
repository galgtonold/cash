"""The walk for the notebook functions a value holds (``held_functions``)
reads a level of slot objects a slot at a time, and does not visit an object
that can lead only to its class.

An openpyxl workbook of 400,000 cells was walked object by object, its
cells and as many style arrays: 2-3 s a walk, and a formatting cell walked
it before each statement that read it (15-20 s a run, next to 13 s of its
own work). These tests pin the work (objects visited) and that the walk
still finds every function such a structure holds.
"""

from __future__ import annotations

import array

import pytest

from cash.notebook import held_functions


class Style(array.array):
    """Like openpyxl's ``StyleArray``: an array with no attributes."""

    __slots__ = ()


class Cell:
    """Like openpyxl's ``Cell``: slots only, naming its sheet."""

    __slots__ = ("value", "style", "parent", "comment")

    def __init__(self, value, parent):
        self.value = value
        self.style = Style("i", [0, 0])
        self.parent = parent
        self.comment = None


class Sheet:
    def __init__(self, n: int):
        self.cells = {(i, 1): Cell(float(i), self) for i in range(n)}


#: The notebook: the functions its cells define.
NOTEBOOK: dict = {}
exec("def make():\n    return 1\ndef other():\n    return 2", NOTEBOOK)
_make = NOTEBOOK["make"]
_other = NOTEBOOK["other"]


N = 5_000


def _visited(monkeypatch, value, namespace) -> tuple[held_functions.Held, dict[str, int]]:
    """``held_notebook_code(value)`` and the objects each type sent to a visit."""
    counts: dict[str, int] = {}
    real = held_functions._visit_group

    def counting(kind, group, *args):
        counts[kind.__name__] = counts.get(kind.__name__, 0) + len(group)
        return real(kind, group, *args)

    monkeypatch.setattr(held_functions, "_visit_group", counting)
    return held_functions.held_notebook_code(value, namespace), counts


@pytest.fixture
def namespace():
    return NOTEBOOK


def test_a_sheet_of_cells_visits_each_cell_once_and_no_style(monkeypatch, namespace):
    held, counts = _visited(monkeypatch, Sheet(N), namespace)
    assert held.complete and held.codes == ()
    assert counts["Cell"] == N
    assert "Style" not in counts, counts
    assert sum(counts.values()) < N + 50, counts


def test_a_function_in_one_cells_slot_is_found(monkeypatch, namespace):
    sheet = Sheet(N)
    sheet.cells[(N - 7, 1)].comment = _make
    held, _ = _visited(monkeypatch, sheet, namespace)
    assert held.codes == (_make.__code__,)


def test_a_slot_unset_on_one_cell_is_still_read_on_the_others(monkeypatch, namespace):
    sheet = Sheet(N)
    del sheet.cells[(3, 1)].style
    sheet.cells[(N - 1, 1)].comment = _make
    held, _ = _visited(monkeypatch, sheet, namespace)
    assert held.codes == (_make.__code__,)


class _StyleWithDict(array.array):
    """An array whose objects take attributes: walked, not stepped over."""


def test_an_array_that_takes_attributes_is_walked(monkeypatch, namespace):
    sheet = Sheet(50)
    style = _StyleWithDict("i", [0])
    style.on_change = _make
    sheet.cells[(10, 1)].style = style
    held, _ = _visited(monkeypatch, sheet, namespace)
    assert held.codes == (_make.__code__,)


class _ComputedCell(Cell):
    """A subclass whose ``value`` is a property over the slot: read as
    ``object.__getattribute__`` reads it, through the property."""

    __slots__ = ()

    def __init__(self, value, parent):
        Cell.value.__set__(self, value)
        self.style = Style("i", [0, 0])
        self.parent = parent
        self.comment = None

    @property
    def value(self):
        return _other


def test_a_property_over_a_slot_is_read_through_the_property(monkeypatch, namespace):
    cells = [_ComputedCell(1.0, None) for _ in range(20)]
    held, _ = _visited(monkeypatch, cells, namespace)
    assert held.codes == (_other.__code__,)


def test_a_level_of_many_types_is_still_walked_whole(monkeypatch, namespace):
    kinds = [type(f"K{i}", (), {}) for i in range(40)]
    objects = []
    for i in range(400):
        obj = kinds[i % 40]()
        obj.i = i
        objects.append(obj)
    objects[399].fn = _make
    objects.extend(Cell(float(i), None) for i in range(2_000))
    objects[-1].comment = _other
    held, counts = _visited(monkeypatch, objects, namespace)
    assert set(held.codes) == {_make.__code__, _other.__code__}
    assert counts["Cell"] == 2_000


class _Private:
    __slots__ = ("__hook",)

    def __init__(self, hook):
        self.__hook = hook


def test_a_function_in_a_private_slot_is_found(monkeypatch, namespace):
    held, _ = _visited(monkeypatch, [_Private(_make) for _ in range(3)], namespace)
    assert held.codes == (_make.__code__,)
