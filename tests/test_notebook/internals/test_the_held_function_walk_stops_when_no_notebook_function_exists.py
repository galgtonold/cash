"""The walk for the notebook functions a value holds stops early when no
notebook function exists at all, and still finds every one that does.

p30-excel-finance: ``wb_out = openpyxl.load_workbook(REPORT_PATH)`` then
three statements formatting it and ``wb_out.save(REPORT_PATH)``. Each
statement changed the workbook, so each walked its 400,000 cells again for
the functions it might hold (``held_functions``): five walks a run, 0.6 s
each, though every helper of the notebook lives in a module and no cell
had defined a function. A walk past `EXISTENCE_CHECK_AT` objects now asks
whether any function with the notebook's globals exists (one
``gc.get_referrers`` of the namespace); when none does, nothing can be
found and the walk stops. Its answer is kept only while that stays so.
"""

from __future__ import annotations

import array
import gc

import pytest

from cash.notebook import held_functions


@pytest.fixture(autouse=True)
def _a_library_s_classes(monkeypatch):
    """The classes below stand in for a library's (openpyxl's): as this
    file's own, they would be the user's, whose objects the walk looks for
    to the end (``Held.module_classes``), as it would the test run's own."""
    monkeypatch.setattr(held_functions, "_of_a_users_module", lambda cls: False)
    monkeypatch.setattr(held_functions, "module_class_objects_exist", lambda: False)


class Style(array.array):
    __slots__ = ()


class Cell:
    """Like openpyxl's ``Cell``: slots only, naming its sheet."""

    __slots__ = ("comment", "parent", "style", "value")

    def __init__(self, value, parent):
        self.value = value
        self.style = Style("i", [0, 0])
        self.parent = parent
        self.comment = None


class Sheet:
    def __init__(self, n: int):
        self.cells = {(i, 1): Cell(float(i), self) for i in range(n)}


N = 3 * held_functions.EXISTENCE_CHECK_AT


def _visited(monkeypatch, value, namespace, lineage=None):
    """``held_notebook_code(value)`` and the number of objects visited."""
    seen = [0]
    real = held_functions._visit_group

    def counting(kind, group, *args):
        seen[0] += len(group)
        return real(kind, group, *args)

    monkeypatch.setattr(held_functions, "_visit_group", counting)
    held = held_functions.held_notebook_code(value, namespace, lineage)
    monkeypatch.setattr(held_functions, "_visit_group", real)
    return held, seen[0]


def _notebook(source: str = "") -> dict:
    namespace: dict = {"__name__": "__main__"}
    if source:
        exec(source, namespace)
    return namespace


def test_a_big_value_is_not_walked_to_the_end_when_no_notebook_function_exists(monkeypatch):
    held, visited = _visited(monkeypatch, Sheet(N), _notebook("x = 1"))
    assert held.complete and held.codes == ()
    assert visited < N / 2, visited


def test_a_big_value_is_walked_to_the_end_when_a_notebook_function_exists(monkeypatch):
    held, visited = _visited(monkeypatch, Sheet(N), _notebook("def helper():\n    return 1"))
    assert held.complete and held.codes == ()
    assert visited >= N


def test_a_function_only_the_value_holds_is_found(monkeypatch):
    """A lambda no name binds, made by a cell and kept deep in the value."""
    namespace = _notebook("hook = lambda: cfg\ncfg = 3")
    hook = namespace.pop("hook")
    sheet = Sheet(N)
    sheet.cells[(N - 5, 1)].comment = hook
    del hook
    held, _ = _visited(monkeypatch, sheet, namespace)
    assert len(held.codes) == 1 and held.codes[0].co_name == "<lambda>"


def test_a_kept_answer_is_asked_again_once_a_notebook_function_exists(monkeypatch):
    namespace = _notebook()
    sheet = Sheet(N)
    held, _ = _visited(monkeypatch, sheet, namespace, lineage="L1")
    assert held.codes == ()
    # The same object, under the same lineage, now holds a function a cell
    # defined (a library object a callback was registered on).
    exec("def on_save():\n    return cfg", namespace)
    sheet.cells[(7, 1)].comment = namespace["on_save"]
    held, visited = _visited(monkeypatch, sheet, namespace, lineage="L1")
    assert held.codes == (namespace["on_save"].__code__,)
    assert visited >= N


def test_methods_and_lambdas_in_containers_count_as_notebook_functions():
    assert not held_functions.notebook_functions_exist(_notebook("x = [1, 2]"))
    with_class = _notebook("class Model:\n    def fit(self):\n        return 1")
    assert held_functions.notebook_functions_exist(with_class)
    in_list = _notebook("fns = [lambda: 1]")
    assert held_functions.notebook_functions_exist(in_list)


def test_frozen_objects_are_not_searched_so_a_function_is_assumed(monkeypatch):
    monkeypatch.setattr(gc, "get_freeze_count", lambda: 1)
    held, visited = _visited(monkeypatch, Sheet(N), _notebook("x = 1"))
    assert held.complete and visited >= N


def test_a_chained_namespace_finds_a_function_in_any_of_its_maps():
    from collections import ChainMap

    outer = _notebook("def f():\n    return 1")
    assert held_functions.notebook_functions_exist(ChainMap({}, outer))
    assert not held_functions.notebook_functions_exist(ChainMap({}, _notebook("y = 2")))


def test_a_function_only_the_namespace_holds_is_found_through_a_value_holding_the_namespace(monkeypatch):
    # Nothing refers to ``make`` but the namespace, and a cell defined an
    # unrelated function too. The value holds the namespace, so it holds
    # ``make``: no check of who refers to a function may stop this walk.
    namespace = _notebook("def unrelated():\n    return 1\ndef make():\n    return cfg")
    sheet = Sheet(N)
    sheet.notes = {"ns": namespace}
    held, _ = _visited(monkeypatch, sheet, namespace)
    assert namespace["make"].__code__ in held.codes


def test_a_function_bound_in_the_namespace_answers_without_a_scan_of_every_object(monkeypatch):
    scans = []
    real = gc.get_referrers

    def counting(*objects):
        scans.append(len(objects))
        return real(*objects)

    monkeypatch.setattr(held_functions.gc, "get_referrers", counting)
    assert held_functions.notebook_functions_exist(_notebook("def helper():\n    return 1"))
    assert scans == []
    # One no name binds is still found, by the scan.
    assert held_functions.notebook_functions_exist(_notebook("keep = [lambda: 1]"))
    assert scans
