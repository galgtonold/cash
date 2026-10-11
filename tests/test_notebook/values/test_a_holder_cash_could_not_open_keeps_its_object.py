"""A variable holding a list through a ``__slots__`` object, a ``UserDict``,
an object array or a pandas ``attrs`` keeps holding that very list.

``ds = Slotted(rows)`` stored and restored on the next Run All held a copy
of ``rows``: the share check could not look inside it, so it did not know
``ds`` and ``rows`` hold one list, and ``rows.append(3)`` was not seen
through ``ds``. The check now looks inside them, so the two are stored and
restored together. A pandas object copies its ``attrs`` apart whatever is
stored with it, so a statement that leaves one holding an object something
else holds too runs every time.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import children_of
from tests._cell_driver import run_cash_cell


def test_an_object_array_keeps_the_very_list_on_every_run(cash_magics):
    """The kernel's own Run All for the other holders is in
    ``test_notebook_integration/calls/test_a_holder_cash_could_not_open_keeps_its_object.py``."""
    pytest.importorskip("numpy")
    ns = cash_magics.shell.user_ns
    seen = []
    for _ in range(3):
        for cell in [
            "import numpy as np\nrows = [1, 2]",
            "ds = np.empty(1, dtype=object)\nds[0] = rows",
            "rows.append(3)",
        ]:
            run_cash_cell(cash_magics, cell)
        seen.append(ns["ds"][0] is ns["rows"])
    assert seen == [True, True, True]


class _Slotted:
    __slots__ = ("rows", "_Slotted__hidden")

    def __init__(self, rows):
        self.rows = rows


class _Child(_Slotted):
    __slots__ = ("more",)


def test_a_slotted_object_gives_every_slot_along_its_classes(monkeypatch):
    from cash.notebook import shared_objects

    monkeypatch.setattr(shared_objects, "_is_local_module_class", lambda cls: True)
    rows, more = [1], [2]
    obj = _Child(rows)
    obj.more = more
    children = children_of(obj)
    assert children is not None
    assert any(child is rows for child in children)
    assert any(child is more for child in children)


def test_an_object_array_view_is_not_opened():
    """A view's items are its base's: counted through the view too, they
    would hide a holder from the reference counts."""
    np = pytest.importorskip("numpy")
    arr = np.empty(4, dtype=object)
    assert children_of(arr) is not None
    assert children_of(arr[1:3]) is None
    assert children_of(np.arange(3)) is None


def test_attrs_copied_apart_only_when_something_else_holds_it():
    from cash.notebook.shared_objects import attrs_copied_apart

    pd = pytest.importorskip("pandas")
    rows = [1, 2]
    df = pd.DataFrame({"a": [1]})
    df.attrs["rows"] = rows
    assert attrs_copied_apart([df, rows])
    assert attrs_copied_apart([{"df": df, "rows": rows}])
    assert not attrs_copied_apart([df])
    other = pd.DataFrame({"a": [1]})
    other.attrs["rows"] = rows
    assert attrs_copied_apart([df, other])
    assert not attrs_copied_apart([pd.DataFrame({"a": [1]}), rows])
