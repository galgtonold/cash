"""A frame of object columns and no rows holds nothing, and reading it for
held objects raises nothing.

``df = pd.DataFrame([], columns=['a', 'b', 'c'])`` under ``%cash_on`` raised
"cannot reshape array of size 0 into shape (0)" in the user's cell: the
object block has shape ``(3, 0)``, and ``reshape(-1, 0)`` cannot infer the
``-1``. Plain Python runs the line fine.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import _pandas_objects

pd = pytest.importorskip("pandas")


def test_a_frame_of_object_columns_and_no_rows():
    assert _pandas_objects(pd.DataFrame([], columns=["a", "b", "c"])) == []


def test_an_empty_object_series():
    assert _pandas_objects(pd.Series([], dtype=object)) == []


def test_an_object_column_with_rows_still_reports_its_lists():
    lists = [[1], [2]]
    found = _pandas_objects(pd.DataFrame({"a": lists, "b": ["x", "y"]}))
    assert any(item is lists[0] for item in found)
