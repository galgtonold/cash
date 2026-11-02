"""A function whose closure holds an Arrow column the value holds is
found, while a value of Arrow columns and no function is checked without
reading below them.

``pd.concat(df_array)`` over thousands of small frames keeps one Arrow
chunk per frame in each text column; the closure check read every chunk
through its ``__reduce_ex__`` (two checks, 40% of the statement), though an
Arrow array of a plain type holds no function. The value is first looked
through for a function with a closure, not below such arrays; only when it
holds one is it walked as before.
"""

from __future__ import annotations

import pytest

from cash.closure_sharing import closures_sharing

pa = pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

#: pandas 3's ``str``, Arrow text; pandas 2 keeps text as Python strings unless asked.
STR = pd.StringDtype("pyarrow", na_value=np.nan)


def _closing_over(part):
    return lambda: part


def _column():
    return pa.chunked_array([pa.array(["a", "b"]), pa.array(["c"])])


def test_a_function_closing_over_an_arrow_column_the_value_holds_is_found():
    chunked = _column()
    found = closures_sharing({"x": (chunked, _closing_over(chunked))}, 5.0)
    assert found.finished and found.names == ["x"]


def test_a_function_closing_over_a_frames_text_column_is_found():
    frame = pd.concat([pd.DataFrame({"t": pd.array(["a", "b"], dtype=STR)}) for _ in range(50)], ignore_index=True)
    chunked = frame["t"].array._pa_array
    found = closures_sharing({"df": frame, "f": _closing_over(chunked)}, 5.0)
    assert found.finished and found.names == ["f"]


def test_a_function_next_to_arrow_columns_closing_over_a_list_is_found():
    box = [1]
    frame = pd.concat([pd.DataFrame({"t": pd.array(["a", "b"], dtype=STR)}) for _ in range(50)], ignore_index=True)
    found = closures_sharing({"x": (frame, box, _closing_over(box))}, 5.0)
    assert found.finished and found.names == ["x"]


def test_a_function_closing_over_nothing_of_the_value_is_not_reported():
    found = closures_sharing({"x": (_column(), _closing_over([1]))}, 5.0)
    assert found.finished and found.names == []


def test_arrow_columns_with_no_function_are_not_reported():
    frame = pd.concat(
        [pd.DataFrame({"t": pd.array(["a", "b"], dtype=STR), "n": [1, 2]}) for _ in range(50)], ignore_index=True
    )
    found = closures_sharing({"df": frame}, 5.0)
    assert found.finished and found.names == []
