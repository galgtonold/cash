"""A list that holds itself through a pandas object column, a dict in such
a column or an object NumPy array holds itself, and is not restored into.

An object array is not tracked by the garbage collector, so
``gc.get_referents`` reports nothing in it: ``L = [df];
L.append(pd.DataFrame({'o': [L, 1]}))`` was read as holding nothing of its
own, and a loop growing ``L`` was restored into a copy whose frames held a
copy of ``L``. Frames and arrays are now read by type: their attributes,
labels and the cells of their object arrays.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import in_place

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

_ITSELF = "it holds itself"


def _in_a_column(owner):
    return pd.DataFrame({"o": [owner, 1]})


def _in_a_dict_in_a_column(owner):
    return pd.DataFrame({"o": [{"k": owner}, 1]})


def _in_an_object_array(owner):
    cells = np.empty(2, dtype=object)
    cells[0] = owner
    cells[1] = 1
    return cells


def _in_a_series(owner):
    return pd.Series([owner, 1])


def _in_the_attrs(owner):
    frame = pd.DataFrame({"a": [1]})
    frame.attrs["owner"] = owner
    return frame


def _in_an_index_label(owner):
    return pd.DataFrame({"a": [1, 2]}, index=pd.Index([owner, 2], dtype=object))


@pytest.mark.parametrize(
    "holder",
    [_in_a_column, _in_a_dict_in_a_column, _in_an_object_array, _in_a_series, _in_the_attrs, _in_an_index_label],
)
def test_the_list_holds_itself(holder):
    owner = [pd.DataFrame({"a": [1, 2]})]
    owner.append(holder(owner))

    assert _ITSELF in in_place.refusal(owner)


def test_frames_and_arrays_of_values_hold_nothing_of_it():
    """The control: text, numbers, categories and a MultiIndex hold no
    object a restore would copy, and the list is restored into."""
    owner = [
        pd.DataFrame({"s": ["a", None], "n": [1, 2]}),
        pd.DataFrame({"c": pd.Categorical(["x", "y"])}, index=pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])),
        np.array(["x", None, 3], dtype=object),
        np.arange(4.0),
        pd.Series([1.5]),
    ]

    assert in_place.refusal(owner) is None


def test_too_many_cells_to_read_refuses(monkeypatch):
    """Past the cells one check reads, the list is refused as one holding
    too many objects to look through is, not read on."""
    monkeypatch.setattr(in_place, "_MAX_CELLS", 10)
    owner = [np.array([[1], 2] * 10, dtype=object)]

    assert "too many objects" in in_place.refusal(owner)
