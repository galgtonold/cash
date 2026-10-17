"""A value pandas keeps in a frame's or an index's own slots is refused
without searching the heap, with the same answer.

``y = df['t']`` stays in ``df._item_cache``, and a fitted model's
``feature_names_in_`` is ``X.columns._data``: the search up the garbage
collector's referrers climbed to that frame or index, stopped there (a
library's object), and refused the store, 20-60 ms a statement for one
search of every tracked object per level. The references are now counted
against those slots; one from anywhere else still sends the search.
"""

from __future__ import annotations

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import GaveUp, share_group

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _check(names, ns, *, search: bool):
    """`share_group` for *names*, with the slot check off when *search*;
    ``(shared, gave_up names, complete, heap searches)``."""
    searches = []
    real_search = shared_objects._holders_from_referrers
    real_slots = shared_objects._held_in_pandas_slots
    try:
        shared_objects._holders_from_referrers = lambda *a, **k: searches.append(1) or real_search(*a, **k)
        if search:
            shared_objects._held_in_pandas_slots = lambda *a, **k: None
        gave_up = GaveUp()
        # The captured values are a dict of their own, as the processor passes them.
        _holders, shared = share_group(names, {n: ns[n] for n in names}, ns, gave_up=gave_up)
    finally:
        shared_objects._holders_from_referrers = real_search
        shared_objects._held_in_pandas_slots = real_slots
    return shared, gave_up.names, gave_up.complete, len(searches)


def _frame():
    df = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0), "t": np.arange(200) % 2})
    return df


def test_a_column_the_frame_caches_is_refused_as_the_search_refuses_it():
    df = _frame()
    ns = {"df": df, "y": df["t"]}
    if "t" not in vars(df).get("_item_cache", {}):
        pytest.skip("this pandas keeps no item cache (copy-on-write)")
    searched = _check(["y"], ns, search=True)
    counted = _check(["y"], ns, search=False)
    assert searched[:3] == counted[:3] == ({"y"}, {"df"}, True), (searched, counted)
    assert searched[3] >= 1 and counted[3] == 0


def test_an_array_an_index_holds_is_refused_as_the_search_refuses_it():
    df = _frame()
    X = df[["a", "b"]]
    if not isinstance(X.columns._data, np.ndarray):
        pytest.skip("this pandas keeps the index in another array type")
    # No local name for the array: a reference from this frame is one more holder.
    ns = {"X": X, "model": {"feature_names_in_": X.columns._data}}
    searched = _check(["model"], ns, search=True)
    counted = _check(["model"], ns, search=False)
    assert searched[:3] == counted[:3], (searched, counted)
    assert searched[0] == {"model"}
    assert counted[3] == 0


def test_a_reference_from_anywhere_else_still_searches():
    df = _frame()
    y = df["t"]
    if "t" not in vars(df).get("_item_cache", {}):
        pytest.skip("this pandas keeps no item cache (copy-on-write)")
    elsewhere = [y]  # held by a list no variable names: only the search finds it
    ns = {"df": df, "y": y}
    searched = _check(["y"], ns, search=True)
    counted = _check(["y"], ns, search=False)
    assert counted == searched
    assert counted[3] >= 1
    del elsewhere
