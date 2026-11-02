"""The closure check passes over each tuple of leaves on its own, not only
a chunk of nothing but such tuples: the state numpy's ``__reduce_ex__``
gives an object column is a tuple holding a list, met beside the column's
cells, and it kept a frame's two million ``(action, time)`` records each a
place of the check, 4 s a statement."""

from __future__ import annotations

import datetime

import pytest

from cash import closure_sharing

pd = pytest.importorskip("pandas")

T0 = datetime.datetime(2019, 10, 1)


@pytest.fixture(params=["native", "python"])
def kernels(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(closure_sharing, "_KERNELS", None, raising=False)
    elif getattr(closure_sharing, "_KERNELS", None) is None:
        pytest.skip("no C extension")


def test_only_the_tuples_holding_more_than_leaves_are_kept(kernels):
    held = [1]
    more = (2, held)
    records = [(f"a{i}", T0) for i in range(10_000)]
    chunk = ["x", *records[:5000], more, (), *records[5000:], held, ("y", ("z",))]
    assert closure_sharing._without_leaf_records(chunk) == ["x", more, held, ("y", ("z",))]


def test_a_column_of_records_is_no_level_of_places(kernels, monkeypatch):
    frame = pd.DataFrame({"actions": [(f"a{i % 7}", T0 + datetime.timedelta(seconds=i)) for i in range(20_000)]})
    assert frame["actions"].dtype == object
    sizes: list[int] = []
    fresh = closure_sharing._fresh

    def counted(held, places, limit):
        level = fresh(held, places, limit)
        sizes.append(len(level or ()))
        return level

    monkeypatch.setattr(closure_sharing, "_fresh", counted)
    assert closure_sharing.closures_sharing({"df": frame}, 30.0).names == []
    assert max(sizes) < 1000


def test_a_function_in_a_tuple_beside_records_is_still_found(kernels):
    box = {"a": [1]}
    records = [(f"a{i}", T0) for i in range(10_000)]
    out = [*records, (1, lambda: box["a"])]
    assert closure_sharing.closures_sharing({"out": out, "box": box}, 30.0).names == ["out"]
