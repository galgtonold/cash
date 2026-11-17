"""The check that a frame's memory is pandas' own walks each list of
references once, however many Series share the index.

Every Series built over one ``DatetimeIndex``, and every result of an
operation on them, holds a view of that index, and all the views share one
list of references. The walk (`frame_borrows_its_data`) went over that list
again for each view it reached, so its steps grew with the square of the
live Series: with a year of hourly ``pvlib`` inputs, a cached
``get_total_irradiance`` call that missed spent about 60 ms in cash for a
3 ms body. Each list is now walked once; what the walk finds is the same.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import frame_borrows_its_data

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the walk runs under copy-on-write only"
)


def _steps(monkeypatch, series: pd.Series) -> int:
    """How many live references the walk followed for *series*."""
    calls = 0
    real = arg_hashing.is_pin

    def counting(obj):
        nonlocal calls
        calls += 1
        return real(obj)

    monkeypatch.setattr(arg_hashing, "is_pin", counting)
    assert frame_borrows_its_data(series) is False
    monkeypatch.setattr(arg_hashing, "is_pin", real)
    return calls


def _siblings(index: pd.Index, count: int) -> list[pd.Series]:
    """*count* Series over views of *index* (as ``clip`` and ``np.maximum``
    give their results), which all share its list of references."""
    base = pd.Series(np.arange(len(index), dtype=float), index)
    return [base.clip(lower=i) for i in range(count)]


def test_the_walk_grows_with_the_series_not_their_square(monkeypatch):
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    few = _siblings(index, 20)
    few_steps = _steps(monkeypatch, few[0])
    many = few + _siblings(index, 180)
    many_steps = _steps(monkeypatch, many[0])
    # Ten times the Series: about ten times the steps (each Series' block
    # and index view), not a hundred times.
    assert few_steps > 20
    assert many_steps < 15 * few_steps, (few_steps, many_steps)


def test_an_array_held_outside_pandas_is_still_found_among_many_siblings():
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    siblings = _siblings(index, 200)
    outside = np.arange(48, dtype=float)
    borrowing = pd.Series(outside, index, copy=False)
    assert frame_borrows_its_data(borrowing) is True
    labels = np.arange(48)
    over_labels = pd.Series(np.arange(48, dtype=float), index=labels)
    siblings += [over_labels.clip(lower=i) for i in range(50)]
    # ``index=arr`` is not copied: the caller still holds the labels' array.
    assert frame_borrows_its_data(siblings[-1]) is True
    assert frame_borrows_its_data(siblings[0]) is False
