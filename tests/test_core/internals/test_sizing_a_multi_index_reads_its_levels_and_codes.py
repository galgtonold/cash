"""The pickled-size estimate reads a MultiIndex as pickle writes it.

`pickled_size_estimate` read a frame's index through ``index._values``. For
a MultiIndex that is an object array of one tuple per row, which pandas
builds on first read and keeps in the index's own cache: on a frame of
876,000 daily profiles (``nb-speed`` set item p26-load-clustering) the
estimate took 0.13 s, left 876,000 tuples the garbage collector tracks for
as long as the frame lived, made the closure check of the statement's
outputs walk them (0.3 s), and came to twice what pickle writes. A
MultiIndex pickles as its levels and codes; the estimate now reads those,
and leaves the index as the user's code left it.
"""

from __future__ import annotations

import pickle

import pytest

from cash import sizing

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.core]


def _profiles() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    idx = pd.MultiIndex.from_product([np.arange(300, dtype="int32"), dates], names=["meter_id", "date"])
    return pd.DataFrame(np.ones((len(idx), 4)), index=idx, columns=list("abcd"))


def test_the_index_is_left_without_its_tuples():
    frame = _profiles()
    assert "_values" not in frame.index._cache
    sizing.pickled_size_estimate(frame)
    assert "_values" not in frame.index._cache


def test_the_estimate_is_about_what_pickle_writes():
    frame = _profiles()
    real = len(pickle.dumps(frame, protocol=5))
    estimate = sizing.pickled_size_estimate(frame)
    assert 0.8 * real <= estimate <= 1.2 * real


def test_a_series_with_a_multi_index_too():
    series = _profiles()["a"]
    real = len(pickle.dumps(series, protocol=5))
    estimate = sizing.pickled_size_estimate(series)
    assert "_values" not in series.index._cache
    # Pickle may copy the column out of its frame's block; the index is
    # no longer counted as a tuple per row.
    assert 0.5 * real <= estimate <= 1.2 * real


def test_string_levels_are_still_sampled_as_objects():
    # The levels are indexes of their own: one of strings is sampled as
    # its objects, as a flat index of strings is, not skipped.
    names = [f"meter-{i:05d}-{'x' * 40}" for i in range(2_000)]
    idx = pd.MultiIndex.from_product([pd.Index(names, dtype=object), [0, 1]])
    frame = pd.DataFrame({"v": np.zeros(len(idx))}, index=idx)
    estimate = sizing.pickled_size_estimate(frame)
    assert estimate >= 2_000 * 40
    assert "_values" not in frame.index._cache
