"""A frame of long lists is sized without pickling each list whole.

The size estimate pickles each sampled Python object of an object column to
see what it adds. A frame of ``groupby(...).agg(list)`` holds one list per
group, so every list was pickled whole: seven lists of 140,000 timestamps,
3 s on each save. A long list or tuple is sized from every so many of its
items, which still comes close to its pickle.
"""

from __future__ import annotations

import pickle

import pytest

from cash import sizing

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.core]


@pytest.mark.parametrize(
    "build",
    [
        lambda n: list(pd.date_range("2020-01-01", periods=n, freq="min")),
        lambda n: list(range(n)),
        lambda n: [str(i) for i in range(n)],
        lambda n: tuple(np.random.default_rng(0).random(n)),
    ],
    ids=["timestamps", "ints", "strings", "tuple of floats"],
)
def test_a_long_list_is_sized_close_to_its_pickle(build):
    items = build(50_000)
    real = len(pickle.dumps(items, protocol=5))
    assert 0.8 * real < sizing._pickled_item_cost(items) < 1.25 * real


def test_a_long_list_is_not_pickled_whole(monkeypatch):
    items = list(range(50_000))
    pickled: list = []
    real = pickle.dumps

    def counted(value, *args, **kwargs):
        pickled.append(len(value))
        return real(value, *args, **kwargs)

    monkeypatch.setattr(sizing.pickle, "dumps", counted)
    sizing._pickled_item_cost(items)
    assert pickled and max(pickled) <= sizing._OBJECT_SAMPLE


def test_a_short_list_is_still_pickled_whole():
    items = [{"a": i} for i in range(10)]
    assert sizing._pickled_item_cost(items) == len(pickle.dumps(items, protocol=5))
