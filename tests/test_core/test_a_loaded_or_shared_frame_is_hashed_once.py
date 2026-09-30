"""A frame read from a file, built from numpy, or sharing its data with
another pandas object is hashed once, not on every call.

The copy-on-write memo re-hashed a frame whenever a block's array was a view
(``pd.DataFrame(ndarray)``, ``read_csv``, a ``groupby`` result: pandas keeps
2-D blocks as a transposed view of the array it allocated) or had a reference
besides its block (``d2 = df.assign(z=1)``, ``df[cols]``, ``s = df["c"]``,
``df.reset_index()`` all share the blocks under copy-on-write). That is how
most frames live, so a 1M x 20 frame paid 150-190 ms per cache hit.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash import Cash, object_hashing
from cash.backends import InMemoryBackend
from cash.decorator import arg_hashing

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

N = 200


@pytest.fixture
def hashed(monkeypatch):
    """The frames whose content was hashed, by call."""
    calls: list[str] = []
    real = object_hashing.hash_pandas

    def counting(value):
        calls.append(type(value).__name__)
        return real(value)

    monkeypatch.setattr(object_hashing, "hash_pandas", counting)
    return calls


@pytest.fixture
def shape():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def shape(frame):
        return frame.shape

    return shape


def _numbers(n=N):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((n, 4)), columns=["a", "b", "c", "d"])


def _mixed():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "i": rng.integers(0, 5, N),
            "f": rng.random(N),
            "s": [f"n{i % 7}" for i in range(N)],
            "d": pd.date_range("2020-01-01", periods=N, freq="min"),
        }
    )


def _from_csv(tmp_path):
    path = tmp_path / "x.csv"
    _mixed().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["d"])


SOURCES = {
    "from a 2-D numpy array": lambda tmp_path: _numbers(),
    "read_csv": _from_csv,
    "groupby sum": lambda tmp_path: _mixed().groupby("i")[["f"]].sum(),
    "groupby on two keys": lambda tmp_path: _mixed().groupby(["i", "s"]).agg(f=("f", "mean")),
    "a date index": lambda tmp_path: _numbers().set_index(pd.date_range("2020-01-01", periods=N)),
}


@pytest.mark.parametrize("make", list(SOURCES.values()), ids=list(SOURCES))
def test_a_frame_from_a_common_source_is_hashed_once(make, tmp_path, shape, hashed):
    frame = make(tmp_path)
    for _ in range(4):
        assert shape(frame) == frame.shape
    assert len(hashed) == 1, hashed


SHARING = {
    "d.assign(z=1)": lambda d: d.assign(z=1),
    "d[cols]": lambda d: d[["a", "b"]],
    "d['a']": lambda d: d["a"],
    "d.reset_index()": lambda d: d.reset_index(),
    "d.set_index('a')": lambda d: d.set_index("a"),
    "a filtered view": lambda d: d[d["a"] > 0.5],
}


@pytest.mark.parametrize("share", list(SHARING.values()), ids=list(SHARING))
def test_a_frame_another_pandas_object_shares_is_hashed_once(share, shape, hashed):
    frame = _numbers()
    other = share(frame)
    for _ in range(4):
        assert shape(frame) == frame.shape
    assert len(hashed) == 1, hashed
    del other


DERIVED = {name: share for name, share in SHARING.items() if name != "a filtered view"}  # that one is a copy


@pytest.mark.parametrize("share", list(DERIVED.values()), ids=list(DERIVED))
def test_the_derived_frame_itself_is_hashed_once(share, shape, hashed):
    frame = _numbers()
    derived = share(frame)
    for _ in range(4):
        assert shape(derived) == derived.shape
    assert len(hashed) == 1, hashed


def test_a_frame_resampled_by_pandas_is_hashed_once(shape, hashed):
    """``resample`` reads the index through ``asi8``, which is recorded only
    when code outside pandas asks for it."""
    frame = _numbers().set_index(pd.date_range("2020-01-01", periods=N, freq="h"))
    frame.resample("D").mean()
    frame.rolling("2h").sum()
    for _ in range(4):
        assert shape(frame) == frame.shape
    assert len(hashed) == 1, hashed
