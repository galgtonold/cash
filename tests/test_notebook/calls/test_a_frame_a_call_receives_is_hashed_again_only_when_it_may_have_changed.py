"""A frame passed to many calls is hashed once, and again whenever it may have changed.

``scores = [evaluate(df, a) for a in alphas]`` hashed ``df`` in full before
and after every call, to see whether the callee changed it: 0.7 s a call for
an 80 MB frame. The hash is now kept while copy-on-write proves the frame
unchanged (`ArgFingerprints`). Every way of changing it -- through pandas,
through a handle pandas gave out, or through the array it was built over --
must still read as a change; a change the memo missed would let a callee's
in-place edit be skipped by a cache hit.
"""

from __future__ import annotations

import pytest

from cash.notebook import call_effects
from cash.notebook.call_effects import hash_args

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)

ROWS = 100_000  # 1.6 MB: remembered


@pytest.fixture
def full_hashes(monkeypatch):
    frames = []
    real = call_effects.compute_hash

    def counting(value):
        if isinstance(value, (pd.DataFrame, pd.Series)):
            frames.append(1)
        return real(value)

    monkeypatch.setattr(call_effects, "compute_hash", counting)
    return frames


def _frame():
    return pd.DataFrame({"x": np.arange(ROWS, dtype=float), "y": np.ones(ROWS)})


def _fingerprints():
    return call_effects.ArgFingerprints()


def test_an_unchanged_frame_is_hashed_once(full_hashes):
    df = _frame()
    fingerprints = _fingerprints()
    first = hash_args((df, 1), {}, fingerprints)
    for _ in range(5):
        assert hash_args((df, 1), {}, fingerprints) == first
        df.x.mean()  # reading it changes nothing
    assert len(full_hashes) == 1


def test_a_small_frame_is_hashed_every_time(full_hashes):
    df = pd.DataFrame({"x": [1.0, 2.0]})
    fingerprints = _fingerprints()
    hash_args((df,), {}, fingerprints)
    hash_args((df,), {}, fingerprints)
    assert len(full_hashes) == 2


def _array_write(df):
    df["x"].array[0] = -1.0


def _values_made_writable(df):
    values = df.values
    values.flags.writeable = True
    values[0, 0] = -1.0


def _index_array_write(df):
    df.index.array[0] = -1


WRITES = {
    "iloc": lambda df: df.iloc.__setitem__((0, 0), -1.0),
    "column": lambda df: df.__setitem__("x", 0.0),
    "new column": lambda df: df.__setitem__("z", 0.0),
    "inplace method": lambda df: df.clip(0, 1, inplace=True),
    "series .array": _array_write,
    "read-only view made writable": _values_made_writable,
    "index .array": _index_array_write,
    "index name": lambda df: setattr(df.index, "name", "row"),
    "columns renamed": lambda df: df.rename(columns={"x": "w"}, inplace=True),
    "attrs": lambda df: df.attrs.__setitem__("unit", "m"),
}


@pytest.mark.parametrize("write", list(WRITES))
def test_every_kind_of_change_is_seen(write):
    df = _frame()
    if write == "index .array":
        df.index = pd.Index(np.arange(ROWS))
    fingerprints = _fingerprints()
    before = hash_args((df,), {}, fingerprints)
    assert hash_args((df,), {}, fingerprints) == before  # remembered
    WRITES[write](df)
    assert hash_args((df,), {}, fingerprints) != before


def test_a_frame_built_over_an_array_someone_holds_is_hashed_every_time(full_hashes):
    data = np.zeros((ROWS, 2))
    df = pd.DataFrame(data, columns=["x", "y"], copy=False)
    fingerprints = _fingerprints()
    before = hash_args((df,), {}, fingerprints)
    data[0, 0] = 5.0
    assert hash_args((df,), {}, fingerprints) != before
    assert len(full_hashes) == 2


def test_a_cleared_memo_hashes_again(full_hashes):
    df = _frame()
    fingerprints = _fingerprints()
    hash_args((df,), {}, fingerprints)
    fingerprints.clear()
    hash_args((df,), {}, fingerprints)
    assert len(full_hashes) == 2
