"""``compute_hash`` reads a big frame once, then checks it, while copy-on-write proves it unchanged.

A notebook loop handing a 50 MB frame to a cached function on every
iteration took the frame's digest several times per iteration -- the loop's
in-place-change watch before and after, the checks around each call -- and
spent 0.7 s a call doing so (p04-backtest: 47 s against 26 s plain). The
digest is kept (`FrameDigests`) while the frame's blocks, axes and manager
are the same objects and no handle that writes past pandas was given out.
Every way of changing the frame must still give a new digest, the one a
fresh read gives: a stale one would let a loop's in-place edit go unseen.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from cash.content_hashers import builtin_hash
from cash.value_hash import compute_hash

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)

ROWS = 100_000  # 2.4 MB: above the memo's floor


@pytest.fixture
def reads(monkeypatch):
    seen = []
    real = content_hashers.hash_pandas

    def counting(value):
        seen.append(id(value))
        return real(value)

    monkeypatch.setattr(content_hashers, "hash_pandas", counting)
    return seen


def _frame():
    return pd.DataFrame({"x": np.arange(ROWS, dtype=float), "y": np.ones(ROWS), "z": np.arange(ROWS)})


def test_an_unchanged_frame_is_read_once(reads):
    df = _frame()
    first = compute_hash(df)
    for _ in range(5):
        df.x.mean()
        assert compute_hash(df) == first
    assert reads.count(id(df)) == 1


def test_a_small_frame_is_read_every_time(reads):
    df = pd.DataFrame({"x": np.arange(10.0)})
    for _ in range(3):
        compute_hash(df)
    assert reads.count(id(df)) == 3


def _write_loc(df):
    df.loc[5, "x"] = -1.0


def _write_iloc(df):
    df.iloc[7, 1] = 3.0


def _assign_column(df):
    df["y"] = df["y"] * 2


def _add_column(df):
    df["w"] = 1.0


def _through_array(df):
    df["x"].array[0] = 100.0


def _through_writable_view(df):
    values = df["x"].to_numpy()
    values.flags.writeable = True
    values[1] = 50.0


def _rename(df):
    df.columns = ["a", "b", "c"]


def _index_name(df):
    df.index.name = "row"


def _attrs(df):
    df.attrs["unit"] = "m"


def _inplace_method(df):
    df.fillna(0.0, inplace=True)
    df.clip(lower=10.0, inplace=True)


@pytest.mark.parametrize(
    "change",
    [
        _write_loc,
        _write_iloc,
        _assign_column,
        _add_column,
        _through_array,
        _through_writable_view,
        _rename,
        _index_name,
        _attrs,
        _inplace_method,
    ],
)
def test_every_change_gives_the_digest_a_fresh_read_gives(change):
    df = _frame()
    before = compute_hash(df)
    assert compute_hash(df) == before
    change(df)
    after = compute_hash(df)
    assert after != before
    assert after == builtin_hash(df.copy(deep=True))


def test_a_frame_over_an_array_the_caller_keeps_is_read_every_time(reads):
    data = np.zeros((ROWS, 3))
    df = pd.DataFrame(data, copy=False)
    first = compute_hash(df)
    data[0, 0] = 9.0
    assert compute_hash(df) != first
    assert reads.count(id(df)) == 2
