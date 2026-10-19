"""A wide frame's digest looks its column dtypes up once, not once per column.

p04-backtest: a grid loop hands a fresh 3131 x 2000 signal frame to a cached
backtest. Its digest took 0.27 s, 0.25 s of it in ``frame.dtypes``: the memo
key of each column looked its dtype up there, and ``dtypes`` builds a Series
of all 2000 columns' on every call. The column walk now passes the dtype it
already has, so a digest is 0.03 s; the digest and the memo key are the same.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from cash.decorator import arg_hashing

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

COLUMNS = 400


@pytest.fixture(autouse=True)
def empty_memo():
    arg_hashing.FRAME_DIGESTS.clear()
    yield
    arg_hashing.FRAME_DIGESTS.clear()


def _wide() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((3000, COLUMNS)), columns=[f"t{i}" for i in range(COLUMNS)])
    frame["when"] = pd.date_range("2000-01-01", periods=3000, freq="D")
    frame["n"] = pd.array(rng.integers(0, 9, 3000), dtype="Int64")
    return frame


def test_a_wide_frame_digest_reads_dtypes_once_and_digests_as_a_full_read(monkeypatch):
    frame = _wide()
    full = content_hashers.builtin_hash(frame)
    arg_hashing.FRAME_DIGESTS.clear()
    looked = []
    dtypes = pd.DataFrame.dtypes
    monkeypatch.setattr(pd.DataFrame, "dtypes", property(lambda self: looked.append(1) or dtypes.fget(self)))
    digest = arg_hashing.FRAME_DIGESTS.hash_frame(frame)
    monkeypatch.undo()
    assert digest == full
    assert len(looked) < 10, len(looked)


def test_the_column_memory_key_is_the_same_with_the_dtype_passed():
    frame = _wide()
    for pos, dtype in enumerate(frame.dtypes):
        route = content_hashers.pandas_value_route(dtype, pd)
        assert content_hashers.column_memory_key(frame, pos, route, dtype) == content_hashers.column_memory_key(
            frame, pos, route
        )
