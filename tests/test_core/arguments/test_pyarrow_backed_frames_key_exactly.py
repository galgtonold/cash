"""A pyarrow-backed pandas column keys on its exact values and nulls.

Read through ``to_numpy()``, an int64 column with a missing value became
float64, so ids beyond 2**53 that differ keyed alike; a float null and a
NaN both became NaN; list and struct columns lost the same. One frame was
served the other's result.
"""

from __future__ import annotations

import pytest

from cash.content_hashers import hash_pandas
from cash.value_hash import compute_hash

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

BIG = 1234567890123456789

PAIRS = {
    "int64 ids with a null": (pa.int64(), [BIG, None], [BIG - 89, None]),
    "uint64 with a null": (pa.uint64(), [2**64 - 1, None], [2**64 - 2, None]),
    "float null against NaN": (pa.float64(), [1.0, None], [1.0, float("nan")]),
    "list of int64": (pa.list_(pa.int64()), [[BIG], None], [[BIG - 89], None]),
    "list of float, null against NaN": (pa.list_(pa.float64()), [[1.0, None]], [[1.0, float("nan")]]),
    "struct of int64": (pa.struct([("x", pa.int64())]), [{"x": BIG}, None], [{"x": BIG - 89}, None]),
}


def _series(values, kind):
    return pd.Series(pd.array(pa.array(values, type=kind), dtype=pd.ArrowDtype(kind)))


@pytest.mark.parametrize("pair", PAIRS)
def test_different_arrow_columns_key_apart(pair):
    kind, a, b = PAIRS[pair]
    for wrap in (lambda s: s, lambda s: s.to_frame("x"), lambda s: pd.Series([1] * len(s), index=pd.Index(s))):
        assert hash_pandas(wrap(_series(a, kind))) != hash_pandas(wrap(_series(b, kind)))
    assert compute_hash(_series(a, kind).to_frame("x")) != compute_hash(_series(b, kind).to_frame("x"))


@pytest.mark.parametrize("pair", PAIRS)
def test_equal_arrow_columns_key_alike(pair):
    kind, a, _b = PAIRS[pair]
    assert hash_pandas(_series(a, kind)) == hash_pandas(_series(a, kind)) is not None


def test_a_slice_keys_like_its_copy():
    s = _series(list(range(10)), pa.int64())
    assert hash_pandas(s.iloc[2:4].reset_index(drop=True)) == hash_pandas(_series([2, 3], pa.int64()))
    assert hash_pandas(s.iloc[:3].reset_index(drop=True)) != hash_pandas(s.iloc[3:6].reset_index(drop=True))


def test_a_call_is_served_its_own_result(cash_instance):
    @cash_instance.cache
    def max_id(df):
        return int(df["id"].max())

    @cash_instance.cache
    def count_missing(df):
        return int(df["x"].isna().sum())

    for ids in ([BIG, None], [BIG - 89, None]):
        assert max_id(_series(ids, pa.int64()).to_frame("id")) == max(i for i in ids if i is not None)
    assert count_missing(_series([1.0, None], pa.float64()).to_frame("x")) == 1
    assert count_missing(_series([1.0, float("nan")], pa.float64()).to_frame("x")) == 0
