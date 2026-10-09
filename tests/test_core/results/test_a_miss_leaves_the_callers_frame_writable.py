"""A miss leaves the caller's tables as writable as they were.

The RAM tier shares a stored table between hits by freezing it
(`cash.backends.frame_sharing`). It froze the caller's own data in place --
the result a miss returned, and a frame passed in -- so after a miss
``df["x"].array[0] = v`` raised "assignment destination is read-only" where
plain Python works. Cash now freezes only its own copy.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from cash import Cash  # noqa: E402
from cash.backends import InMemoryBackend  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"x": np.arange(1000, dtype=float), "when": pd.date_range("2024", periods=1000, freq="min")},
        index=pd.Index(np.arange(1000) * 2),
    )


@pytest.fixture(params=["memory", "disk"])
def cache(request, tmp_path):
    if request.param == "memory":
        return Cash(backend=InMemoryBackend(), register_magic=False)
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _write_through_handles(df: pd.DataFrame) -> None:
    column = df["x"]  # a view of the block
    df["x"].array[0] = -1.0
    column.array[1] = -2.0
    df["when"].array[0] = pd.Timestamp("1999-01-01")
    df.index.array[0] = -4
    assert df["x"].iloc[0] == -1.0 and df["x"].iloc[1] == -2.0 and df.index[0] == -4


def test_the_result_of_a_miss_stays_writable(cache):
    @cache.cache
    def build(n):
        return _frame().head(n)

    _write_through_handles(build(500))


def test_a_frame_passed_to_a_miss_stays_writable(cache):
    @cache.cache
    def total(df):
        return float(df["x"].sum())

    df = _frame()
    view = df["x"]
    assert total(df) == float(np.arange(1000).sum())
    _write_through_handles(df)
    assert view.iloc[0] == -1.0


def test_a_write_to_the_result_of_a_miss_reaches_no_hit(cache):
    @cache.cache
    def build():
        return _frame()

    first = build()
    first["x"].array[0] = -1.0
    pd.testing.assert_frame_equal(build(), _frame())
