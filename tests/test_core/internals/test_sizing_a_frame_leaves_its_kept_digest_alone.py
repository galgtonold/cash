"""Sizing a frame reads its arrays where pandas holds them.

`pandas_nbytes` asked each column for ``.array``, a handle the frame memo
(`FrameDigests`) must take as one a caller could write through, so every
size check cost a frame with a text or time-zone column its kept
digest; and it boxed text with ``to_numpy(dtype=object)``, which looks for
missing values and copies: 75 ms per 900,000-row column, 11 s over the
p28-elo-ratings grid cell. Both are read from the held array now.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from cash.sizing import pandas_nbytes
from cash.value_hash import compute_hash

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = [pytest.mark.core]

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)

ROWS = 300_000  # every frame above the memo's 1 MiB floor


def _frames():
    with pd.option_context("mode.string_storage", "python"):
        yield "python text", pd.DataFrame({"s": pd.Series([f"t{i % 50}" for i in range(ROWS)], dtype="str")})
    yield "object text", pd.DataFrame({"s": pd.Series([f"t{i % 50}" for i in range(ROWS)], dtype=object)})
    yield "zoned dates", pd.DataFrame({"t": pd.date_range("2000", periods=ROWS, freq="s", tz="UTC")})


@pytest.mark.parametrize(("name", "df"), list(_frames()), ids=[name for name, _ in _frames()])
def test_a_size_check_between_two_digests_does_not_cost_a_read(monkeypatch, name, df):
    exact = int(df.copy(deep=True).memory_usage(deep=True).sum())  # memory_usage gives out handles
    reads = []
    real = content_hashers.hash_pandas
    monkeypatch.setattr(content_hashers, "hash_pandas", lambda value: reads.append(1) or real(value))
    first = compute_hash(df)
    assert pandas_nbytes(df) == pytest.approx(exact, rel=0.03)
    assert compute_hash(df) == first
    assert len(reads) == 1, f"{name}: read {len(reads)} times"


def test_text_is_sized_without_a_boxed_copy(monkeypatch):
    with pd.option_context("mode.string_storage", "python"):
        df = pd.DataFrame({"s": pd.Series([f"t{i % 50}" for i in range(ROWS)], dtype="str")})
    expected = int(df.memory_usage(deep=True).sum())

    def no_copy(*_a, **_k):
        raise AssertionError("to_numpy copied the column")

    monkeypatch.setattr(pd.Series, "to_numpy", no_copy)
    assert pandas_nbytes(df) == pytest.approx(expected, rel=0.03)
