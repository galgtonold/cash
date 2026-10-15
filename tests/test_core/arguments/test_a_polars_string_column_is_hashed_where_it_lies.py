"""A polars string column is hashed from the views polars holds.

The hash exported each string and binary column at Arrow's oldest compat
level, which copies every byte of it into a new ``large_string`` column,
and then built an array of each value's length: 0.2 s and 80 MB of new
memory for each 10M-row column. With the C extension it reads the views
where they are, and the digest is byte for byte the one the export gave,
so every key made before still matches.
"""

import datetime as dt

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from cash import _native, content_hashers

pytestmark = pytest.mark.skipif(_native.kernels is None, reason="no C extension")

WORDS = ["", "a", "é", "x" * 12, "y" * 13, "z" * 100, None, "ünïcödé" * 5]


def _frame(n=3000):
    rng = np.random.default_rng(0)
    words = [WORDS[i] for i in rng.integers(0, len(WORDS), n)]
    return pl.DataFrame(
        {
            "s": words,
            "i": rng.integers(0, 9, n),
            "f": rng.random(n),
            "b": rng.random(n) > 0.5,
            "bin": [w.encode() if w is not None else None for w in words],
            "d": [dt.date(2020, 1, 1)] * n,
            "l": [[w] for w in words],
            "c": pl.Series(words).cast(pl.Categorical),
            "st": [{"a": w} for w in words],
        }
    )


def _exported(value):
    """The digest as the export at the oldest compat level gives it."""
    kernels = _native.kernels
    _native.kernels = None
    try:
        return content_hashers.hash_polars(value)
    finally:
        _native.kernels = kernels


@pytest.mark.parametrize(
    "make",
    [
        lambda: _frame(),
        lambda: _frame().slice(17, 2000),
        lambda: _frame()["s"],
        lambda: _frame()["bin"].slice(5, 100),
        lambda: pl.concat([_frame(), _frame().slice(3, 50)], rechunk=False),
        lambda: _frame().filter(pl.col("s").is_not_null()),
        lambda: _frame().head(0),
        lambda: pl.DataFrame({"s": pl.Series([None] * 10, dtype=pl.String)}),
        lambda: pl.DataFrame({"s": ["q" * (20 << 20), "r" * 3, "s" * (9 << 20)]}),
    ],
    ids=["frame", "slice", "series", "binary-slice", "chunks", "no-nulls", "empty", "all-missing", "long-values"],
)
def test_the_digest_is_the_one_the_export_gave(make, monkeypatch):
    value = make()
    exported = _exported(value)
    # Read where it lies: no string column is exported.
    monkeypatch.setattr(content_hashers, "_polars_to_arrow", _refuse_strings(content_hashers._polars_to_arrow))
    assert content_hashers.hash_polars(value) == exported


def _refuse_strings(export):
    def checked(frame, pl):
        assert not any(dtype in (pl.String, pl.Binary) for dtype in frame.dtypes), frame.schema
        return export(frame, pl)

    return checked


def test_a_value_split_across_blocks_is_read_whole(monkeypatch):
    monkeypatch.setattr(content_hashers, "_VIEW_BLOCKS", type(content_hashers._VIEW_BLOCKS)())
    monkeypatch.setattr(content_hashers, "VIEW_BLOCK_BYTES", 16)
    value = pl.Series("s", ["abcdefghijklmnopq", None, "", "rs", "t" * 30])
    assert content_hashers.hash_polars(value) == _exported(value)


def test_a_changed_byte_moves_the_digest():
    one = pl.Series("s", ["a" * 40, "b"])
    other = pl.Series("s", ["a" * 39 + "c", "b"])
    assert content_hashers.hash_polars(one) != content_hashers.hash_polars(other)
