"""Within a statement, a big polars frame is read once while its buffers stay where they were.

`ArgFingerprints` keeps a polars digest by `polars_signature` (schema, height
and every buffer's address and length) with a ``clone()`` held, so that a
write through polars must move a buffer instead of writing where the old
one lay. Every write polars offers must give the digest a fresh read gives;
and nothing is kept from one statement to the next, since a frame built
over a numpy array without a copy changes with that array.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from cash.content_hashers import builtin_hash, polars_signature
from cash.notebook.call_effects import ArgFingerprints

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")

ROWS = 100_000  # ~1.9 MB: above the memo's floor


@pytest.fixture
def reads(monkeypatch):
    seen = []
    real = content_hashers.hash_polars

    def counting(value, *a, **k):
        seen.append(id(value))
        return real(value, *a, **k)

    monkeypatch.setattr(content_hashers, "hash_polars", counting)
    return seen


def _frame():
    return pl.DataFrame(
        {
            "a": np.arange(ROWS, dtype=np.float64) * 1.5,
            "s": [f"x{i % 7}" for i in range(ROWS)],
            "n": pl.Series([None if i % 5 == 0 else i for i in range(ROWS)], dtype=pl.Int64),
        }
    )


def test_an_unchanged_frame_is_read_once_in_a_statement(reads):
    df = _frame()
    memo = ArgFingerprints()
    memo.begin_statement()
    first = memo.digest(df)
    for _ in range(3):
        df.select(pl.col("a").sum())
        assert memo.digest(df) == first
    assert reads.count(id(df)) == 1
    memo.begin_statement()
    assert memo.digest(df) == first
    assert reads.count(id(df)) == 2, "a digest was carried into the next statement"


def test_a_small_frame_is_read_every_time(reads):
    df = pl.DataFrame({"a": [1.0, 2.0]})
    memo = ArgFingerprints()
    for _ in range(3):
        memo.digest(df)
    assert reads.count(id(df)) == 3


def _cell(df):
    df[0, "a"] = -1.0


def _text_cell(df):
    df[1, "s"] = "ZZ"


def _replace_column(df):
    df.replace_column(0, pl.Series("a", np.zeros(ROWS)))


def _insert_column(df):
    df.insert_column(0, pl.Series("q", np.zeros(ROWS)))


def _drop_in_place(df):
    df.drop_in_place("s")


def _rename(df):
    df.columns = ["A", "S", "N"]


def _extend(df):
    df.extend(df.head(3))


def _vstack(df):
    df.vstack(df.head(3), in_place=True)


def _hstack(df):
    df.hstack([pl.Series("z", np.zeros(ROWS))], in_place=True)


def _scatter(df):
    df.get_column("a").scatter([0], [99.0])


def _series_item(df):
    column = df["n"]
    column[3] = 7


@pytest.mark.parametrize(
    "change",
    [
        _cell,
        _text_cell,
        _replace_column,
        _insert_column,
        _drop_in_place,
        _rename,
        _extend,
        _vstack,
        _hstack,
        _scatter,
        _series_item,
    ],
)
def test_every_change_gives_the_digest_a_fresh_read_gives(change):
    df = _frame()
    memo = ArgFingerprints()
    memo.begin_statement()
    before = memo.digest(df)
    change(df)
    after = memo.digest(df)
    assert after == builtin_hash(df.clone())
    assert after != before or builtin_hash(df) == before


def test_a_frame_with_python_objects_has_no_signature():
    df = pl.DataFrame({"o": pl.Series([object() for _ in range(3)], dtype=pl.Object)})
    assert polars_signature(df) is None


@pytest.mark.parametrize("write", ["scatter", "item"])
def test_a_series_written_where_it_lies_gives_a_fresh_digest(write):
    """A series owning its buffer alone is written where it lies: only the
    clone the memo holds makes polars copy it first, so its address moves."""
    series = pl.Series("v", np.arange(ROWS * 3, dtype=np.float64)) * 1.0
    memo = ArgFingerprints()
    memo.begin_statement()
    before = memo.digest(series)
    if write == "scatter":
        series.scatter([0], [99.0])
    else:
        series[0] = 99.0
    after = memo.digest(series)
    assert after != before
    assert after == builtin_hash(series.clone())


def test_a_write_moves_the_signature_while_a_clone_is_held():
    """Without the clone a series owning its buffer alone is written where it
    lies, and its signature stays the same."""
    series = pl.Series("v", np.arange(ROWS, dtype=np.float64)) * 1.0
    held = series.clone()
    signature = polars_signature(series)
    series.scatter([0], [99.0])
    assert polars_signature(series) != signature
    assert held[0] == 0.0
