"""A big frame holding text or time-zone dates is read once while copy-on-write proves it unchanged.

The kept digest (`FrameDigests`) refused every block it could not follow to
memory: object columns, Python-stored strings and time-zone-aware dates.
Under pandas 3 without pyarrow, where text is stored as Python strings, that
was every frame with column names (p28-elo-ratings: a 900k-row frame of team
names read again by each call of a 15-call grid, 38 s against 10 s plain).
Such blocks are now followed to the numpy array that holds them, an object
block only while it holds nothing but immutable items. Every write must
still give the digest a fresh read gives, also through ``.values``, which
hands out a Python-string column's own array.
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

ROWS = 100_000


@pytest.fixture
def reads(monkeypatch):
    seen = []
    real = content_hashers.hash_pandas

    def counting(value):
        seen.append(id(value))
        return real(value)

    monkeypatch.setattr(content_hashers, "hash_pandas", counting)
    return seen


@pytest.fixture
def python_strings():
    """Text stored as Python strings, as pandas 3 does without pyarrow."""
    with pd.option_context("mode.string_storage", "python"):
        yield


def _python_text():
    labels = pd.Index([f"r{i}" for i in range(ROWS)], dtype="str")
    return pd.DataFrame(
        {
            "team": pd.Series([f"t{i % 50}" for i in range(ROWS)], dtype="str", index=labels),
            "x": np.arange(ROWS, dtype=float),
        },
        index=labels,
    )


def _object_text():
    return pd.DataFrame(
        {
            "team": pd.Series([f"t{i % 50}" if i % 9 else None for i in range(ROWS)], dtype=object),
            "x": np.arange(ROWS, dtype=float),
        }
    )


def _zoned_dates():
    return pd.DataFrame(
        {"at": pd.date_range("2000", periods=ROWS, freq="s", tz="UTC"), "x": np.arange(ROWS, dtype=float)}
    )


@pytest.mark.usefixtures("python_strings")
@pytest.mark.parametrize("make", [_python_text, _object_text, _zoned_dates])
def test_an_unchanged_frame_is_read_once(reads, make):
    df = make()
    first = compute_hash(df)
    for _ in range(4):
        len(df)
        assert compute_hash(df) == first
    assert reads.count(id(df)) == 1


def test_an_object_column_holding_lists_is_read_every_time(reads):
    df = pd.DataFrame({"cell": pd.Series([[i] for i in range(ROWS)], dtype=object), "x": np.arange(ROWS, dtype=float)})
    compute_hash(df)
    df["cell"].iloc[0].append(7)  # a list in the frame changes; no handle shows it
    compute_hash(df)
    assert reads.count(id(df)) == 2


def _through_values(df):
    df["team"].values[0] = "ZZ"


def _through_array(df):
    df["team"].array[0] = "ZZ"


def _through_index_values(df):
    df.index.values[0] = "ZZ"


def _through_a_kept_handle(df):
    handle = df["team"].values
    compute_hash(df)
    handle[1] = "QQ"


def _loc(df):
    df.loc[df.index[0], "team"] = "ZZ"


def _replace_column(df):
    df["team"] = df["team"].str.upper()


@pytest.mark.usefixtures("python_strings")
@pytest.mark.parametrize(
    "change", [_through_values, _through_array, _through_index_values, _through_a_kept_handle, _loc, _replace_column]
)
def test_every_change_of_python_text_gives_the_digest_a_fresh_read_gives(change):
    df = _python_text()
    before = compute_hash(df)
    assert compute_hash(df) == before
    change(df)
    after = compute_hash(df)
    assert after != before
    assert after == builtin_hash(df.copy(deep=True))


def _object_through_values(df):
    values = df["team"].values
    values.flags.writeable = True
    values[1] = "ZZ"


def _object_iloc(df):
    df.iloc[1, 0] = "ZZ"


@pytest.mark.parametrize("change", [_object_through_values, _object_iloc])
def test_every_change_of_object_text_gives_the_digest_a_fresh_read_gives(change):
    df = _object_text()
    before = compute_hash(df)
    assert compute_hash(df) == before
    change(df)
    after = compute_hash(df)
    assert after != before
    assert after == builtin_hash(df.copy(deep=True))


def test_a_change_of_zoned_dates_gives_the_digest_a_fresh_read_gives():
    df = _zoned_dates()
    before = compute_hash(df)
    assert compute_hash(df) == before
    df["at"].array[0] = pd.Timestamp("1990", tz="UTC")
    after = compute_hash(df)
    assert after != before
    assert after == builtin_hash(df.copy(deep=True))
