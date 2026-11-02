"""A frame given a new column in place reads only that column for its digest.

student_8: ``df["action_name"] = df["action"].str[0]`` on two million
exploded ``(action, time)`` records. The frame's entry in the memo no longer
matched it, so its next digest (the value cutoff takes one per statement)
read every unchanged column again, 3.5 s, though their memory was the very
memory the entry had vouched for and still held. Now an entry whose frame
changed lends the bytes of each column whose memory it kept, once nothing
outside pandas can have written that memory and the lists and records it
kept are the same (`_memory_kept`). The digest is the one a full read gives.
"""

from __future__ import annotations

import datetime

import pytest

from cash import content_hashers
from cash.decorator import arg_hashing
from tests._work_counts import hashed_bytes

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

ROWS = 200_000


@pytest.fixture(autouse=True)
def empty_memo():
    arg_hashing.FRAME_DIGESTS.clear()
    yield
    arg_hashing.FRAME_DIGESTS.clear()


def _records() -> pd.DataFrame:
    t0 = datetime.datetime(2020, 1, 1)
    records = [(f"Action_{n % 7}", t0 + datetime.timedelta(seconds=n)) for n in range(ROWS)]
    rng = np.random.default_rng(0)
    return pd.DataFrame({"session": rng.integers(0, 5000, ROWS), "record": records, "x": rng.random(ROWS)})


def _numbers() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"a": rng.random(ROWS), "b": rng.integers(0, 9, ROWS), "c": rng.random(ROWS)})


def _lists(rows: int = ROWS // 10) -> pd.DataFrame:
    """Lists are checked item by item on every lookup: fewer rows."""
    rng = np.random.default_rng(2)
    return pd.DataFrame({"a": rng.random(rows), "l": [[n, n + 1] for n in range(rows)]})


def _full(frame: pd.DataFrame) -> str:
    return content_hashers.builtin_hash(frame)


@pytest.mark.parametrize("make", [_records, _numbers, _lists], ids=["records", "numbers", "lists"])
def test_a_new_column_is_the_only_one_read(make):
    df = make()
    with hashed_bytes() as first:
        assert arg_hashing.cow_frame_digest(df) == _full(df)
    df["n"] = 2
    with hashed_bytes() as again:
        digest = arg_hashing.cow_frame_digest(df)
    assert digest == _full(df)
    assert again.bytes < 0.3 * first.bytes, (again.bytes, first.bytes)


def _list_item(df):
    df["l"].iloc[0].append(9)


def _handle(df):
    view = df["a"].to_numpy()
    view.flags.writeable = True
    view[0] = 1234.5


CHANGES = {
    "cell": lambda df: df.loc.__setitem__((0, "a"), -1.0),
    "column": lambda df: df.__setitem__("a", df["a"] + 1),
    "list in place": _list_item,
    "raw handle": _handle,
    "taken, not written": lambda df: df["a"].to_numpy(),
    "sort": lambda df: df.sort_values("a", inplace=True),
    "drop": lambda df: df.drop(columns=["a"], inplace=True),
}


@pytest.mark.parametrize("change", list(CHANGES.values()), ids=list(CHANGES))
def test_a_column_changed_after_the_new_one_is_read_again(change):
    df = _lists()
    arg_hashing.cow_frame_digest(df)
    df["n"] = 2
    before = arg_hashing.cow_frame_digest(df)
    try:
        change(df)
    except ValueError:  # pragma: no cover - numpy refusing writeable=True is fine too
        pytest.skip("this numpy does not let the view be made writable")
    df["m"] = 3
    assert arg_hashing.cow_frame_digest(df) == _full(df)
    assert arg_hashing.cow_frame_digest(df) != before


def test_a_frame_sharing_a_changed_frames_columns_digests_their_content():
    df = _records()
    arg_hashing.cow_frame_digest(df)
    df["n"] = 2
    view = df[["session", "record"]]
    df.loc[0, "session"] = -7
    for frame in (view, df):
        assert arg_hashing.cow_frame_digest(frame) == _full(frame)
    assert view.loc[0, "session"] != -7
