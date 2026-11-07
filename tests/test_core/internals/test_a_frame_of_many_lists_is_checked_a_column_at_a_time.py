"""The frame memo checks a column of lists a column at a time, not a list
at a time.

To let a frame of lists (``groupby(...).agg(list)``) keep its digest, the
memo checks that each list holds immutable scalars and, at each lookup,
the same objects as when it was hashed. It did so with Python calls per
list: pandas' type inference, a scalar check and a copy of each list. A
notebook cell ``df['l'] = [[1] for _ in range(300_000)]`` then spent about
20 s on those checks across a few statements
(``test_an_unlisted_method_that_changes_the_frame_still_counts``), where
reading the column in full takes 30 ms. The checks are now passes over
all the lists of a column together; their number does not grow with the
rows.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cash.content_hashers import builtin_hash
from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import cow_frame_digest

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)


def _checks(monkeypatch) -> list[str]:
    """Each call of a check that used to run once per list."""
    calls: list[str] = []
    import pandas.api.types as pd_types

    for owner, name in (
        (arg_hashing, "_scalars_of_no_state"),
        (arg_hashing, "_stateful_of"),
        (pd_types, "infer_dtype"),
    ):
        real = getattr(owner, name)

        def counting(*args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(owner, name, counting)
    return calls


def _frame(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame({"a": range(rows)})
    frame["l"] = [[1] for _ in range(rows)]
    frame["t"] = [[pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")] for _ in range(rows)]
    return frame


def _hash_look_change_look(frame: pd.DataFrame) -> None:
    first = cow_frame_digest(frame)
    assert first == builtin_hash(frame)
    assert cow_frame_digest(frame) == first  # served from the memo
    frame["l"].iloc[-1].append(0)
    assert cow_frame_digest(frame) == builtin_hash(frame) != first


def test_the_checks_do_not_grow_with_the_rows(monkeypatch):
    rows = 20_000
    calls = _checks(monkeypatch)
    _hash_look_change_look(_frame(100))
    few = len(calls)
    calls.clear()
    _hash_look_change_look(_frame(rows))
    assert len(calls) == few, f"{len(calls)} checks for {rows} rows, {few} for 100"
    assert few < 40


def test_a_kept_frame_of_lists_is_served_from_the_memo(monkeypatch):
    reads: list = []
    real = arg_hashing.FrameDigests.hash_frame

    def counting(self, obj):
        reads.append(obj.shape)
        return real(self, obj)

    monkeypatch.setattr(arg_hashing.FrameDigests, "hash_frame", counting)
    frame = _frame(5_000)
    first = cow_frame_digest(frame)
    assert all(cow_frame_digest(frame) == first for _ in range(3))
    assert len(reads) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f["t"].iloc[3].append(pd.Timestamp("2022-01-01")),
        lambda f: f["t"].iloc[3].pop(),
        lambda f: f["t"].iloc[3].__setitem__(1, pd.Timestamp("2023-01-01")),
        lambda f: f["l"].iloc[0].__setitem__(0, 2),
        lambda f: setattr(f["t"].iloc[7][0], "tag", 1),
    ],
)
def test_a_list_changed_in_place_after_a_column_wide_check_is_read_again(change):
    frame = _frame(5_000)
    cow_frame_digest(frame)
    assert cow_frame_digest(frame) == builtin_hash(frame)
    change(frame)
    assert cow_frame_digest(frame) == builtin_hash(frame)
