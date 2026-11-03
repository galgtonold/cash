"""A frame's object column of records holding dates is hashed by what they
hold, written as tagged bytes (`cash.dated_records`), not pickled a
``datetime`` at a time: student_8's 87,000 sessions of ``(action,
datetime)`` records took 3.5 s to hash after a 0.2 s ``pd.DataFrame(logs)``.

The bytes are the same from the C kernel and from the Python reference,
equal content hashes alike, and any change of a value, a type, the fold or
the nesting hashes apart."""

from __future__ import annotations

import datetime
import random

import pytest

from cash import _native, dated_records
from cash.content_hashers import builtin_hash
from tests._work_counts import pickled_bytes

pd = pytest.importorskip("pandas")

T0 = datetime.datetime(2020, 1, 1)


def _sessions(n: int = 300, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [
        [(f"Action_{rng.randrange(11)}", T0 + datetime.timedelta(seconds=rng.randrange(10**7))) for _ in range(5)]
        for _ in range(n)
    ]


def _frame(sessions: list) -> pd.DataFrame:
    return pd.DataFrame({"user": [f"u{i % 7}" for i in range(len(sessions))], "actions": sessions})


def test_a_column_of_dated_records_is_hashed_without_pickling_them():
    frame = _frame(_sessions(2_000))
    with pickled_bytes() as count:
        assert builtin_hash(frame) is not None
    # The two million records of the trace pickled 90 MB; here 2,000
    # sessions of five records would pickle 200 KB.
    assert count.bytes < 20_000


def test_equal_columns_hash_alike_and_any_change_hashes_apart():
    base = builtin_hash(_frame(_sessions()))
    assert builtin_hash(_frame(_sessions())) == base
    changes = []
    one = _sessions()
    one[17][2] = (one[17][2][0], one[17][2][1] + datetime.timedelta(microseconds=1))
    changes.append(one)
    one = _sessions()
    one[17][2] = (one[17][2][0], one[17][2][1].replace(fold=1))
    changes.append(one)
    one = _sessions()
    one[17][2] = (one[17][2][0], one[17][2][1].date())
    changes.append(one)
    one = _sessions()
    one[17][2] = [one[17][2][0], one[17][2][1]]
    changes.append(one)
    one = _sessions()
    one[17][2] = (one[17][2][0] + "x", one[17][2][1])
    changes.append(one)
    one = _sessions()
    one[17].append(())
    changes.append(one)
    digests = {builtin_hash(_frame(sessions)) for sessions in changes}
    assert base not in digests
    assert len(digests) == len(changes)


def _odd_values() -> list:
    return [
        [
            (
                "é\ud800x",
                "",
                1,
                -(2**63),
                2**63 - 1,
                1.5,
                -0.0,
                float("inf"),
                True,
                False,
                None,
                b"\x00z",
                datetime.date(1, 1, 1),
                datetime.datetime(9999, 12, 31, 23, 59, 59, 999_999, fold=1),
                [(), []],
            )
        ],
        [],
        ("😀", T0),
    ]


@pytest.mark.parametrize(
    ("items", "written"),
    [
        (_sessions(), True),
        (_odd_values(), True),
        ([[("a", 2**63)], [("b", T0)]], False),  # past 64 bits: pickled
        ([[("a", T0.replace(tzinfo=datetime.timezone.utc))]], False),  # aware: pickled
        ([["a"], ["b"]], False),  # no date: pickled
        ([["a"]] * 70 + [[T0]], False),  # no date among the first items: pickled
        ([[[[[[[[T0]]]]]]]], True),  # as deep as read
        ([[[[[[[[[T0]]]]]]]]], False),  # deeper: pickled
        ([[(1, (2, (3, (4, (5, (6, (T0,)))))))]], False),
        ([[{"a": T0}]], False),  # a dict: pickled
        ([[pd.Timestamp(T0)]], False),  # a subclass: pickled
    ],
)
def test_the_kernel_writes_what_the_python_reference_writes(items, written):
    python = dated_records.python_dated_records_bytes(items)
    assert (python is not None) is written
    assert dated_records.dated_records_bytes(items) == python
    if _native.kernels is not None:
        assert _native.kernels.dated_records(items, dated_records.PROBE, dated_records.LEVELS) == python


def test_what_is_written_is_what_the_values_are():
    written = dated_records.python_dated_records_bytes([[("a", datetime.datetime(2020, 2, 3, 4, 5, 6, 7))]])
    assert written == (
        b"[" + (1).to_bytes(8, "little")
        + b"[" + (1).to_bytes(8, "little")
        + b"(" + (2).to_bytes(8, "little")
        + b"s" + (1).to_bytes(8, "little") + b"a"
        + b"D" + bytes((0x07, 0xE4, 2, 3, 4, 5, 6, 0, 0, 7))
    )  # fmt: skip
