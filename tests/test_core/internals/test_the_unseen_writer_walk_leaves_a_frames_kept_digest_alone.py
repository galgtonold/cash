"""The no-trace walk for unseen writers reads a frame's object columns
where pandas holds them.

Taken as Series (``frame[c]``), each column was a view that copy-on-write
counts as a handle a caller could write through, so the frame memo
(`FrameDigests`) dropped the frame's kept digest: with no trace, every pass
of a grid loop over a frame with a text column read it again (24 reads
where 2 were expected). The walk reads the frame's own arrays now.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from cash.notebook.statement.unseen_writes import writers_without_a_trace
from cash.value_hash import compute_hash

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = [pytest.mark.core]

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)

ROWS = 300_000  # above the memo's 1 MiB floor


class Vec:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return Vec(self.v + other)


def _frames():
    yield "object text", pd.DataFrame({"s": pd.Series([f"t{i % 50}" for i in range(ROWS)], dtype=object)})
    yield (
        "text and numbers",
        pd.DataFrame({"s": [f"t{i % 50}" for i in range(ROWS)], "n": np.arange(ROWS)}).astype({"s": object}),
    )


@pytest.mark.parametrize(("name", "df"), list(_frames()), ids=[name for name, _ in _frames()])
def test_each_pass_of_a_loop_reads_the_frame_once(monkeypatch, name, df):
    reads = []
    real = content_hashers.hash_pandas
    monkeypatch.setattr(content_hashers, "hash_pandas", lambda value: reads.append(1) or real(value))
    first = compute_hash(df)
    for _ in range(4):
        assert writers_without_a_trace("y = df['s'].str.len().sum()", {"df": df}) == []
        assert compute_hash(df) == first
    assert len(reads) == 1, f"{name}: read {len(reads)} times"


def test_a_writing_object_in_a_column_is_still_found():
    log = []

    class Logged(Vec):
        def __add__(self, other):
            log.append(other)
            return Logged(self.v + other)

    df = pd.DataFrame({"v": pd.Series([Logged(i) for i in range(3)], dtype=object), "n": [1, 2, 3]})
    found = writers_without_a_trace("w = df['v'].sum()", {"df": df, "log": log})
    assert [name for name, _ in found] == [f"{Logged.__qualname__}.__add__"]
