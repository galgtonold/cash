"""Review of claude/launch-bar-speed-s1n2lu 98a6a77 (polars columns folded side
by side).

Put this file in tests/test_core/internals/.

Keys are byte-identical (checked on frames with ints, floats, strings,
booleans, datetimes, categoricals, enums, nulls, lists, structs, decimals,
binary, a slice and a two-chunk concat). But a column that is not of a
fixed, boolean, offsets or view layout (a list, a struct, a decimal) is
folded as an Arrow IPC stream written straight into the hash
(`_fold_arrow_ipc`). On the side-by-side path the hash is a `_Recorded`,
which keeps a copy of every byte of that stream until all columns are
done: a 3M-row list column took 97 MB more Python memory (1 MB on one
thread) and was slower (0.35 s vs 0.25 s). The copy grows with the
column, so a large nested column costs its whole size again in RAM.

Suggested fix: fold side by side only the columns whose fold sends digests
to *h* (fixed, bool, offsets, view), and fold an IPC column on the calling
thread in its turn; or keep one thread when any column goes through IPC.
"""

from __future__ import annotations

import tracemalloc

import pytest

from cash import content_hashers
from cash.bulk_digest import hashing_threads

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")


@pytest.mark.skipif(hashing_threads() < 2, reason="the side-by-side path needs two hashing threads")
def test_a_big_list_column_is_not_copied_while_its_frame_is_hashed():
    n = 1_000_000
    frame = pl.DataFrame({"lst": [[i, i + 1, i + 2] for i in range(n)], "x": np.arange(n)})
    assert content_hashers._rows_per_chunk(frame) >= content_hashers.PARALLEL_COLUMNS_MIN_ROWS
    content_hashers.hash_polars(frame.head(10))  # imports and first-use set-up outside the measure
    tracemalloc.start()
    try:
        content_hashers.hash_polars(frame)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 << 20, f"hashing held {peak / 1e6:.0f} MB: the list column's IPC stream, copied"
