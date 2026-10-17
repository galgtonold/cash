"""A table cash is writing in the background is not read again for its digest.

A store hands a table's large arrays to the background write as views of
their own memory (`serialize_split` with ``copy=False``). The frame memo
read each view as a handle someone else holds, so while a 50 MB result of a
cached call was being written, the checks right after it read the whole
table again: twice per pass of p04's grid loop, ~0.4 s each. Cash's own
views are counted as cash's (`cash._cash_views`); any other view still
makes the memo read the table.
"""

from __future__ import annotations

import gc

import pytest

from cash import _cash_views
from cash.backends.serialization import PickleSerializer
from cash.decorator import arg_hashing

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)


@pytest.fixture(autouse=True)
def empty_memo():
    arg_hashing.FRAME_DIGESTS.clear()
    yield
    arg_hashing.FRAME_DIGESTS.clear()


def _table() -> pd.DataFrame:
    return pd.DataFrame(np.random.default_rng(0).random((2_000, 200)))


def test_a_table_being_written_keeps_its_digest():
    df = _table()
    digest = arg_hashing.cow_frame_digest(df)
    _stream, buffers = PickleSerializer().serialize_split(df, copy=False)
    assert buffers, "the table's block goes out of band"
    assert arg_hashing.FRAME_DIGESTS.lookup(df) == digest
    del buffers
    gc.collect()
    assert all(_cash_views.held(block.values) == 0 for block in df._mgr.blocks)
    assert arg_hashing.FRAME_DIGESTS.lookup(df) == digest


def test_a_view_of_someone_elses_still_reads_it_again():
    """Control: a view cash did not note is a handle."""
    df = _table()
    arg_hashing.cow_frame_digest(df)
    held = memoryview(df._mgr.blocks[0].values)
    assert arg_hashing.FRAME_DIGESTS.lookup(df) is None
    del held
