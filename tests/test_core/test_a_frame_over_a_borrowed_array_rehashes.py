"""A frame built over an array the caller keeps is hashed afresh each call.

Found while stress-testing the decorator: the content-hash memo
trusts copy-on-write -- a frame can only change by getting new block arrays, so
the block identities are an exact change signal. A write straight through the
numpy array a frame was built from with ``copy=False`` bypasses that: the
blocks are the same objects, the memo answers with the old hash, and the call
was served 10.0 where an uncached run returns 109.0.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash import Cash
from cash.backends import InMemoryBackend
from cash.decorator import arg_hashing


@pytest.fixture
def total():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def total(frame):
        return float(frame.to_numpy().sum())

    return total


def test_a_write_through_the_shared_array_is_seen(total):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = pd.DataFrame(arr, copy=False)
    assert total(frame) == 10.0
    arr[0, 0] = 100.0
    assert total(frame) == 109.0


def test_a_series_over_a_shared_array_is_seen(total):
    arr = np.array([1.0, 2.0, 3.0])
    series = pd.Series(arr, copy=False)
    assert total(series) == 6.0
    arr[0] = 100.0
    assert total(series) == 105.0


def test_an_ordinary_frame_still_uses_the_memo(total):
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert total(frame) == 10.0
    assert total(frame) == 10.0
    assert total(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})) == 10.0


def test_a_check_that_cannot_run_rehashes(total, monkeypatch):
    """If the borrowed-data check itself fails (a pandas internals change),
    the frame is re-hashed. Keeping the memo then served the stale 10.0."""

    def broken(block):
        raise AttributeError("pandas moved its internals")

    monkeypatch.setattr(arg_hashing, "_block_refcount", broken)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    frame = pd.DataFrame(arr, copy=False)
    assert total(frame) == 10.0
    arr[0, 0] = 100.0
    assert total(frame) == 109.0
