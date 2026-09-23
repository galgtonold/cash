"""A session-long memo stays within its size, dropping what was used least recently.

The notebook's memos are keyed by code object, statement or cache entry, and
a long session with many redefined cells keeps producing new keys.
"""

import pytest

from cash.notebook._memo import LruMemo
from cash.notebook.statement.amplification import AmplificationGuard


def test_the_least_recently_used_entry_is_dropped():
    memo = LruMemo(2)
    memo["a"] = 1
    memo["b"] = 2
    assert memo.get("a") == 1  # "a" is now the most recently used
    memo["c"] = 3

    assert len(memo) == 2
    assert "b" not in memo
    assert memo.get("a") == 1
    assert memo.get("c") == 3


def test_a_miss_returns_the_default():
    memo = LruMemo(1)
    assert memo.get("x") is None
    assert memo.get("x", 0) == 0


def test_a_memo_needs_room_for_one_entry():
    with pytest.raises(ValueError):
        LruMemo(0)


def test_the_amplification_guard_tracks_a_bounded_number_of_statements():
    guard = AmplificationGuard()
    assert guard._bytes_by_stmt.maxsize == AmplificationGuard.MAX_STATEMENTS
    assert guard._warned.maxsize == AmplificationGuard.MAX_STATEMENTS
