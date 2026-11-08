"""A function that changes its argument in place only on some calls is
warned about by what a call actually did.

Reading ``if flag: a.sort()`` finds a change to ``a``; whether a call that
makes it runs every time depends on what ``a`` is, which only a call that
sorts shows. A first call that did not sort said the call "is not stored
and runs every time", and the next call, which sorted and was stored, said
a hit writes ``a`` back: two warnings that contradict each other.
"""

from __future__ import annotations

import warnings


def _branchy(a, flag):
    if flag:
        a.sort()
    return 0


def test_only_the_line_saying_a_hit_writes_it_back(disk_cash):
    cached = disk_cash.cache(_branchy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached([3, 1, 2], False)
        for _ in range(3):
            a = [3, 1, 2]
            cached(a, True)
            assert a == [1, 2, 3]
    texts = [str(w.message) for w in caught if "IMPURE" in str(w.message)]
    assert cached.cache_info()["hits"] == 1
    assert not any("runs every time" in t for t in texts), texts
    assert any("a hit writes it back" in t for t in texts), texts


def test_a_call_that_changes_it_where_no_hit_can_write_back_is_still_warned(disk_cash):
    """A tuple holding a list is not written back: the finding stands."""

    def nested(a, flag):
        if flag:
            a[0].sort()
            a[0].append(0)
        return 0

    cached = disk_cash.cache(nested)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(([3, 1],), False)
        assert not [w for w in caught if "IMPURE" in str(w.message)]
        cached(([3, 1],), True)
    texts = [str(w.message) for w in caught if "IMPURE" in str(w.message)]
    assert any("runs every time" in t for t in texts), texts
