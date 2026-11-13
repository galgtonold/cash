"""A function that changes its argument in place only on some calls is
warned about by what a call actually did.

Reading ``if flag: a.sort()`` finds a change to ``a``; whether a call that
makes it runs every time depends on what ``a`` is, which only a call that
sorts shows. A first call that did not sort said the call "is not stored
and runs every time", and the next call, which sorted and was stored, said
a hit writes ``a`` back: two warnings that contradict each other.
"""

from __future__ import annotations

import sys
import warnings

import pytest

needs_trace = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="what a call ran is recorded with sys.monitoring, Python 3.12+; below it a call that changes its "
    "argument in place runs every time, with a warning",
)


def _branchy(a, flag):
    if flag:
        a.sort()
    return 0


@needs_trace
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


def test_an_argument_no_hit_can_write_back_is_warned_about_at_once(disk_cash):
    """A tuple holding a list is not written back: no later call can make
    the finding wrong, so it is said on the first call, changed or not."""

    def nested(a, flag):
        if flag:
            a[0].sort()
        return 0

    cached = disk_cash.cache(nested)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(([3, 1],), False)
    texts = [str(w.message) for w in caught if "IMPURE" in str(w.message)]
    assert any("runs every time" in t for t in texts), texts


class _Keyed:
    def __init__(self):
        self.items = []

    def __cash_key__(self):
        return "v1"


def test_an_argument_no_hit_writes_back_is_warned_about_on_the_first_call(disk_cash):
    """Its key hashes alike before and after the body, so no call is seen
    changing it: waiting for one held the finding forever, and nothing
    said that a hit skips the change."""

    def add(k):
        k.items.append(1)
        return 0

    cached = disk_cash.cache(add)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(_Keyed())
    texts = [str(w.message) for w in caught if "IMPURE" in str(w.message)]
    assert any("'k' in place" in t for t in texts), texts
