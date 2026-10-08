"""The arguments of a ``dynamic_depends_on=`` call inside a cached caller are
not kept alive for good.

Each such call was kept, with its arguments, for as long as the process ran,
so a service passing arrays through the loader grew by every array: 614 MB
after 80 calls with an 8 MB array each. What is kept for asking the resolver
again is bounded now.
"""

from __future__ import annotations

import gc
import weakref

import cash


class _Version(cash.DataSource):
    def get_id(self):
        return "version"

    def state_token(self):
        return "1"


class _Big:
    """An argument that is weak-referenceable and pickles to ~4 MB."""

    def __init__(self, i):
        self.i = i
        self.payload = bytes(4 << 20)


def test_old_arguments_are_freed(cash_instance):
    @cash_instance.cache(dynamic_depends_on=lambda big: _Version())
    def summarize(big):
        return big.i

    @cash_instance.cache
    def report(i):
        big = _Big(i)
        refs.append(weakref.ref(big))
        return summarize(big) + 1

    refs: list = []
    for i in range(40):
        assert report(i) == i + 1
    gc.collect()
    alive = sum(ref() is not None for ref in refs)
    assert alive < 30, f"{alive} of 40 arguments are still alive"
    assert report(39) == 40
