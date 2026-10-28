"""A function ``@cash.cache`` made is read as the user's function it caches.

Its closure holds cash's own call cache and hit memo, which cash's records
hold too: counted as the user's, a cell that only defined a cached function
searched the heap for their holders and, on a slow machine, ran out of time
(NOTEBOOK-SHARE-UNCHECKED "an object of 'f' ... within 0.1s"). What the
user's own function closes over is still read.
"""

from __future__ import annotations

from cash.notebook.shared_objects import VALUE_TYPES, is_value, share_group


def test_a_cached_function_shares_nothing(disk_cash):
    @disk_cash.cache
    def f(x):
        return x + 1

    f(1)
    ns = {"f": f}
    assert is_value(f, VALUE_TYPES)
    assert share_group(["f"], ns, ns) == ({}, set())


def test_what_the_cached_function_closes_over_is_still_read(disk_cash):
    rows = [1, 2]

    def make():
        @disk_cash.cache
        def f(x):
            return len(rows) + x

        return f

    f = make()
    ns = {"f": f}
    assert not is_value(f, VALUE_TYPES)
    holders, shared = share_group(["f"], ns, ns)
    assert shared == {"f"} and holders == {}
