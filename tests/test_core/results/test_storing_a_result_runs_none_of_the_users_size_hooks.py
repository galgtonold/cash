"""Storing a ``@cash.cache`` result runs none of the user's size hooks.

Before a result goes into the RAM tier or the store, cash weighs it: it read
an ``nbytes`` and called ``sys.getsizeof``. On a class of the user's that ran
its ``nbytes`` property, its ``__getattr__`` (asked for ``nbytes``) or its
``__sizeof__``, side effects and all, on every store. A size is only a guess:
cash reads the one it can without calling the user.
"""

from __future__ import annotations

import pytest

CALLS: list[str] = []


class NbytesProperty:
    @property
    def nbytes(self):
        CALLS.append("nbytes")
        return 8


class GetattrFallback:
    def __getattr__(self, name):
        if not name.startswith("__"):
            CALLS.append(name)
        raise AttributeError(name)


class Sizeof:
    def __sizeof__(self):
        CALLS.append("sizeof")
        return 64


@pytest.mark.parametrize("kind", [NbytesProperty, GetattrFallback, Sizeof], ids=lambda k: k.__name__)
@pytest.mark.parametrize("cache_name", ["cash_instance", "disk_cash"])
def test_a_stored_result_is_weighed_without_its_hooks(request, kind, cache_name):
    cache = request.getfixturevalue(cache_name)

    @cache.cache
    def build(n):
        return [kind(), list(range(n))]

    CALLS.clear()
    first = build(3)
    second = build(3)
    assert isinstance(first[0], kind) and second[1] == [0, 1, 2]
    assert CALLS == [], f"cash ran the user's {CALLS} while storing the result"


class DataFrame:
    """Named like pandas' own: sized by name, its methods were run."""

    def memory_usage(self, deep=False):
        CALLS.append("memory_usage")
        return 10

    @property
    def _mgr(self):
        CALLS.append("_mgr")


class csr_matrix:
    """Named like scipy's own."""

    @property
    def data(self):
        CALLS.append("data")
        return b""


@pytest.mark.parametrize("kind", [DataFrame, csr_matrix], ids=lambda k: k.__name__)
@pytest.mark.parametrize("cache_name", ["cash_instance", "disk_cash"])
def test_a_result_named_like_a_librarys_type_is_not_sized_as_one(request, kind, cache_name):
    cache = request.getfixturevalue(cache_name)

    @cache.cache
    def build(n):
        return [kind(), n]

    CALLS.clear()
    build(3)
    build(3)
    assert CALLS == [], f"cash ran the user's {CALLS} while storing the result"
