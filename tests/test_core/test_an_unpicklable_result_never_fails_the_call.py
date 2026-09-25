"""A result a single-tier backend cannot pickle is returned, with a warning.

`FileBackend` and `SQLiteBackend` pickle on the caller's thread. pickle raises
AttributeError for a local object (a lambda in the result) and whatever a
``__reduce__`` raises; only OSError, TypeError, PicklingError and RuntimeError
were caught, so the exception escaped the cached call after the body had run,
and the result was lost. The default tiered stack caught it per tier and hid
the difference.
"""

from __future__ import annotations

import os
import warnings

import pytest

from cash import Cash
from cash.backends import FileBackend, SQLiteBackend
from cash.exceptions import CashCacheStoreFailedWarning


class Handle:
    def __reduce__(self):
        raise ValueError("a Handle cannot be pickled")


@pytest.fixture(params=["file", "sqlite"])
def c(request, tmp_path):
    d = str(tmp_path / ".cash")
    backend = FileBackend(d) if request.param == "file" else SQLiteBackend(os.path.join(str(tmp_path), "c.db"))
    cache = Cash(cache_dir=d, backend=backend, register_magic=False)
    yield cache
    cache.shutdown()


def _call(fn, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args)
    codes = [str(w.message) for w in caught if issubclass(w.category, CashCacheStoreFailedWarning)]
    return result, codes


def test_a_local_function_in_the_result(c):
    @c.cache(assume_safe=True)
    def make_callback(n):
        return {"n": n, "cb": lambda: n}

    result, warned = _call(make_callback, 1)
    assert result["cb"]() == 1
    assert any("STORE-FAILED" in w for w in warned), warned


def test_a_reduce_that_raises(c):
    @c.cache(assume_safe=True)
    def open_handle(n):
        return Handle()

    result, warned = _call(open_handle, 1)
    assert isinstance(result, Handle)
    assert any("STORE-FAILED" in w for w in warned), warned


def test_an_unpicklable_item_in_a_stream(c):
    @c.cache(assume_safe=True, chunk_max_items=2)
    def handles(n):
        for i in range(n):
            yield Handle() if i == 3 else i

    items, warned = _call(lambda: list(handles(5)))
    assert len(items) == 5 and isinstance(items[3], Handle)
    assert any("STORE-CHUNK-FAILED" in w for w in warned), warned
