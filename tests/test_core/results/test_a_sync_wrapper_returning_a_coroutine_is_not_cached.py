"""``@cash.cache`` over a sync wrapper around an ``async def`` says so, once.

A retry or timing decorator written as a plain ``def`` returns the coroutine
of the ``async def`` it wraps, so cash took the sync path and tried to pickle
the coroutine on every call: a STORE-FAILED warning whose fix ("return the
data, not the handle") did not apply, plus two log lines, every call. The
value does not exist until the caller awaits it, so it is not stored, and one
CACHE-RETURNS-AWAITABLE warning names the fix that works.
"""

from __future__ import annotations

import asyncio
import functools
import warnings


def retry(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def test_a_coroutine_returned_by_a_sync_wrapper_is_not_stored_and_warns_once(disk_cash):
    calls = []

    @disk_cash.cache(assume_safe=True)
    @retry
    async def fetch(x):
        calls.append(x)
        return x * 2

    async def main():
        return [await fetch(3), await fetch(3)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert asyncio.run(main()) == [6, 6]

    codes = [str(w.message) for w in caught if "[" in str(w.message)]
    assert len(calls) == 2
    assert len([c for c in codes if "CACHE-RETURNS-AWAITABLE" in c]) == 1, codes
    assert not [c for c in codes if "STORE-FAILED" in c], codes


def test_the_suggested_fix_caches(disk_cash):
    calls = []

    @retry
    @disk_cash.cache(assume_safe=True)
    async def fetch(x):
        calls.append(x)
        return x * 2

    async def main():
        return [await fetch(3), await fetch(3)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert asyncio.run(main()) == [6, 6]
    assert calls == [3]
    assert not [w for w in caught if "CACHE-RETURNS-AWAITABLE" in str(w.message)]
