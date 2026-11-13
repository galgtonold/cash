"""A global that holds a ``weakref.proxy`` whose object is gone still keys.

Every read of a dead proxy raises ReferenceError, an ``isinstance`` too.
Library objects keep them (a compiled function's dispatcher pointing back
at an owner that was collected), and the key build met one while hashing
the global or searching a library object for user code: KEY-BUILD-FAILED,
and the function never cached. A dead proxy refers to nothing, so it holds
nothing a result depends on; it keys as one fixed form.
"""

from __future__ import annotations

import argparse
import types
import warnings
import weakref

from cash import CashCacheIneffectiveWarning


class _Owner:
    pass


def _dead_proxy():
    owner = _Owner()
    proxy = weakref.proxy(owner)
    del owner
    return proxy


#: Hashed as data: the function reads ``record.n``.
record = types.SimpleNamespace(back=_dead_proxy(), n=1)

#: An object of a library's class, searched for user code it may hold.
options = argparse.Namespace(owner=_dead_proxy(), scale=3)


def reads_record(x):
    return x + record.n


def reads_options(x):
    return x * options.scale


def _hits_on_the_second_call(disk_cash, func):
    f = disk_cash.cache(func)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CashCacheIneffectiveWarning)
        first = f(2)
        assert f.explain(2).reason == "hit"
        assert f(2) == first


def test_a_dead_proxy_in_data_a_function_reads_is_keyed(disk_cash):
    _hits_on_the_second_call(disk_cash, reads_record)


def test_a_dead_proxy_in_a_library_object_is_keyed(disk_cash):
    _hits_on_the_second_call(disk_cash, reads_options)


def test_a_live_proxy_still_keys_on_what_it_refers_to(disk_cash, monkeypatch):
    """The positive control: a live proxy keys as its object, so a change
    to that object reaches the key."""
    owner = _Owner()
    owner.n = 1
    monkeypatch.setattr(record, "back", weakref.proxy(owner))
    f = disk_cash.cache(reads_record)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = f.explain(2).cache_key
        owner.n = 2
        assert f.explain(2).cache_key != before
