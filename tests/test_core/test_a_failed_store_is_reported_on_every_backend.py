"""A backend that cannot store a result says so, tiered or not.

Found while attacking the decorator before round 26: with the default (tiered)
backend an unpicklable result produced only a ``logger.warning`` line --
no ``STORE-FAILED`` warning and an empty ``cache_info()['warnings']``, while
the same code on a ``FileBackend`` reported both. The STORE-FAILED page says
cash "deliberately reports it rather than raising it into your code", and
``cache_info()['warnings']`` is documented as the way to discover silent
misbehaviour after the fact.
"""
from __future__ import annotations

import threading
import time
import warnings

import pytest

from cash import Cash
from cash.backends.file_backend import FileBackend
from cash.exceptions import CashCacheStoreFailedWarning


def _unpicklable_maker(cash):
    @cash.cache
    def make_lock(n):
        time.sleep(0.3)      # past the persistence floor: the disk tier is asked
        return {"n": n, "lock": threading.Lock()}

    return make_lock


@pytest.mark.parametrize("kind", ["tiered", "file"])
def test_an_unstorable_result_is_reported(tmp_path, kind):
    cache_dir = str(tmp_path / kind)
    cash = (Cash(cache_dir=cache_dir, register_magic=False) if kind == "tiered"
            else Cash(backend=FileBackend(cache_dir=cache_dir), register_magic=False))
    make_lock = _unpicklable_maker(cash)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert make_lock(1)["n"] == 1
    store_failed = [w for w in seen if issubclass(w.category, CashCacheStoreFailedWarning)]
    assert store_failed, [str(w.message) for w in seen]
    assert "STORE-FAILED" in str(store_failed[0].message)
    codes = [w.get("code") for w in (make_lock.cache_info().get("warnings") or [])]
    assert "STORE-FAILED" in codes, codes


def test_a_storable_result_reports_nothing(tmp_path):
    cash = Cash(cache_dir=str(tmp_path / "ok"), register_magic=False)

    @cash.cache
    def fine(n):
        return n * 2

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert fine(2) == 4
    assert not [w for w in seen if issubclass(w.category, CashCacheStoreFailedWarning)]
    assert not (fine.cache_info().get("warnings") or [])
