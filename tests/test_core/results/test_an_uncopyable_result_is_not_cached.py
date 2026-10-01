"""A result cash cannot copy is not cached at all.

Found while stress-testing the decorator: the RAM tier deep-copies
what it stores so a caller mutating a hit cannot poison later hits. When the
copy fails -- a result holding a lock, a socket, a connection -- it fell back to
storing and returning the object ITSELF. Measured: the caller appended to the
returned dict and the next call returned the same object, mutation included,
and four threads all received that one object. The docs promise the opposite
("a hit returns a copy"; "the RAM tier deep-copies every value it stores").

Isolation is what makes a cache safe to hand out, so a value that cannot be
isolated is not stored -- the same rule as an identity-coupled result or a call
that mutated its arguments. Such a value is unpicklable too, so no disk tier
could hold it either; only in-process repeat hits are given up.
"""

from __future__ import annotations

import threading
import warnings

import pytest

from cash import Cash
from cash.backends import InMemoryBackend
from cash.exceptions import CashCacheStoreFailedWarning


@pytest.fixture
def cash(tmp_path):
    return Cash(cache_dir=str(tmp_path / "c"), register_magic=False)


def test_a_result_holding_a_lock_is_recomputed(cash):
    runs = []

    @cash.cache
    def context(n):
        runs.append(n)
        return {"rows": [1, 2, 3], "lock": threading.Lock()}

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        first = context(1)
        first["rows"].append(99)
        second = context(1)

    assert second["rows"] == [1, 2, 3], "the caller's mutation reached the next call"
    assert first is not second
    assert len(runs) == 2
    assert any(issubclass(w.category, CashCacheStoreFailedWarning) for w in seen), [str(w.message) for w in seen]


def test_the_reason_is_available_afterwards(cash):
    @cash.cache
    def context(n):
        return {"lock": threading.Lock()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        context(1)
    codes = [w.get("code") for w in (context.cache_info().get("warnings") or [])]
    assert "STORE-FAILED" in codes, codes


def test_an_ordinary_result_still_caches(cash):
    runs = []

    @cash.cache
    def rows(n):
        runs.append(n)
        return {"rows": [1, 2, 3]}

    first = rows(1)
    first["rows"].append(99)
    assert rows(1)["rows"] == [1, 2, 3]
    assert len(runs) == 1


def test_the_memory_backend_refuses_a_value_it_must_copy(tmp_path):
    """The refusal is scoped to what the decorator stores, where the value IS
    what the next call hands back. A notebook statement's payload is the
    variables a cell left behind, and one unisolatable variable among them (an
    open handle in scope) must not stop the statement being cached.

    The flag is ``copy_required``, not ``decorator_entry``. The two used to be
    one, which made ``frozen=True`` -- excluded from the copy promise on
    purpose -- look undecorated to the persistence gates as well, and it
    quietly stopped reaching disk at all.
    """
    backend = InMemoryBackend()
    from cash.exceptions import CacheBackendError

    with pytest.raises(CacheBackendError):
        backend.set("k", {"lock": threading.Lock()}, {"copy_required": True})
    assert backend.get("k") == (None, None)

    backend.set("stmt", {"lock": threading.Lock()}, {})  # a notebook payload
    assert backend.get("stmt")[1] is not None

    # Decorated, but frozen: the caller has promised not to modify the result,
    # so handing the same object back is exactly what was asked for.
    backend.set("frozen", {"lock": threading.Lock()}, {"decorator_entry": True})
    assert backend.get("frozen")[1] is not None


def test_a_frozen_result_may_be_one_cash_cannot_copy(cash):
    """`frozen=True` is the caller promising the result is not modified.

    That promise is what lets the RAM tier hand the same object back instead
    of copying it, so a result holding a lock still caches -- where the same
    function without `frozen=` is recomputed every call
    (`test_a_result_holding_a_lock_is_recomputed`).

    The DISK tier still cannot hold it -- a lock does not pickle -- so
    STORE-FAILED is expected here. The claim is about the in-process hit.
    """
    runs = []

    @cash.cache(frozen=True)
    def context(n):
        runs.append(n)
        return {"rows": [1, 2, 3], "lock": threading.Lock()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = context(1)
        second = context(1)

    assert len(runs) == 1, "a frozen result was refused for being uncopyable"
    assert first is second, "frozen promises the same object back, not a copy"
