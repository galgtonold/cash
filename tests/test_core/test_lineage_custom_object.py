"""The lineage short-circuit should apply to custom objects, not just DataFrames.

A cached function returning a custom class instance used to get no
``_cash_lineage_hash``, so a downstream cached function content-hashed
(pickled) it on every call instead of keying on the cheap lineage hash. Any
object that accepts attribute assignment now carries the lineage hash.

Plain ``list``/``dict``/``tuple``/scalar returns still cannot - builtins reject
attribute assignment - so those keep content-hashing (a hard Python limit, not
a regression).
"""
from __future__ import annotations

import tempfile

from cash import Cash, FileBackend, InMemoryBackend


class Result:
    """A custom, expensive-to-hash result object."""

    def __init__(self, payload):
        self.payload = payload


def _cash_mem() -> Cash:
    return Cash(backend=InMemoryBackend())


def test_custom_object_carries_lineage_hash():
    c = _cash_mem()

    @c.cache
    def produce(n):
        return Result(list(range(n)))

    r = produce(5)
    assert hasattr(r, "_cash_lineage_hash")


def test_custom_object_short_circuits_downstream():
    c = _cash_mem()
    calls = {"n": 0}

    @c.cache
    def produce(n):
        return Result(list(range(n)))

    @c.cache
    def consume(r):
        calls["n"] += 1
        return sum(r.payload)

    r = produce(5)
    assert consume(r) == 10
    assert consume(r) == 10            # same object -> lineage hit
    assert calls["n"] == 1


def test_custom_object_lineage_distinguishes_producers():
    c = _cash_mem()
    calls = {"n": 0}

    @c.cache
    def produce(n):
        return Result(list(range(n)))

    @c.cache
    def consume(r):
        calls["n"] += 1
        return sum(r.payload)

    consume(produce(3))               # 0+1+2 = 3
    consume(produce(4))               # 0+1+2+3 = 6 -> different lineage, recompute
    assert calls["n"] == 2


def test_custom_object_lineage_survives_disk_restore():
    c = Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))

    @c.cache
    def produce(n):
        return Result(list(range(n)))

    r1 = produce(5)
    assert hasattr(r1, "_cash_lineage_hash")
    r2 = produce(5)                   # restored from disk
    assert hasattr(r2, "_cash_lineage_hash"), "lineage lost on disk restore"
    assert r1._cash_lineage_hash == r2._cash_lineage_hash


def test_builtin_containers_do_not_crash():
    """list/dict/tuple/scalar can't take attributes; attachment is skipped
    silently and the call still works (downstream content-hashes them)."""
    c = _cash_mem()

    @c.cache
    def make_list(n):
        return list(range(n))

    @c.cache
    def make_dict(n):
        return {"k": n}

    @c.cache
    def total(xs):
        return sum(xs) if isinstance(xs, list) else xs["k"]

    assert total(make_list(4)) == 6
    assert total(make_list(4)) == 6
    assert total(make_dict(7)) == 7
    # No _cash_lineage_hash on builtins (can't be set), but everything works.
    assert not hasattr(make_list(4), "_cash_lineage_hash")
