"""@cash.cache on functions that return iterators/generators.

The decorator materializes the iterator into a list, caches the list,
and returns a CachedIterator wrapper that preserves the iterator
protocol. Each call yields a fresh, independent iterator over the
cached values.
"""
from __future__ import annotations

import asyncio
import types
import warnings

import pytest

from cash import Cash, CashCacheIneffectiveWarning


def test_generator_function_caches(tmp_path):
    """A generator-returning function is materialized, cached, and
    subsequent calls return an iterator over the same values without
    recomputing."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache
    def gen(stop):
        n["calls"] += 1
        for i in range(stop):
            yield i * 10

    r1 = list(gen(5))
    assert r1 == [0, 10, 20, 30, 40]
    assert n["calls"] == 1

    r2 = list(gen(5))  # hit
    assert r2 == [0, 10, 20, 30, 40]
    assert n["calls"] == 1, f"expected cache hit, got calls={n['calls']}"


def test_generator_returns_independent_iterators(tmp_path):
    """Two calls produce independent iterators over the same values
    (consuming one does not affect the other)."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def gen():
        yield 1
        yield 2
        yield 3

    g1 = gen()
    g2 = gen()
    assert next(g1) == 1
    assert next(g2) == 1
    assert next(g1) == 2
    assert next(g2) == 2
    assert next(g1) == 3
    with pytest.raises(StopIteration):
        next(g1)
    # g2 still has one element left
    assert next(g2) == 3


def test_map_filter_results_cached(tmp_path):
    """map() and filter() results are one-shot iterators; they should
    also be detected and materialized."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache
    def double_evens(seq):
        n["calls"] += 1
        return map(lambda x: x * 2, filter(lambda x: x % 2 == 0, seq))

    assert list(double_evens([1, 2, 3, 4, 5])) == [4, 8]
    assert list(double_evens([1, 2, 3, 4, 5])) == [4, 8]
    assert n["calls"] == 1


def test_returned_value_satisfies_iterator_protocol(tmp_path):
    """The returned wrapper must satisfy iter(x) is x — the iterator
    protocol's reflexivity test."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def gen():
        yield from range(3)

    result = gen()
    # iter(x) is x — what the user expects from a generator/iterator.
    assert iter(result) is result, (
        f"cached iterator must satisfy iter(x) is x; got type {type(result).__name__}"
    )


def test_send_raises_attribute_error(tmp_path):
    """The wrapper does not support generator.send() — raises AttributeError
    with a message pointing at caching as the cause."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def gen():
        yield 1
        yield 2

    result = gen()
    with pytest.raises(AttributeError, match="send"):
        result.send(None)


def test_throw_raises_attribute_error(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def gen():
        yield 1

    result = gen()
    with pytest.raises(AttributeError, match="throw"):
        result.throw(ValueError())


def test_close_stops_iteration(tmp_path):
    """close() on the wrapper stops further iteration (subsequent next
    raises StopIteration)."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def gen():
        yield 1
        yield 2
        yield 3

    result = gen()
    assert next(result) == 1
    result.close()
    with pytest.raises(StopIteration):
        next(result)


def test_collections_pass_through_unchanged(tmp_path):
    """list, dict, set, tuple, str — iterable but NOT one-shot — must
    be cached as-is, not wrapped."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f_list():
        return [1, 2, 3]

    @c.cache
    def f_dict():
        return {"a": 1, "b": 2}

    @c.cache
    def f_set():
        return {1, 2, 3}

    @c.cache
    def f_tuple():
        return (1, 2, 3)

    @c.cache
    def f_str():
        return "hello"

    # All should hit on the second call and return the same type as
    # they did on the first call.
    for fn, expected_type in (
        (f_list, list), (f_dict, dict), (f_set, set),
        (f_tuple, tuple), (f_str, str),
    ):
        r1 = fn()
        r2 = fn()
        assert type(r1) is expected_type
        assert type(r2) is expected_type
        assert r1 == r2


def test_empty_generator_caches(tmp_path):
    """An empty generator (yields nothing) still goes through the
    materialize-and-cache path and returns an empty iterator on hit."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache
    def gen():
        n["calls"] += 1
        return
        yield  # makes it a generator function syntactically

    assert list(gen()) == []
    assert list(gen()) == []
    assert n["calls"] == 1


async def test_async_function_returning_iterator(tmp_path):
    """An async def that RETURNS a sync iterator (not yields — that's an
    async generator, handled separately at decoration time) should also
    have its iterator materialized and cached."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache
    async def make_iter(stop):
        n["calls"] += 1
        await asyncio.sleep(0)
        return (i for i in range(stop))

    r1 = list(await make_iter(4))
    assert r1 == [0, 1, 2, 3]
    r2 = list(await make_iter(4))
    assert r2 == [0, 1, 2, 3]
    assert n["calls"] == 1


def test_cache_persists_across_instances(tmp_path):
    """The cached list (not the wrapper) is what's persisted — so a
    fresh Cash instance pointed at the same cache_dir should still
    return the values via a fresh CachedIterator wrapper.

    Uses an explicit FileBackend so we don't depend on TieredBackend's
    smart-persistence floor (which gates short calls from reaching disk).
    """
    from cash.backends.file_backend import FileBackend
    store = str(tmp_path / "store")
    c1 = Cash(backend=FileBackend(store, flush_interval=0), register_magic=False)
    n = {"calls": 0}

    # Define the function inside a helper so __qualname__ is stable
    # across the two Cash instances.
    def _make(c, n):
        @c.cache
        def gen():
            n["calls"] += 1
            yield from [10, 20, 30]
        return gen

    g1 = _make(c1, n)
    assert list(g1()) == [10, 20, 30]
    c1.shutdown()

    # New instance, same backend store
    c2 = Cash(backend=FileBackend(store, flush_interval=0), register_magic=False)
    g2 = _make(c2, n)
    r = list(g2())
    assert r == [10, 20, 30]
    # The first instance computed once; the second should hit the
    # persisted list and not recompute.
    assert n["calls"] == 1, (
        f"second instance should have hit the persisted cache (calls={n['calls']})"
    )


def test_cache_if_predicate_sees_materialized_list_for_iterator_function(tmp_path):
    """When a function returns an iterator, the cache_if predicate
    receives the MATERIALIZED LIST, not the raw generator. This is
    the natural composition — predicates can use len(), bool(), etc."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    # Cache only when the materialized result is non-empty.
    @c.cache(cache_if=lambda result: len(result) > 0)
    def gen(stop):
        n["calls"] += 1
        for i in range(stop):
            yield i

    # stop=0 → empty list → predicate False → no cache
    list(gen(0))
    list(gen(0))
    assert n["calls"] == 2, "empty-result calls should not be cached"

    # stop=3 → non-empty list → predicate True → cached
    list(gen(3))
    list(gen(3))
    assert n["calls"] == 3, f"non-empty result should be cached (calls={n['calls']})"


class _FakeBackend:
    """Minimal in-memory backend for unit-testing _ChunkedCachedIterator
    in isolation. Tracks .get() calls so we can verify lazy chunk reads.
    """

    def __init__(self, store):
        # store: dict[str, list[Any]] — pre-arranged chunks keyed by full chunk_key.
        self._store = dict(store)
        self.get_calls = []

    def get(self, key):
        self.get_calls.append(key)
        if key in self._store:
            return ({}, self._store[key])
        return (None, None)


def test_chunked_iterator_lazy_chunk_reads():
    """The iterator must only fetch a chunk when iteration enters it."""
    from cash.core import _ChunkedCachedIterator

    backend = _FakeBackend({
        "K:chunk_0": [1, 2, 3],
        "K:chunk_1": [4, 5, 6],
        "K:chunk_2": [7, 8, 9],
    })

    class _FakeCash:
        def __init__(self, backend):
            self.backend = backend

    cash = _FakeCash(backend)
    it = _ChunkedCachedIterator(cash, "K", n_chunks=3)

    # No reads until iteration starts.
    assert backend.get_calls == []

    # First next() pulls chunk_0.
    assert next(it) == 1
    assert backend.get_calls == ["K:chunk_0"]

    # Remaining items in chunk_0: no new backend reads.
    assert next(it) == 2
    assert next(it) == 3
    assert backend.get_calls == ["K:chunk_0"]

    # Crossing into chunk_1 triggers exactly one new read.
    assert next(it) == 4
    assert backend.get_calls == ["K:chunk_0", "K:chunk_1"]


def test_chunked_iterator_iter_is_self():
    """_ChunkedCachedIterator must satisfy iter(x) is x (iterator protocol)."""
    from cash.core import _ChunkedCachedIterator

    class _EmptyCash:
        class backend:
            @staticmethod
            def get(key):
                return (None, None)

    it = _ChunkedCachedIterator(_EmptyCash(), "K", n_chunks=0)
    assert iter(it) is it


def test_chunked_iterator_close_stops_iteration():
    """After close(), next() raises StopIteration."""
    from cash.core import _ChunkedCachedIterator

    backend = _FakeBackend({"K:chunk_0": [1, 2, 3]})

    class _FakeCash:
        def __init__(self, backend):
            self.backend = backend

    it = _ChunkedCachedIterator(_FakeCash(backend), "K", n_chunks=1)
    assert next(it) == 1
    it.close()
    with pytest.raises(StopIteration):
        next(it)


def test_chunked_iterator_send_throw_raise():
    """send and throw raise AttributeError with a clear message."""
    from cash.core import _ChunkedCachedIterator

    class _EmptyCash:
        class backend:
            @staticmethod
            def get(key):
                return (None, None)

    it = _ChunkedCachedIterator(_EmptyCash(), "K", n_chunks=0)
    with pytest.raises(AttributeError, match="send"):
        it.send(None)
    with pytest.raises(AttributeError, match="throw"):
        it.throw(ValueError())


def test_chunked_iterator_missing_chunk_terminates_safely():
    """If a chunk read returns (None, None) — e.g. chunk was evicted from
    the backend — the iterator must terminate via StopIteration, not raise."""
    from cash.core import _ChunkedCachedIterator

    backend = _FakeBackend({
        "K:chunk_0": [1, 2],
        # chunk_1 is missing on purpose
        "K:chunk_2": [5, 6],
    })

    class _FakeCash:
        def __init__(self, backend):
            self.backend = backend

    it = _ChunkedCachedIterator(_FakeCash(backend), "K", n_chunks=3)
    values = list(it)
    # Chunk_0 yields [1, 2]; chunk_1 missing → iteration stops there.
    assert values == [1, 2]


def test_list_cached_iterator_alias_still_imports():
    """CachedIterator was renamed to _ListCachedIterator. The original name
    is kept as a deprecation-friendly alias for one release."""
    from cash.core import _ListCachedIterator, CachedIterator
    # The alias must point at the new internal class.
    assert CachedIterator is _ListCachedIterator


def test_chunk_max_items_kwarg_accepted(tmp_path):
    """Cash.cache(chunk_max_items=N) is a valid signature and does not
    break the existing iterator-caching behavior."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(chunk_max_items=100)
    def gen():
        n["calls"] += 1
        yield from range(5)

    assert list(gen()) == [0, 1, 2, 3, 4]
    assert list(gen()) == [0, 1, 2, 3, 4]
    assert n["calls"] == 1


def test_chunk_max_bytes_kwarg_accepted(tmp_path):
    """Cash.cache(chunk_max_bytes=N) is a valid signature."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(chunk_max_bytes=1024)
    def gen():
        n["calls"] += 1
        yield from range(5)

    assert list(gen()) == [0, 1, 2, 3, 4]
    assert list(gen()) == [0, 1, 2, 3, 4]
    assert n["calls"] == 1


def test_chunked_storage_multi_chunk_round_trip(tmp_path):
    """A generator yielding more items than chunk_max_items must produce
    multiple chunks. The second call hits the cache and yields the same
    values without recomputing."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(chunk_max_items=10)  # force multiple chunks for small generator
    def gen():
        n["calls"] += 1
        yield from range(25)

    r1 = list(gen())
    assert r1 == list(range(25))
    assert n["calls"] == 1

    r2 = list(gen())  # hit
    assert r2 == list(range(25))
    assert n["calls"] == 1, f"expected cache hit, got calls={n['calls']}"


def test_chunked_storage_persists_manifest_metadata(tmp_path):
    """After a chunked write, the canonical key's metadata identifies
    the storage type as 'chunked' and records n_chunks."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache(chunk_max_items=10)
    def gen():
        yield from range(25)

    list(gen())  # populate

    # Find the canonical cache key for this call. We don't introspect
    # private internals — instead, iterate backend entries and find the
    # manifest (the one without a :chunk_N suffix).
    all_keys = [e['key'] for e in c.backend.list_entries() if e.get('key')]
    manifest_keys = [k for k in all_keys
                     if k.startswith("tests.test_core.test_iterator_caching")
                     and not k.rsplit(":", 1)[-1].startswith("chunk_")]
    assert len(manifest_keys) == 1, f"expected exactly one manifest, got {manifest_keys}"
    canonical_key = manifest_keys[0]

    metadata, _ = c.backend.get(canonical_key)
    assert metadata is not None
    assert metadata.get("iterator_storage") == "chunked"
    assert metadata.get("n_chunks") == 3  # 25 items / 10 per chunk = 3 chunks


def test_chunked_storage_single_chunk_below_thresholds(tmp_path):
    """If the iterator fits in one chunk, storage produces exactly
    one chunk entry and a manifest. Behavior matches v1 from the
    user's perspective."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    @c.cache(chunk_max_items=1000)  # well above the 5-item generator
    def gen():
        n["calls"] += 1
        yield from range(5)

    assert list(gen()) == [0, 1, 2, 3, 4]
    assert list(gen()) == [0, 1, 2, 3, 4]
    assert n["calls"] == 1


def test_chunked_storage_byte_threshold_closes_chunk_early(tmp_path):
    """When chunk_max_bytes is reached before chunk_max_items, the
    chunk closes at the byte boundary."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    n = {"calls": 0}

    # Each item is a large string; force the byte threshold before
    # the item count threshold.
    @c.cache(chunk_max_items=1000, chunk_max_bytes=4096)
    def gen():
        n["calls"] += 1
        for _ in range(20):
            yield "x" * 1024  # ~1 KiB per item

    r1 = list(gen())
    assert len(r1) == 20
    assert all(s == "x" * 1024 for s in r1)
    assert n["calls"] == 1

    # Verify multiple chunks were written.
    all_keys = [e['key'] for e in c.backend.list_entries() if e.get('key')]
    chunk_keys = [k for k in all_keys if ":chunk_" in k]
    assert len(chunk_keys) >= 2, (
        f"expected the 4 KiB byte threshold to produce multiple chunks "
        f"from 20 × 1 KiB items; got {chunk_keys}"
    )

    r2 = list(gen())
    assert r2 == r1
    assert n["calls"] == 1


def test_cache_if_bypass_warning_fires_on_partial_last_chunk(tmp_path):
    """Bug regression: when chunk_1 is created via the tail-flush path
    (chunk_0 filled at the threshold, chunk_1 has a partial buffer),
    the cache_if-bypass warning must still fire — we ARE committing to
    multi-chunk caching with the predicate bypassed.

    The original code only fired the warning when chunk_1's threshold
    was hit; the partial-chunk_1 case slipped through silently.
    """
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    predicate_calls = {"n": 0}

    def predicate(result):
        predicate_calls["n"] += 1
        return False

    @c.cache(chunk_max_items=10, cache_if=predicate)
    def gen():
        # 11 items → chunk_0 fills (10 items), chunk_1 gets 1 item via tail flush.
        yield from range(11)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        r = list(gen())

    assert r == list(range(11))
    assert predicate_calls["n"] == 0  # bypassed

    ineffective = [
        w for w in captured
        if issubclass(w.category, CashCacheIneffectiveWarning)
        and "cache_if was bypassed" in str(w.message)
    ]
    assert len(ineffective) == 1, (
        f"expected one cache_if-bypassed warning for partial chunk_1, got "
        f"{[str(w.message) for w in captured]}"
    )
