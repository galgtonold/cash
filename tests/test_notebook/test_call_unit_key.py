"""A call unit's key is order-independent — that IS the CAS-242 fix."""
from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_interception import CallSite
from cash.notebook.call_unit import call_cache_key


def _site(source="compute(x)", names=("compute", "x"), occ=0):
    return CallSite(source=source, free_names=frozenset(names), occurrence_index=occ)


def _ctx(lineage, ns):
    return CacheKeyContext(variable_lineage=lineage, user_ns=ns)


def test_key_is_namespaced_to_call():
    key = call_cache_key(_site(), ctx=_ctx({"x": "aaa"}, {"x": 1, "compute": len}))
    assert key.startswith("call:")


def test_same_argument_lineage_gives_same_key_regardless_of_call_order():
    """Reordering the iterable must not change any item's key."""
    ctx_a = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    ctx_b = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    assert call_cache_key(_site(), ctx=ctx_a) == call_cache_key(_site(), ctx=ctx_b)


def test_different_argument_lineage_gives_different_key():
    a = call_cache_key(_site(), ctx=_ctx({"x": "hash-of-5"}, {"x": 5, "compute": len}))
    b = call_cache_key(_site(), ctx=_ctx({"x": "hash-of-9"}, {"x": 9, "compute": len}))
    assert a != b


def test_occurrence_index_separates_identical_sites():
    ctx = _ctx({"x": "aaa"}, {"x": 1, "compute": len})
    first = call_cache_key(_site(occ=0), ctx=ctx)
    second = call_cache_key(_site(occ=1), ctx=ctx)
    assert first != second


def test_a_computed_argument_discriminates_by_VALUE_not_lineage():
    """The iterator-collapse guard. `compute(next(it))` must not collapse.

    `it`'s lineage is an id-hash and never moves, so if the key ignored the
    evaluated argument every iteration would share one key and iterations 2..N
    would be served iteration 1's value -- wrong on the FIRST run.
    """
    ctx = _ctx({"it": "id-hash-stable"}, {"it": iter(range(3)), "compute": len})
    site = _site(source="compute(next(it))", names=("compute", "next", "it"))

    first = call_cache_key(site, ctx=ctx, arg_digests=["digest-of-0"])
    second = call_cache_key(site, ctx=ctx, arg_digests=["digest-of-1"])
    assert first != second


def test_repeat_index_separates_identical_keys_within_one_run():
    """Closes hidden state behind a BARE NAME (`fetch_next(conn)`)."""
    ctx = _ctx({"conn": "aaa"}, {"conn": object(), "compute": len})
    site = _site(source="fetch_next(conn)", names=("fetch_next", "conn"))

    first = call_cache_key(site, ctx=ctx, arg_digests=[], repeat_index=0)
    second = call_cache_key(site, ctx=ctx, arg_digests=[], repeat_index=1)
    assert first != second


def test_repeat_index_does_not_break_reorder_reuse():
    """Distinct argument lineages never collide, so each still counts 0.

    This is why the counter is per-(site, KEY) and not per-site: reordering a
    loop's items gives each item the same key it had before, each seen once.
    """
    ctx = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    assert (
        call_cache_key(_site(), ctx=ctx, arg_digests=["d5"], repeat_index=0)
        == call_cache_key(_site(), ctx=ctx, arg_digests=["d5"], repeat_index=0)
    )
