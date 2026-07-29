"""A call unit's key is order-independent — that IS the CAS-242 fix."""
from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_interception import CallSite
from cash.notebook.call_unit import call_cache_key


def _site(source="compute(x)", names=("compute", "x"), occ=0, computed_arg_positions=()):
    return CallSite(
        source=source,
        free_names=frozenset(names),
        occurrence_index=occ,
        computed_arg_positions=computed_arg_positions,
    )


def _ctx(lineage, ns):
    return CacheKeyContext(variable_lineage=lineage, user_ns=ns)


def test_key_is_namespaced_to_call():
    key = call_cache_key(_site(), ctx=_ctx({"x": "aaa"}, {"x": 1, "compute": len}), arg_digests=[], loop_vars={})
    assert key.startswith("call:")


def test_different_argument_lineage_gives_different_key():
    a = call_cache_key(_site(), ctx=_ctx({"x": "hash-of-5"}, {"x": 5, "compute": len}), arg_digests=[], loop_vars={})
    b = call_cache_key(_site(), ctx=_ctx({"x": "hash-of-9"}, {"x": 9, "compute": len}), arg_digests=[], loop_vars={})
    assert a != b


def test_occurrence_index_separates_identical_sites():
    ctx = _ctx({"x": "aaa"}, {"x": 1, "compute": len})
    first = call_cache_key(_site(occ=0), ctx=ctx, arg_digests=[], loop_vars={})
    second = call_cache_key(_site(occ=1), ctx=ctx, arg_digests=[], loop_vars={})
    assert first != second


def test_a_computed_argument_discriminates_by_VALUE_not_lineage():
    """The iterator-collapse guard. `compute(next(it))` must not collapse.

    `it`'s lineage is an id-hash and never moves, so if the key ignored the
    evaluated argument every iteration would share one key and iterations 2..N
    would be served iteration 1's value -- wrong on the FIRST run.
    """
    ctx = _ctx({"it": "id-hash-stable"}, {"it": iter(range(3)), "compute": len})
    site = _site(
        source="compute(next(it))",
        names=("compute", "next", "it"),
        computed_arg_positions=(0,),
    )

    first = call_cache_key(site, ctx=ctx, arg_digests=["digest-of-0"], loop_vars={})
    second = call_cache_key(site, ctx=ctx, arg_digests=["digest-of-1"], loop_vars={})
    assert first != second
    assert first is not None
    assert second is not None


def test_loop_vars_discriminate_iterations_with_no_varying_argument():
    """`for _ in range(3): out.append(fetch_next(conn))`.

    `conn` is a bare Name whose lineage never moves, and there is no computed
    argument to hash -- so without the loop variable every iteration would
    build one key and iterations 2..N would be served iteration 1's value.
    """
    ctx = _ctx({"conn": "aaa"}, {"conn": object(), "fetch_next": len})
    site = _site(source="fetch_next(conn)", names=("fetch_next", "conn"))

    keys = {
        call_cache_key(site, ctx=ctx, arg_digests=[], loop_vars={"_": i})
        for i in range(3)
    }
    assert len(keys) == 3


def test_reordering_the_iterable_preserves_every_items_key():
    """Simulate one loop in two orders; every item must keep its key.

    NOT `f(same args) == f(same args)` -- that is a tautology for any
    deterministic function and cannot fail. `call_cache_key` has no positional
    input to vary, so the only way to test order-independence honestly is to
    build the keys a whole loop would produce in two different orders and
    compare the SETS. This fails the moment anything positional enters the key.
    """
    def keys_for(order):
        return {
            call_cache_key(
                _site(),
                ctx=_ctx({"x": f"hash-of-{v}"}, {"x": v, "compute": len}),
                arg_digests=[],
                loop_vars={"x": v},
            )
            for v in order
        }

    assert keys_for([1, 10, 5]) == keys_for([5, 10, 1])


def test_duplicate_loop_items_collapse_to_one_key():
    """`for x in [5, 5, 5]` -- collapsing here is CORRECT, not a hazard.

    Same callee, same inputs, no dependency cash cannot see: serving the cache
    is the right answer and a speedup. The statement path already does exactly
    this -- `compute_context_hash` yields one hash for three identical
    iteration contexts -- so this is shipped behaviour being matched.
    """
    ctx = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    first = call_cache_key(_site(), ctx=ctx, arg_digests=[], loop_vars={"x": 5})
    second = call_cache_key(_site(), ctx=ctx, arg_digests=[], loop_vars={"x": 5})
    assert first == second


def test_outside_a_loop_empty_loop_vars_is_accepted():
    """Not every call site is in a loop; the parameter is still required."""
    ctx = _ctx({"x": "aaa"}, {"x": 1, "compute": len})
    assert call_cache_key(_site(), ctx=ctx, arg_digests=[], loop_vars={}).startswith("call:")


def test_mismatched_arg_digest_count_refuses_to_cache():
    """F1: a caller that loses the only discriminator must not get a key.

    `computed_arg_positions=(0,)` means the call has one non-Name argument
    whose evaluated value must be hashed into the key. Supplying zero digests
    means that discriminator is missing -- minting a key anyway risks a
    collapsed, wrong one (the exact failure
    ``test_a_computed_argument_discriminates_by_VALUE_not_lineage`` guards
    against), so this refuses (returns ``None``) rather than guess. A caching
    optimisation must never be why user code fails: an uncached call is
    merely slow, a wrong cached value is silently incorrect.
    """
    site = _site(
        source="compute(next(it))",
        names=("compute", "next", "it"),
        computed_arg_positions=(0,),
    )
    ctx = _ctx({"it": "id-hash-stable"}, {"it": iter(range(3)), "compute": len})
    assert call_cache_key(site, ctx=ctx, arg_digests=[], loop_vars={}) is None


def test_a_dunder_entry_in_loop_vars_cannot_reach_the_key():
    """The CAS-242 channel this feature actually creates.

    `loop_vars` is documented as "the non-dunder entries", but documenting a
    contract is not enforcing it: if `__iterable_lineage__` ever reaches this
    function it would be hashed in like any other entry, and reordering would
    re-run the tail again -- the exact bug the omission exists to fix.

    Filter dunder keys INSIDE `call_cache_key` rather than trusting the
    caller, and pin it here. Asserting that `__iterable_lineage__` is absent
    from `ctx.variable_lineage` does NOT test this -- `compute_cache_key` only
    reads lineage for names in `site.free_names`, which can never contain a
    dunder, so such a test passes for a reason that predates this feature.
    """
    ctx = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    clean = call_cache_key(_site(), ctx=ctx, arg_digests=[], loop_vars={"x": 5})
    polluted = call_cache_key(
        _site(), ctx=ctx, arg_digests=[],
        loop_vars={"x": 5, "__iterable_lineage__": "whole-iterable-hash"},
    )
    assert clean == polluted
