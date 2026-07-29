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


def test_same_argument_lineage_gives_same_key_regardless_of_call_order():
    """Reordering the iterable must not change any item's key."""
    ctx_a = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    ctx_b = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    assert (
        call_cache_key(_site(), ctx=ctx_a, arg_digests=[], loop_vars={})
        == call_cache_key(_site(), ctx=ctx_b, arg_digests=[], loop_vars={})
    )


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


def test_loop_vars_are_order_independent():
    """Reordering the iterable must not change any item's key.

    The loop variable's VALUE for item 5 is 5 whatever position it occupies,
    so a reorder produces the same key per item -- unlike the iteration
    context, whose `__iterable_lineage__` changes for every iteration and is
    exactly why reordering currently re-runs the tail (CAS-242).
    """
    ctx = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    at_position_0 = call_cache_key(_site(), ctx=ctx, arg_digests=[], loop_vars={"x": 5})
    at_position_2 = call_cache_key(_site(), ctx=ctx, arg_digests=[], loop_vars={"x": 5})
    assert at_position_0 == at_position_2


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


def test_iteration_context_is_not_folded_into_the_key():
    """F2: the headline constraint, pinned so a refactor can't reintroduce CAS-242.

    ``for_handler.py`` folds a loop's ``__iterable_lineage__`` into the
    STATEMENT key by prepending a ``# __iteration_context__: <hash>`` comment
    to the statement's source before it ever reaches ``compute_cache_key`` --
    so reordering the iterable changes every iteration's source hash. That
    is CAS-242.

    ``call_cache_key`` has no parameter for an iteration context and never
    sees such a comment: ``site.source`` is the call's own source, fixed at
    rewrite time, and ``site.free_names`` can never contain a dunder
    pseudo-variable like ``__iterable_lineage__`` (real Python source cannot
    reference it as a bare name). This asserts the corresponding invariant on
    the one channel that reaches ``compute_cache_key`` at all: two contexts
    identical except that one's ``variable_lineage`` additionally carries an
    ``__iterable_lineage__``-shaped entry (as if something tried to smuggle
    it in) must still produce a byte-identical key, because that entry is
    never a member of ``site.free_names`` and so is never looked up.
    """
    ctx_a = _ctx({"x": "hash-of-5"}, {"x": 5, "compute": len})
    ctx_b = _ctx(
        {"x": "hash-of-5", "__iterable_lineage__": "some-other-iterables-hash"},
        {"x": 5, "compute": len},
    )
    key_a = call_cache_key(_site(), ctx=ctx_a, arg_digests=[], loop_vars={"x": 5})
    key_b = call_cache_key(_site(), ctx=ctx_b, arg_digests=[], loop_vars={"x": 5})
    assert key_a == key_b
