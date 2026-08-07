"""A callee whose effects cannot be replayed must not be cached at all.

Task 6 refused a call by inspecting its *returned value* (result-is-an-arg,
identity-coupled library objects). This module refuses by what the call
*did*: mutated an argument in place, or consumed RNG. Both are things the
statement's ambient capture would have recorded for free on a genuine miss,
but silently loses on a served hit -- so the refusal is permanent per site.

Every test states the ONE-LINE change that would break it, in a comment
right above the assertion it protects.
"""
import time

from cash.notebook.call_interception import CallCache, CallSite
from cash.notebook.call_unit import CallUnit


def _site(source="f(d)", names=("f", "d"), computed_arg_positions=()):
    """Bare-Name argument by default (`computed_arg_positions=()`, `CallSite`'s
    own default): the key is then driven purely by `site` + variable lineage,
    NOT by the live argument's content hash. That matters for
    `test_mutation_refusal_is_permanent_across_wrap_calls` below, which
    deliberately leaves `d`'s content different across calls -- if the key
    depended on that content it would change key on its own, which would
    mask a broken (non-permanent) refusal rather than expose it.
    """
    return CallSite(
        source=source, free_names=frozenset(names), occurrence_index=0,
        computed_arg_positions=computed_arg_positions,
    )


def _computed_site(source="f(d)", names=("f", "d")):
    """Matches the convention `test_call_interception_respects_refusals.py`
    uses for its production-dispatch `_site` helper (`computed_arg_positions
    =(0,)`): needed there because `CallCache`'s default context has no live
    variable lineage to resolve a bare Name against.
    """
    return CallSite(
        source=source, free_names=frozenset(names), occurrence_index=0,
        computed_arg_positions=(0,),
    )


def test_a_callee_that_mutates_its_argument_is_never_cached(call_unit_harness):
    """`log.append(clean(df))` where clean does df.dropna(inplace=True).

    A warm hit would skip the cleaning and leave `df` dirty, while the cold
    run cleans it -- the two runs disagree about `df`, not just about the
    return value. The arg-IDENTITY check (Task 6) only catches `return arg`.

    Mutation-that-breaks-this-test: dropping the `_hash_args(...) !=
    arg_hashes_before` comparison in `CallUnit.wrap` (i.e. never refusing on
    argument mutation) makes the second call a hit, so `calls == [1]` and
    `payload["dirty"] is True`.
    """
    calls = []

    def clean(d):
        calls.append(1)
        time.sleep(0.05)
        d["dirty"] = False
        return len(d)

    payload = {"dirty": True}
    unit = call_unit_harness(lineage={"d": "hash-d"}, user_ns={"d": payload, "clean": clean})

    unit.wrap(clean, _site())(payload)
    payload["dirty"] = True          # reset, as a fresh cold run would find it
    unit.wrap(clean, _site())(payload)

    # Asserting only on the return value (`len(d)`) would pass even if the
    # second call were served from cache without re-running `clean` --
    # `len(d)` doesn't change whether `dirty` is flipped or not. The
    # mutation flag is the only observable that distinguishes "recomputed"
    # from "hit, mutation skipped", which is the entire bug this guards.
    assert calls == [1, 1], "must recompute -- the mutation has to re-happen"
    assert payload["dirty"] is False


def test_a_callee_that_draws_is_never_cached(call_unit_harness):
    """RNG is a consumed linear resource; v1 refuses rather than replays.

    See CAS-254. A hit would leave the global stream where it was, so every
    downstream draw diverges from the uncached oracle.

    Mutation-that-breaks-this-test: dropping the
    `rng_modules_changed(rng_before, capture_rng_state())` check in
    `CallUnit.wrap` makes the second call a hit, so `calls == [1]` and
    `first == second` (the cached value is replayed verbatim).
    """
    import random

    calls = []

    def sample():
        calls.append(1)
        time.sleep(0.05)
        return random.random()

    unit = call_unit_harness(lineage={}, user_ns={"sample": sample})
    site = _site(source="sample()", names=("sample",), computed_arg_positions=())

    first = unit.wrap(sample, site)()
    second = unit.wrap(sample, site)()

    assert calls == [1, 1], "an RNG-consuming callee must never be served from cache"
    assert first != second


def test_mutation_refusal_is_permanent_across_wrap_calls(call_unit_harness):
    """The refusal is keyed by cache key and must survive re-wrapping the
    same site -- `wrap` is called fresh per statement execution in
    production (`CallCache.resolve` builds a new wrapper on every cell run),
    so the refusal set has to live on `CallUnit`, not on the closure `wrap`
    returns.

    The second call is deliberately left in the ALREADY-CLEAN state (`payload`
    is not reset to `dirty=True` beforehand): `clean` is then a no-op on that
    call, so the mutation is invisible to a fresh, one-call-only hash
    comparison. This is what actually distinguishes "refusal recorded on
    `self`" from "refusal recorded on a closure that gets thrown away":

    - correct (permanent) impl: call 2 short-circuits on `key in
      self._refused` (learned from call 1) BEFORE ever comparing hashes, so
      it never gets a chance to be fooled by looking clean, and never stores.
    - broken (closure-scoped) impl: call 2 gets a brand-new, empty local
      refusal set, does the full miss cycle, sees NO hash change (idempotent
      this round), and stores a cache entry -- so call 3 (payload reset back
      to dirty again, a fresh wrapper again) would be served that STALE
      cached result instead of re-running `clean`, and `calls` would stop
      growing on the third call.

    Mutation-that-breaks-this-test: scoping `self._refused` (or the guard
    that checks it) to the `_invoke` closure created by one `wrap()` call
    instead of `self`.
    """
    calls = []

    def clean(d):
        calls.append(1)
        time.sleep(0.05)
        d["dirty"] = False
        return len(d)

    payload = {"dirty": True}
    unit = call_unit_harness(lineage={"d": "hash-d"}, user_ns={"d": payload, "clean": clean})

    unit.wrap(clean, _site())(payload)          # call 1: dirty True->False, detected, refused
    unit.wrap(clean, _site())(payload)          # call 2: already clean -- no visible mutation THIS round
    payload["dirty"] = True                     # a fresh cold run would find it dirty again
    unit.wrap(clean, _site())(payload)          # call 3: must still recompute, never a cache hit

    assert calls == [1, 1, 1], "the site must stay refused even once a call looks clean"
    assert payload["dirty"] is False


def test_a_callee_that_mutates_an_unpicklable_argument_is_never_cached(call_unit_harness):
    """`compute_hash` never raises -- its tier-3 fallback for an object that
    fails pickling (holds a `threading.Lock`, a socket, an open file, ...) is
    `sha256(str(id(obj)))`: an IDENTITY hash, invariant across an in-place
    mutation of that same object. Naively diffing two such hashes reads a
    genuine mutation as "unchanged", which is not the "misses a mutation
    outside a sampled byte range" trade `compute_hash`'s sampling already
    makes for large picklable objects (`test_a_callee_that_mutates_its_argument
    _is_never_cached` above) -- it is blind to EVERY mutation, always, for the
    whole unpicklable-object class, and would silently violate "fail closed".

    Mutation-that-breaks-this-test: in `CallUnit._hash_args`, replace
    `out.append(object() if is_identity_fallback_hash(value, h) else h)`
    (i.e. drop the `is_identity_fallback_hash` check and always append `h`).
    The before/after hash tuples then compare equal (id-based hash of the
    SAME object, unchanged by mutation) even though `carrier.value` really
    did change, so the second call is served from cache: `calls == [1]` and
    `carrier.value` stays `101` instead of advancing to `201`.
    """
    import threading

    calls = []

    class Carrier:
        def __init__(self):
            self.lock = threading.Lock()   # unpicklable -> forces the id-based tier
            self.value = 1

    def bump(c):
        calls.append(1)
        time.sleep(0.05)
        c.value += 100
        return "ok"          # the return value is unrelated to what mutated

    carrier = Carrier()
    unit = call_unit_harness(lineage={"c": "hash-c"}, user_ns={"c": carrier, "bump": bump})
    site = _site(source="f(c)", names=("f", "c"))

    unit.wrap(bump, site)(carrier)
    unit.wrap(bump, site)(carrier)

    # Asserting only on the return value ("ok" both times, cached or not)
    # would pass even when the second call was served from cache and the
    # mutation was skipped -- exactly the bug this test exists to catch.
    assert calls == [1, 1], "must recompute -- an unpicklable argument's content is unprovable"
    assert carrier.value == 201


def test_a_refused_call_is_never_stored_through_the_production_dispatch_path(tmp_path):
    """Exercises the refusal through `CallCache.resolve` + `set_sites` --
    the production entry point (`statement/processor.py` resolves a callee
    this way before every intercepted call), not `CallUnit` directly.

    Mutation-that-breaks-this-test: same as the first test -- removing the
    argument-mutation check in `CallUnit.wrap` -- but observed one layer up,
    proving the refusal actually reaches code the notebook runtime calls.
    """
    import cash

    call_cache = CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))
    calls = []

    def clean(d):
        calls.append(1)
        time.sleep(0.2)   # above the cost-model floor
        d["dirty"] = False
        return len(d)

    call_cache.set_sites([_computed_site()])
    resolved = call_cache.resolve(clean)

    payload = {"dirty": True}
    resolved(payload)
    payload["dirty"] = True
    resolved(payload)

    assert calls == [1, 1], "a mutating callee was served from cache via CallCache.resolve"
    assert payload["dirty"] is False


def test_an_unhashable_argument_reads_as_changed_not_unchanged(monkeypatch):
    """`_hash_args`'s `except` branch must fail CLOSED, on every Python.

    It used to append `None` there, reasoning that "'cannot prove unmutated' is
    exactly what a `None` here already means to the caller". It did not:
    `None == None`, so two unknowable snapshots compared EQUAL and read as
    "argument unchanged" — a callee mutating that argument was cached and its
    mutation silently skipped.

    That branch was believed unreachable ("compute_hash's tier-3 fallback
    hashes id(obj), which cannot itself raise"). It is reachable on **every
    Python before 3.14**: hashing an instance of a locally-defined class raises
    `AttributeError: Can't pickle local object '<f>.<locals>.C'` instead of
    falling through to the identity tier. A class defined inside a function is
    ordinary in a notebook and ubiquitous in tests.

    CI caught it — every 3.10-3.13 job red, all three 3.14 jobs green — and a
    fully green local suite did not, because local dev runs 3.14. So this test
    forces the branch directly rather than relying on a version's pickling
    quirk to reach it, and therefore keeps protecting the property on 3.14 and
    on whatever 3.15 does.
    """
    from cash.notebook import call_unit as cu

    def boom(_obj):
        raise AttributeError("Can't pickle local object")

    monkeypatch.setattr(cu, "compute_hash", boom)

    unit = cu.CallUnit.__new__(cu.CallUnit)
    subject = object()
    before = cu.CallUnit._hash_args(unit, (subject,), {})
    after = cu.CallUnit._hash_args(unit, (subject,), {})

    assert len(before) == 1, "the argument must still produce a slot"
    assert before != after, (
        "two unknowable snapshots compared equal — an argument whose content "
        "cannot be hashed is being read as 'unchanged', which is fail-open"
    )
