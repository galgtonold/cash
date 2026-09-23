"""A call unit refuses to store values that must not be copied."""

import time

from cash.notebook.call_interception import CallSite


def _site(source="f(d)", names=("f", "d")):
    return CallSite(source=source, free_names=frozenset(names), occurrence_index=0)


def test_returning_an_argument_is_never_cached(call_unit_harness):
    """`a = f(d)` where f returns d must keep `a is d`.

    Restoring a deserialised copy silently breaks an identity Python
    guarantees. The statement path can only catch this for a bare bind
    (`b = a`); at the call node the live arguments are in hand, so it is a
    direct identity check.
    """
    calls = []

    def f(d):
        calls.append(1)
        time.sleep(0.05)
        d["k"] = 1
        return d

    payload = {"seed": 0}
    unit = call_unit_harness(lineage={"d": "hash-d"}, user_ns={"d": payload, "f": f})
    wrapped = unit.wrap(f, _site())

    first = wrapped(payload)
    second = wrapped(payload)

    assert first is payload
    assert second is payload, "a cache hit would have returned a copy"
    assert calls == [1, 1], "must recompute, because it must not be stored"


def test_a_scalar_that_happens_to_be_an_argument_is_still_cached(call_unit_harness):
    """`score(1, 10)` returns `1 * 10`, and CPython hands back the very `10`
    object it was passed -- small ints are shared. The identity rule above
    refused it on every run, so the first iteration of a sweep always re-ran
    (found by the tester-session tests). Nothing can rely on an int's identity.
    """
    calls = []

    def score(c, a):
        calls.append((c, a))
        time.sleep(0.05)
        return c * a

    unit = call_unit_harness(lineage={"c": "hash-c", "a": "hash-a"}, user_ns={"c": 1, "a": 10, "score": score})
    wrapped = unit.wrap(score, _site(source="score(c, a)", names=("score", "c", "a")))

    ten = 10
    assert wrapped(1, ten) is ten, "precondition: the result IS the argument"
    assert wrapped(1, ten) == 10
    assert calls == [(1, 10)], "a shared small int blocked the store"


def test_an_identity_coupled_value_is_never_cached(call_unit_harness):
    """Same refusal the statement path applies to a matplotlib Figure."""
    matplotlib = __import__("pytest").importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make_fig():
        time.sleep(0.05)
        return plt.figure()

    unit = call_unit_harness(lineage={}, user_ns={"make_fig": make_fig})
    wrapped = unit.wrap(make_fig, _site(source="make_fig()", names=("make_fig",)))

    a = wrapped()
    b = wrapped()
    assert a is not b, "a cached Figure would hijack pyplot's current figure"
    # `a is not b` alone is trivially true even when caching happens: the RAM
    # tier deep-copies on both store and retrieval, so a hit's returned object
    # is never identical to the original regardless of whether the guard
    # fired. The behaviour actually under test -- that the site is refused,
    # not merely that the returned object differs -- only shows up in
    # whether the second call was logged as a hit at all.
    assert [e["cache_hit"] for e in unit.call_log] == [False, False], (
        "an identity-coupled result must never be served from cache -- the second call has to recompute, not hit"
    )
