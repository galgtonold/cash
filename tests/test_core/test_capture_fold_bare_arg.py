"""A global OR closure capture passed to a call must still invalidate (CAS-270).

`_read_global_data_names` folds module globals a function reads, so reassigning
one invalidates. But it subtracted `_unsafe_uses_of`, which disqualified any
name **passed as a bare argument to any call** — the callee might mutate it, and
folding a mutated global would key the entry on the function's own output and
miss forever.

That exclusion was far wider than its purpose and chose the worse failure.
Measured before the fix, one global, reassigned between two calls:

    sum(G)              STALE      G[0] + G[1]         ok
    len(G)              STALE      sum(v for v in G)   ok
    helper(G)           STALE
    model.predict(G)    STALE   <- the shape that found it: X_train/X_test
                                   read as free variables by a decorated fn

Four of six spellings. Anything handed to a call went stale, silently, forever,
with `.explain()` reporting `[HIT]`.

Globals are folded now and the mutation question is answered by OBSERVATION:
the value is hashed once for the key, then again after the body runs. Changed
across the call => calling the function is what moves it => stop folding that
one name. See `Cash._learn_mutating_globals`.

**The controls are the point of this file.** A "fix" that stopped folding
globals entirely, or that folded them and never demoted an accumulator, would
pass a naive version of the staleness tests:

* `test_a_subscript_read_still_invalidates` / `..._an_iteration_...` — these
  spellings worked BEFORE the fix. They must keep working, or folding was
  disabled wholesale rather than widened.
* `test_an_accumulator_converges_instead_of_missing_forever` — the regression
  the old exclusion existed to prevent. The decorator has no perpetual-miss
  guard (that machinery is statement-path only), so an un-demoted accumulator
  would recompute AND write a fresh entry on every call, unbounded.
* `test_an_unrelated_global_does_not_invalidate` — over-invalidation control.
"""
from __future__ import annotations

import warnings

import pytest

import cash
from cash.exceptions import CashImpurityWarning


@pytest.fixture()
def c(tmp_path):
    return cash.Cash(cache_dir=str(tmp_path / "cache"))


# --- the four spellings that were broken ---------------------------------- #

def test_a_global_passed_to_a_builtin_invalidates(c):
    """The CAS-270 repro: `sum(G)` put G beyond the argument rule."""
    ns = _make_module_ns()
    ns["G"] = [1, 2, 3]

    fn = _define(c, ns, "def f():\n    return sum(G)\n")
    assert fn() == 6
    ns["G"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert fn() == 55, "reassigning a global passed to sum() served stale"


def test_a_global_passed_to_a_user_function_invalidates(c):
    ns = _make_module_ns()
    ns["G"] = [1, 2, 3]

    fn = _define(c, ns, "def f():\n    return helper(G)\n")
    assert fn() == 6
    ns["G"] = [1, 2, 3, 4]
    assert fn() == 10, "a global passed to a user helper served stale"


def test_a_global_passed_to_a_method_invalidates(c):
    """P2's real shape: X_test read as a free variable, handed to model.predict."""
    ns = _make_module_ns()
    ns["G"] = [1, 2, 3]

    fn = _define(c, ns, "def f():\n    return model.predict(G)\n")
    assert fn() == 6
    ns["G"] = [1, 2, 3, 4]
    assert fn() == 10, "a global passed to a method served stale"


def test_a_global_passed_to_len_invalidates(c):
    ns = _make_module_ns()
    ns["G"] = [1, 2, 3]

    fn = _define(c, ns, "def f():\n    return len(G)\n")
    assert fn() == 3
    ns["G"] = [1, 2, 3, 4]
    assert fn() == 4


# --- controls: these worked BEFORE the fix and must still work ------------- #

def test_a_subscript_read_still_invalidates(c):
    ns = _make_module_ns()
    ns["G"] = [1, 2, 3]

    fn = _define(c, ns, "def f():\n    return G[0] + G[1]\n")
    assert fn() == 3
    ns["G"] = [10, 20, 30]
    assert fn() == 30, "folding regressed for a spelling that already worked"


def test_an_iteration_still_invalidates(c):
    ns = _make_module_ns()
    ns["G"] = [1, 2, 3]

    fn = _define(c, ns, "def f():\n    return sum(v for v in G)\n")
    assert fn() == 6
    ns["G"] = [1, 2, 3, 4]
    assert fn() == 10


def test_an_unrelated_global_does_not_invalidate(c):
    """Over-invalidation control: folding must stay scoped to what is READ."""
    ns = _make_module_ns()
    ns["G"] = [1, 2, 3]
    ns["UNRELATED"] = [9, 9, 9]
    calls = []

    ns["_count"] = calls.append
    fn = _define(c, ns, "def f():\n    _count(1)\n    return sum(G)\n")
    assert fn() == 6
    ns["UNRELATED"] = [7]
    assert fn() == 6
    assert len(calls) == 1, "an unrelated global invalidated the entry"


# --- the regression the old exclusion existed to prevent ------------------- #

def test_an_accumulator_converges_instead_of_missing_forever(c):
    """A callee that MUTATES the global must not miss forever.

    Folding a value the call itself moves would key every entry on the previous
    call's output. The decorator has no perpetual-miss guard, so that would
    recompute AND write a new entry every call, without bound.

    Expected shape: one learning miss, then stable. `[1, 2, 2, 2, 2]` — call 1
    runs, call 2 misses because the key changed when the name was demoted, and
    calls 3+ hit.
    """
    ns = _make_module_ns()
    ns["ACC"] = []

    fn = _define(c, ns, "def f():\n    return mutate(ACC)\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vals = [fn() for _ in range(5)]

    assert vals[2:] == [vals[2]] * 3, f"never converged: {vals}"
    assert len(ns["ACC"]) <= 2, f"accumulator kept growing: {ns['ACC']}"
    ours = [w for w in caught
            if issubclass(w.category, CashImpurityWarning)
            and "modifies the module global" in str(w.message)]
    assert len(ours) == 1, f"expected exactly one warning, got {len(ours)}"
    assert "'ACC'" in str(ours[0].message)


# --- helpers --------------------------------------------------------------- #

def _make_module_ns() -> dict:
    """A fresh module-like namespace with the callees the tests reference.

    Each test gets its own, so a demoted name in one cannot leak into another
    (the demotion is keyed on the code object, and `_define` compiles a new one
    per call).
    """
    class _M:
        def predict(self, x):
            return sum(x)

    def helper(x):
        return sum(x)

    def mutate(x):
        x.append(len(x))
        return len(x)

    return {"__name__": "cash_test_ns", "helper": helper,
            "model": _M(), "mutate": mutate, "sum": sum, "len": len}


def _define(c, ns: dict, src: str):
    """Compile *src* into *ns* so its function has real module globals.

    `inspect.getsource` must be able to find it, so the source is registered
    with linecache under a unique fake filename.
    """
    import linecache
    import uuid

    name = f"<cash-test-{uuid.uuid4().hex}>"
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    exec(compile(src, name, "exec"), ns)  # noqa: S102 - fixture construction
    return c.cache(ns["f"])


# --- the same bug, one scope over: closure captures ------------------------ #
#
# `_fold_closure` shares `_unsafe_uses_of`, so a captured variable handed to a
# call was excluded from the key for the same reason and went stale the same
# way. Measured before the fix, with `data` captured and then rebound via
# `nonlocal`: `sum(data)` STALE, `data[0] + data[1]` ok, `sum(v for v in data)`
# ok -- the identical split.


def _make_closure(c, kind: str):
    """A closure over `data`, plus a setter that rebinds it in the factory scope."""
    data = [1, 2, 3]

    class _M:
        def predict(self, x):
            return sum(x)

    if kind == "bare":
        @c.cache
        def inner():
            return sum(data)
    elif kind == "method":
        @c.cache
        def inner():
            return _M().predict(data)
    elif kind == "subscript":
        @c.cache
        def inner():
            return data[0] + data[1]
    else:
        @c.cache
        def inner():
            return sum(v for v in data)

    def setter(v):
        nonlocal data
        data = v

    return inner, setter


@pytest.mark.parametrize("kind", ["bare", "method"])
def test_a_capture_passed_to_a_call_invalidates(c, kind):
    inner, setter = _make_closure(c, kind)
    assert inner() == 6
    setter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert inner() == 55, f"a capture passed via {kind} served stale"


@pytest.mark.parametrize("kind,expected", [("subscript", 30), ("iterate", 55)])
def test_a_capture_read_without_a_call_still_invalidates(c, kind, expected):
    """Control: these spellings worked BEFORE the fix and must still work."""
    inner, setter = _make_closure(c, kind)
    inner()
    setter([10, 20, 30] if kind == "subscript" else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert inner() == expected


def test_a_mutated_capture_converges_instead_of_missing_forever(c):
    """The closure twin of the accumulator control. Same trap, same fallback."""
    acc: list = []

    def bump(x):
        x.append(len(x))
        return len(x)

    @c.cache
    def inner():
        return bump(acc)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vals = [inner() for _ in range(5)]

    assert vals[2:] == [vals[2]] * 3, f"never converged: {vals}"
    assert len(acc) <= 2, f"capture kept growing: {acc}"
    ours = [w for w in caught
            if issubclass(w.category, CashImpurityWarning)
            and "variable it captures" in str(w.message)]
    assert len(ours) == 1, f"expected exactly one warning, got {len(ours)}"


def test_two_closures_from_one_factory_do_not_collide(c):
    """`_fold_closure`'s original reason for existing, unaffected by the change.

    Two closures from the same factory share source AND qualname, so without
    the capture fold they collide on one key and return each other's results.
    """
    def factory(n):
        @c.cache
        def f():
            return sum([n, n])
        return f

    assert factory(2)() == 4
    assert factory(5)() == 10
