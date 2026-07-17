"""A parameter's DEFAULT is an input to the result, so it must key the cache.

`@cash.cache` fingerprints the body and the arguments passed at the call site.
Neither covers a default: defaults live on the FUNCTION OBJECT (`__defaults__` /
`__kwdefaults__`), not in the code object, and they are evaluated in the
*enclosing* scope at `def` time -- so they are invisible both to the bytecode
fingerprint the state hash falls back on when source is unavailable (functions
defined in an IPython cell) and to the read-globals fold.

Editing `n_estimators=300` to `400` therefore returned the 300-tree model on an
instant HIT while `inspect.signature` reported 400 (CAS-183). That is worse than
a plain stale read: sweeping a hyperparameter by editing its default returns
identical scores every time, which reads as "accuracy has plateaued" -- a false
finding rather than a visible bug.

The tests below pin both directions. Retraining on a changed default is the fix;
**still hitting on an unchanged default** is the regression that would make the
fix worse than the bug.
"""
from __future__ import annotations

import tempfile
import textwrap
import threading
import warnings
from typing import Any, Callable

import pytest

from cash import Cash, FileBackend
from cash.exceptions import CashCacheIneffectiveWarning


def _cash() -> Cash:
    return Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))


# Execution counter. A method call on a global is excluded from both the
# read-globals fold and the closure fold, so counting can't itself drift the key.
_RUNS: list[str] = []


def _define_in_cell(src: str) -> Callable:
    """Define a function the way an IPython cell does: no source on disk.

    `inspect.getsource` fails for these, so `get_source_hash` falls back to a
    bytecode hash -- the exact path the reporter hit, and the one where defaults
    are invisible. Compiling under a fake `<ipython-input-N>` filename reproduces
    it without a kernel.
    """
    ns: dict[str, Any] = {"_RUNS": _RUNS}
    exec(compile(textwrap.dedent(src), "<ipython-input-1>", "exec"), ns)
    return ns["train"]


_CELL_SRC = """\
def train(x, n_estimators={n}, *, max_depth={d}):
    _RUNS.append('train')
    return ('model', n_estimators, max_depth)
"""


def _make_train(n_default):
    """A default set from an enclosing value, with the source text held constant.

    `n_default` is evaluated at `def` time, so it lands in `train.__defaults__`
    and is NOT a closure capture (the body never names it). `inspect.getsource`
    returns byte-identical text for every `n_default`, so this reproduces the gap
    even when source IS available -- i.e. a plain .py file whose default is
    driven by a config constant, not just a notebook.
    """
    def train(x, n_estimators=n_default):
        _RUNS.append("train")
        return ("model", n_estimators)
    return train


def test_changed_positional_default_retrains():
    """The headline bug: edit 300 -> 400, get the 400 model, not an instant HIT."""
    c = _cash()
    _RUNS.clear()

    first = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=5)))
    assert first(1) == ("model", 300, 5)

    # Same qualname, same body, only the default edited.
    second = c.cache(_define_in_cell(_CELL_SRC.format(n=400, d=5)))
    assert second(1) == ("model", 400, 5), "returned the stale 300 model on a HIT"
    assert len(_RUNS) == 2, "the edited default did not retrain"


def test_changed_kwonly_default_retrains():
    """__kwdefaults__ is a separate dict from __defaults__ and must count too."""
    c = _cash()
    _RUNS.clear()

    first = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=5)))
    assert first(1) == ("model", 300, 5)

    second = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=9)))
    assert second(1) == ("model", 300, 9), "returned the stale max_depth=5 model"
    assert len(_RUNS) == 2


def test_changed_default_retrains_when_source_is_available():
    """Not a notebook-only gap: a config-driven default keeps its source text."""
    c = _cash()
    _RUNS.clear()

    assert c.cache(_make_train(300))(1) == ("model", 300)
    assert c.cache(_make_train(400))(1) == ("model", 400)
    assert len(_RUNS) == 2


def test_unchanged_default_still_hits():
    """The over-invalidation guard: an unchanged default must not force a miss.

    A fix that keys on something drifting (identity, address, per-call re-read of
    a stable value) would pass every test above and destroy caching. This is the
    test that keeps the fix from being worse than the bug.
    """
    c = _cash()
    _RUNS.clear()

    first = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=5)))
    assert first(1) == ("model", 300, 5)
    assert len(_RUNS) == 1

    # Repeated calls on the same object.
    assert first(1) == ("model", 300, 5)
    assert len(_RUNS) == 1, "repeat call on an unchanged default missed"

    # A re-definition with an IDENTICAL default: still the same result, no rerun.
    again = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=5)))
    assert again(1) == ("model", 300, 5)
    assert len(_RUNS) == 1, "re-defining with an unchanged default missed"


def test_no_defaults_keeps_key_byte_identical():
    """Functions with no defaults must not be re-keyed by this fold at all.

    Bounds the one-time cold cache to functions that actually have defaults.
    """
    c = _cash()

    @c.cache
    def f(x, y):
        return x + y

    before = f.explain(1, 2).cache_key
    assert f(1, 2) == 3
    assert f.explain(1, 2).cache_key == before
    # Pinned literal: the key of a defaults-free function is unchanged by CAS-183.
    assert before == c._compute_cache_key(
        c._get_func_key(f.__wrapped__),
        c._fold_read_globals(
            f.__wrapped__,
            c._get_func_key(f.__wrapped__),
            c._fold_bound_self(
                f.__wrapped__,
                c._get_func_key(f.__wrapped__),
                c._fold_closure(
                    f.__wrapped__,
                    c._get_func_key(f.__wrapped__),
                    c._state_hasher.compute(
                        c._get_func_key(f.__wrapped__),
                        own_source_override=c._pin_own_source(f.__wrapped__),
                    ),
                ),
            ),
        ),
        c._resolve_dynamic_dependencies(c._get_func_key(f.__wrapped__), None, (1, 2), {}),
        c._serialize_args(c._get_func_key(f.__wrapped__), (1, 2), {}),
    )


def test_explicitly_passed_arg_still_behaves():
    """Passing the value at the call site was always correct; keep it that way."""
    c = _cash()
    _RUNS.clear()

    f = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=5)))
    assert f(1, n_estimators=400) == ("model", 400, 5)
    assert f(1, n_estimators=500) == ("model", 500, 5)
    assert len(_RUNS) == 2
    # Distinct explicit values stay distinct, and repeats still hit.
    assert f(1, n_estimators=400) == ("model", 400, 5)
    assert len(_RUNS) == 2


def test_omitted_default_equals_explicit_default():
    """The normalization promise survives: f(1) and f(1, n=<default>) share a key."""
    c = _cash()

    f = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=5)))
    assert f.explain(1).cache_key == f.explain(1, n_estimators=300).cache_key
    assert f.explain(1).cache_key != f.explain(1, n_estimators=400).cache_key


def test_normalization_follows_a_redefined_default():
    """The signature memo must not pin the first definition's default.

    `_normalize_call_args` applies defaults through a cached `inspect.Signature`.
    Keyed by name alone it never noticed a redefinition, so `f(1)` kept folding
    the OLD default and no longer matched `f(1, n=<new default>)`.
    """
    c = _cash()
    _RUNS.clear()

    # Must CALL the first definition: the memo is populated on normalization, so
    # a test that only decorates would never load the stale entry and would pass
    # against the bug.
    first = c.cache(_define_in_cell(_CELL_SRC.format(n=300, d=5)))
    first(1)

    g = c.cache(_define_in_cell(_CELL_SRC.format(n=400, d=5)))
    g(1)
    assert g.explain(1).cache_key == g.explain(1, n_estimators=400).cache_key


def test_unhashable_default_fails_safe():
    """An unhashable default must refuse to cache and say so -- never ignore it.

    If cash cannot hash a default it cannot tell whether that default changed.
    Skipping it silently is precisely the stale-result bug; the safe move is to
    run uncached and warn.
    """
    c = _cash()
    _RUNS.clear()

    lock = threading.Lock()

    def f(x, guard=lock):
        _RUNS.append("f")
        return x * 2

    wrapped = c.cache(f)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert wrapped(3) == 6
        assert wrapped(3) == 6

    assert len(_RUNS) == 2, "cached despite an unhashable default it cannot check"
    assert any(
        issubclass(w.category, CashCacheIneffectiveWarning) and "default" in str(w.message)
        for w in caught
    ), f"no warning naming the default: {[str(w.message) for w in caught]}"


def test_mutable_default_is_not_frozen_at_first_call():
    """A mutated mutable default must not be pinned to the first call's value.

    `def f(xs=[])` accumulates across calls; a memo keyed once per function object
    would hand back the first result forever. Re-reading the content per call
    drifts the key, so the call misses and runs -- no speedup, but the true
    answer. Fail-safe, exactly as an argument would behave.
    """
    c = _cash()

    @c.cache
    def f(x, acc=[]):
        acc.append(x)
        return len(acc)

    assert [f(1), f(1), f(1)] == [1, 2, 3], "froze the accumulating default"


def test_defaults_reassigned_in_place_are_detected():
    """`f.__defaults__ = (...)` rebinds on the SAME function object.

    Defaults are memoized per function object to keep the fold off the hot path.
    That memo must be validated against the live containers, not the function's
    identity, or this reintroduces the exact staleness the fold prevents.
    """
    c = _cash()
    _RUNS.clear()

    def f(x, n=300, *, mode="fast"):
        _RUNS.append("f")
        return ("model", n, mode)

    g = c.cache(f)
    assert g(1) == ("model", 300, "fast")

    f.__defaults__ = (400,)
    assert g(1) == ("model", 400, "fast"), "memo pinned the old positional default"

    f.__kwdefaults__ = {"mode": "slow"}
    assert g(1) == ("model", 400, "slow"), "memo pinned the old kw-only default"
    assert len(_RUNS) == 3


def test_read_only_mutable_default_still_hits():
    """A mutable default that is only READ is stable and must keep caching."""
    c = _cash()
    _RUNS.clear()

    @c.cache
    def f(x, opts={"scale": 2}):
        _RUNS.append("f")
        return x * opts["scale"]

    assert f(3) == 6
    assert f(3) == 6
    assert len(_RUNS) == 1, "a stable read-only mutable default over-invalidated"


def test_changed_default_of_wrapped_callee_retrains():
    """A default hidden behind a wrapper still decides the result.

    `inspect.signature` follows `__wrapped__`, so the wrapper's own (empty)
    defaults are not what the call effectively binds.
    """
    import functools

    c = _cash()
    _RUNS.clear()

    def make(n_default):
        def inner(x, n=n_default):
            _RUNS.append("inner")
            return ("model", n)

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        return wrapper

    assert c.cache(make(300))(1) == ("model", 300)
    assert c.cache(make(400))(1) == ("model", 400), "wrapped callee's default ignored"
    assert len(_RUNS) == 2


@pytest.mark.parametrize("default", [None, True, 1.5, "s", (1, 2), frozenset({1})])
def test_assorted_default_types_hit_on_repeat(default):
    """Every immutable default kind must be stable across calls (no drift)."""
    c = _cash()
    _RUNS.clear()

    def f(x, opt=default):
        _RUNS.append("f")
        return (x, opt)

    wrapped = c.cache(f)
    assert wrapped(1) == (1, default)
    assert wrapped(1) == (1, default)
    assert len(_RUNS) == 1, f"default {default!r} drifted its key"
