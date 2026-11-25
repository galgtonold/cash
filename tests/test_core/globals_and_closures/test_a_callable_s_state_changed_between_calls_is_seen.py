"""What a callable was built with reaches the key in the running process too.

`test_a_callable_is_keyed_the_same_wherever_it_is_held` runs the matrix of
holders and shapes across processes. Here the same state changes between two
calls of one process, after repeated calls have warmed every layer that
remembers a key or a helper's identity: the instance a captured method is
bound to, a number an inner closure captured, an attribute stored on a
function, a list or dict of functions a closure holds.

Also here: the values that must stay OUT of the key because the callable
changes them itself (a call counter on a wrapper, a cache dict it fills), a
callable that holds itself, and a value that cannot be hashed, which runs
the call uncached with a warning rather than being left out.
"""

from __future__ import annotations

import functools
import operator
import threading
import warnings

import pytest

from tests._work_counts import hashed_bytes

pytestmark = pytest.mark.core


class Model:
    def __init__(self, weight):
        self.weight = weight

    def predict(self, x):
        return x * self.weight


def make_scale(k):
    def scale(x):
        return x * k

    return scale


def make_pipeline(k):
    scale = make_scale(k)

    def pipeline(x):
        return scale(x)

    return pipeline


def logged(fn):
    @functools.wraps(fn)
    def wrapper(x):
        return fn(x)

    return wrapper


def make_with_attr(k):
    def scale(x):
        return x * scale.k

    scale.k = k
    return scale


def _warm(fn, *args, times=4):
    """The same call repeated, so every memo that can serve it does."""
    return [fn(*args) for _ in range(times)][-1]


@pytest.fixture
def apply(disk_cash):
    @disk_cash.cache
    def apply(fn, x):
        return fn(x)

    return apply


# --- the cached function's own captures ----------------------------------


def _make_scorer(cache, predict):
    @cache.cache
    def score(x):
        return predict(x)

    return score


def test_a_captured_bound_method_is_keyed_by_its_instance(disk_cash):
    assert _make_scorer(disk_cash, Model(2).predict)(3) == 6
    assert _make_scorer(disk_cash, Model(5).predict)(3) == 15


def test_a_captured_partial_is_keyed_by_what_it_binds(disk_cash):
    assert _make_scorer(disk_cash, functools.partial(operator.mul, 2))(3) == 6
    assert _make_scorer(disk_cash, functools.partial(operator.mul, 5))(3) == 15


def test_a_captured_methods_instance_changed_in_place_recomputes(disk_cash):
    model = Model(2)
    scorer = _make_scorer(disk_cash, model.predict)
    assert _warm(scorer, 3) == 6
    model.weight = 5
    assert _warm(scorer, 3) == 15
    model.weight = 2
    assert scorer(3) == 6
    info = scorer.cache_info()
    assert info["misses"] == 2, info  # the third state is the first one: a hit


def test_a_captured_function_attribute_changed_in_place_recomputes(disk_cash):
    fn = make_with_attr(2)
    scorer = _make_scorer(disk_cash, fn)
    assert _warm(scorer, 3) == 6
    fn.k = 5
    assert _warm(scorer, 3) == 15
    assert scorer.cache_info()["misses"] == 2


# --- a function passed as an argument --------------------------------------


def _make_memoised(k):
    @functools.lru_cache
    def scale(x):
        return x * k

    return scale


def _make_decorated(k):
    @logged
    def scale(x):
        return x * k

    return scale


def _make_from_model(k):
    predict = Model(k).predict
    return lambda x: predict(x)


def _make_from_partial(k):
    mul = functools.partial(operator.mul, k)
    return lambda x: mul(x)


def _make_with_config(k):
    cfg = {"k": k}

    def scale(x):
        return x * cfg["k"]

    def pipeline(x):
        return scale(x)

    return pipeline


def _make_with_steps(k):
    steps = [make_scale(k)]
    return lambda x: steps[0](x)


ARGUMENT_SHAPES = {
    "closure calling a closure": make_pipeline,
    "decorated inner function": _make_decorated,
    "lru_cache'd inner function": _make_memoised,
    "closure over a bound method": _make_from_model,
    "closure over a partial": _make_from_partial,
    "closure calling a closure over a dict": _make_with_config,
    "attribute on the function": make_with_attr,
    "closure over a list of functions": _make_with_steps,
}


@pytest.mark.parametrize("make", ARGUMENT_SHAPES.values(), ids=ARGUMENT_SHAPES.keys())
def test_an_argument_is_keyed_by_what_its_inner_callables_were_built_with(apply, make):
    assert apply(make(2), 3) == 6
    assert apply(make(5), 3) == 15
    assert apply(make(2), 3) == 6
    info = apply.cache_info()
    assert (info["hits"], info["misses"]) == (1, 2), info


def test_a_list_of_functions_is_keyed_by_what_each_was_built_with(disk_cash):
    @disk_cash.cache
    def run_steps(steps, x):
        for step in steps:
            x = step(x)
        return x

    assert run_steps([make_pipeline(2)], 3) == 6
    assert run_steps([make_pipeline(5)], 3) == 15


def test_an_attribute_changed_on_an_argument_function_recomputes(apply):
    fn = make_with_attr(2)
    assert _warm(apply, fn, 3) == 6
    fn.k = 5
    assert _warm(apply, fn, 3) == 15
    del fn.k
    fn.k = 2
    assert apply(fn, 3) == 6
    assert apply.cache_info()["misses"] == 2


def test_a_number_an_inner_closure_captured_changed_by_a_setter_recomputes(apply):
    """Two levels down, behind a remembered identity of the outer function."""

    def make():
        k = 2

        def scale(x):
            return x * k

        def set_k(value):
            nonlocal k
            k = value

        def pipeline(x):
            return scale(x)

        return pipeline, set_k

    pipeline, set_k = make()
    assert _warm(apply, pipeline, 3) == 6
    set_k(5)
    assert _warm(apply, pipeline, 3) == 15
    set_k(2)
    assert apply(pipeline, 3) == 6
    assert apply.cache_info()["misses"] == 2


def test_an_attribute_changed_on_an_inner_function_recomputes(apply):
    inner = make_with_attr(2)

    def pipeline(x):
        return inner(x)

    assert _warm(apply, pipeline, 3) == 6
    inner.k = 5
    assert _warm(apply, pipeline, 3) == 15
    assert apply.cache_info()["misses"] == 2


def test_a_dict_of_functions_an_inner_closure_holds_changed_in_place_recomputes(apply):
    table = {"f": make_scale(2)}

    def pick(x):
        return table["f"](x)

    def pipeline(x):
        return pick(x)

    assert _warm(apply, pipeline, 3) == 6
    table["f"] = make_scale(5)
    assert _warm(apply, pipeline, 3) == 15
    assert apply.cache_info()["misses"] == 2


def test_a_function_default_of_an_inner_function_rebound_recomputes(apply):
    def inner(x, fn=make_scale(2)):
        return fn(x)

    def pipeline(x):
        return inner(x)

    assert _warm(apply, pipeline, 3) == 6
    inner.__defaults__ = (make_scale(5),)
    assert _warm(apply, pipeline, 3) == 15


# --- a module global ----------------------------------------------------------

STEP = _make_with_steps(2)
SCALE = make_with_attr(2)


def test_a_global_closure_over_a_list_of_functions_rebound_recomputes(disk_cash, monkeypatch):
    @disk_cash.cache
    def run_global(x):
        return STEP(x)

    assert _warm(run_global, 3) == 6
    monkeypatch.setitem(globals(), "STEP", _make_with_steps(5))
    assert _warm(run_global, 3) == 15


def test_an_attribute_changed_on_a_global_function_recomputes(disk_cash, monkeypatch):
    @disk_cash.cache
    def run_global(x):
        return SCALE(x)

    assert _warm(run_global, 3) == 6
    monkeypatch.setattr(SCALE, "k", 5)
    assert _warm(run_global, 3) == 15


# --- what stays out, what ends, what cannot be hashed ----------------------


def counted(fn):
    @functools.wraps(fn)
    def wrapper(x):
        wrapper.calls += 1
        return fn(x)

    wrapper.calls = 0
    return wrapper


def memoised(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapper(x):
        if x not in cache:
            cache[x] = fn(x)
        return cache[x]

    wrapper.cache = cache
    return wrapper


@pytest.mark.parametrize("decorator", [counted, memoised], ids=["a call counter", "a cache it fills"])
def test_state_a_wrapper_changes_itself_does_not_keep_the_call_from_hitting(disk_cash, decorator):
    """`wrapper.calls += 1` and `wrapper.cache = cache` are the wrapper's own
    bookkeeping: keyed, every call would miss. (Captured, not passed: a
    function that changes state when called makes a call it is passed to
    run uncached.)"""

    def make(fn):
        @disk_cash.cache(assume_safe=True)
        def run(x):
            return fn(x)

        return run

    run = make(decorator(make_scale(2)))
    with warnings.catch_warnings():
        # A call seen to change what its key read is not stored, once.
        warnings.simplefilter("ignore")
        assert [run(3), run(3), run(3)] == [6, 6, 6]
    before = run.cache_info()["hits"]
    assert [run(3), run(3)] == [6, 6]
    assert run.cache_info()["hits"] == before + 2
    assert make(decorator(make_scale(5)))(3) == 15


def test_a_captured_append_of_a_results_list_does_not_key_the_list(disk_cash):
    """`record = results.append`: what the list holds is the call's output."""
    results: list = []
    record = results.append

    @disk_cash.cache(assume_safe=True)
    def run(x):
        record(x)
        return x * 2

    assert run(3) == 6
    assert run(3) == 6
    assert run.cache_info()["hits"] == 1


def test_a_function_that_holds_itself_is_keyed_and_cached(apply):
    def make(k):
        def fact(n):
            return 1 if n < 2 else n * k * fact(n - 1)

        return fact

    assert apply(make(1), 4) == 24
    assert apply(make(2), 4) == 192
    assert apply(make(1), 4) == 24
    info = apply.cache_info()
    assert (info["hits"], info["misses"]) == (1, 2), info


def test_two_functions_that_hold_each_other_are_keyed_and_cached(apply):
    def make(k):
        def even(n):
            return k if n == 0 else odd(n - 1)

        def odd(n):
            return -k if n == 0 else even(n - 1)

        return even

    assert apply(make(1), 4) == 1
    assert apply(make(7), 4) == 7
    assert apply(make(1), 4) == 1
    info = apply.cache_info()
    assert (info["hits"], info["misses"]) == (1, 2), info


class Holder:
    """Cannot be hashed: it refuses to be pickled."""

    def __init__(self, weight):
        self.weight = weight
        self.lock = threading.Lock()

    def __reduce__(self):
        raise TypeError("a Holder cannot be pickled")

    def predict(self, x):
        return x * self.weight


def test_a_captured_method_of_an_unhashable_object_runs_uncached_with_a_warning(disk_cash):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = _make_scorer(disk_cash, Holder(2).predict)(3)
        second = _make_scorer(disk_cash, Holder(5).predict)(3)
    assert (first, second) == (6, 15)
    assert any("KEY-UNHASHABLE-CAPTURE" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_an_argument_closure_over_a_method_of_an_unhashable_object_runs_uncached(apply):
    def make(k):
        predict = Holder(k).predict
        return lambda x: predict(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert apply(make(2), 3) == 6
        assert apply(make(5), 3) == 15
    assert any("KEY-UNHASHABLE-CAPTURE" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_explain_predicts_the_key_of_a_call_with_a_two_level_closure(apply):
    fn = make_pipeline(2)
    apply(fn, 3)
    assert apply.explain(fn, 3).would_hit
    assert not apply.explain(make_pipeline(5), 3).would_hit


def _make_weighted(weights):
    def weigh(x):
        return float((weights * x).sum())

    return weigh


def _make_weighted_pipeline(weights):
    weigh = _make_weighted(weights)

    def pipeline(x):
        return weigh(x)

    return pipeline


@pytest.mark.parametrize("held", ["argument", "captured"])
def test_a_capture_two_levels_down_is_hashed_once_per_call(disk_cash, apply, held):
    """Work count: each holder of the inner function asked for its state,
    and each time the array it captured was hashed again -- three times a
    call for an argument, twice for a captured helper."""
    np = pytest.importorskip("numpy")
    weights = np.arange(200_000.0)
    fn = _make_weighted_pipeline(weights)
    if held == "argument":

        def call():
            return apply(fn, 2)

    else:
        call = functools.partial(_make_scorer(disk_cash, fn), 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call()
        call()
        with hashed_bytes() as hashed:
            call()
    assert weights.nbytes <= hashed.bytes < 1.5 * weights.nbytes, (hashed.bytes, weights.nbytes)
