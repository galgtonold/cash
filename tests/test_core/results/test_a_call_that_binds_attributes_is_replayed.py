"""A cached call that binds, rebinds or deletes attributes of an instance it
was handed is stored with them, and a hit replays them on the instance the
caller hands in.

``est.fit(X)`` setting ``self.coef_`` (an estimator's ``fit``, a resampler's
``fit_resample``) changed its argument in place, so it ran every time. When
that is all it changed -- top-level attributes of an instance of a plain
class, each bound to a new object of the call's own, an argument, the
instance itself or a module global -- the entry keeps them and a hit binds
them again: the instance then holds what a plain run leaves, the same
attributes in the same order, the same objects where the run bound an
argument or a global, the same aliasing between attributes and with the
result.

Anything else keeps the refusal: a value that was there before changed in
place, slots, a class with its own ``__setattr__``, a property of that name,
an object something outside the call also holds, an array argument written
in place. Each case below runs the function plainly once and checks the
cached calls leave exactly that state, or run every time.
"""

from __future__ import annotations

import dataclasses
import sys
import textwrap
import threading
import types
import warnings

import numpy as np
import pytest

from cash import rewrite_in_place
from tests._scripts import run_python

REGISTRY = ["a module global"]
SHARED = {"pool": [1, 2]}


def _content(value, obj=None):
    if value is obj and obj is not None:
        return "obj"
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.tolist())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_content(v, obj) for v in value])
    if isinstance(value, dict):
        return ("dict", [(k, _content(v, obj)) for k, v in value.items()])
    if isinstance(value, SlotView):
        return ("SlotView", _content(value.held, obj))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return (type(value).__name__, [(k, _content(v, obj)) for k, v in vars(value).items()])
    return value


def _state(obj, result, named):
    """What a run left: each attribute in order, by content and by which
    object it is -- *obj* itself, a named outside object, or the n-th
    distinct object met (so aliasing shows) -- and the same for the result."""
    seen: dict[int, int] = {}

    def ident(value):
        if value is obj:
            return "obj"
        for name, held in named.items():
            if value is held:
                return name
        if type(value) in (int, float, str, bool, type(None)):
            return "atom"
        return seen.setdefault(id(value), len(seen))

    attrs = [(key, ident(value), _content(value, obj)) for key, value in vars(obj).items()]
    return attrs, ident(result), _content(result, obj)


def _check(disk_cash, kind, method, make_args=lambda: (np.arange(6.0).reshape(3, 2),), calls=3):
    """Run *method* plainly on a fresh *kind*, then cached *calls* times on
    fresh ones: ``(runs of the cached body, whether every cached call left
    the plain run's state)``."""
    args = make_args()
    obj = kind()
    plain = _state(obj, method(obj, *args), {f"arg{i}": a for i, a in enumerate(args)} | _GLOBALS)
    runs = []

    def body(obj, *args):
        runs.append(1)
        return method(obj, *args)

    body.__qualname__ = body.__name__ = f"{kind.__name__}_{method.__name__}"
    cached = disk_cash.cache(body)
    same = True
    kept = []
    for _ in range(calls):
        args = make_args()
        obj = kind()
        result = cached(obj, *args)
        same = same and _state(obj, result, {f"arg{i}": a for i, a in enumerate(args)} | _GLOBALS) == plain
        kept.append(obj)
    return len(runs), same, kept


_GLOBALS = {"REGISTRY": REGISTRY, "SHARED": SHARED}


class Est:
    def __init__(self):
        self.k = 3


def fit_fresh(self, X):
    self.coef_ = X.sum(0) * self.k
    self.n_ = len(X)
    self.shape_ = X.shape
    return "fitted"


def fit_relink(self, X):
    self.X_ = X
    self.me_ = self
    self.registry_ = REGISTRY
    self.kind_ = np.ndarray
    return None


def fit_return_self(self, X):
    self.coef_ = X.mean(0)
    return self


def fit_alias(self, X):
    self.a_ = self.b_ = [float(X.sum())]
    self.c_ = {"of": self.a_}
    return self.a_


def fit_tuple(self, X):
    self.lo_, self.hi_ = X.min(0), X.max(0)
    return self.lo_, self.hi_


def fit_delete(self, X):
    del self.k
    self.k_used_ = 3
    return 1


def fit_rebind(self, X):
    self.k = self.k + 1
    return 1


def fit_nested(self, X):
    inner = Est()
    inner.w = X[0].copy()
    self.model_ = inner
    return 1


@pytest.mark.parametrize(
    "method",
    [fit_fresh, fit_relink, fit_return_self, fit_alias, fit_tuple, fit_delete, fit_rebind, fit_nested],
)
def test_a_call_that_binds_attributes_is_replayed(disk_cash, method):
    runs, same, kept = _check(disk_cash, Est, method)
    assert runs == 1
    assert same
    # Each hit binds objects of its own, never one another hit bound.
    for name, value in vars(kept[1]).items():
        if isinstance(value, (np.ndarray, list, dict, Est)) and value is not REGISTRY:
            assert getattr(kept[2], name) is not value, name


def test_return_self_hands_back_the_caller_s_object(disk_cash):
    cached = disk_cash.cache(fit_return_self)
    for _ in range(2):
        est = Est()
        assert cached(est, np.ones((2, 2))) is est
        assert est.coef_.tolist() == [1.0, 1.0]


class History(Est):
    def __init__(self):
        super().__init__()
        self.history = []
        self.cfg = Est()


def fit_append(self, X):
    self.history.append(len(X))
    self.coef_ = X.sum(0)
    return 1


def fit_nested_change(self, X):
    self.cfg.k = 9
    self.coef_ = X.sum(0)
    return 1


def fit_swap(self, X):
    self.coef_ = self.history
    self.history = []
    return 1


def fit_hold_global(self, X):
    self.reg_ = {"r": REGISTRY}
    return 1


def fit_hold_pooled(self, X):
    self.pool_ = SHARED["pool"]
    return 1


class SlotView:
    __slots__ = ("held",)

    def __init__(self, held):
        self.held = held


def fit_slot_view(self, X):
    # A new object, holding (in a slot) the list the instance already had.
    self.view_ = SlotView(self.history)
    return 1


def fit_write_x(self, X):
    X *= 2
    self.n_ = 1
    return 1


@pytest.mark.parametrize(
    "method",
    [fit_append, fit_nested_change, fit_swap, fit_hold_global, fit_hold_pooled, fit_slot_view, fit_write_x],
)
def test_what_a_hit_cannot_replay_runs_every_time(full_warning_text, disk_cash, method):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs, same, _ = _check(disk_cash, History, method)
    assert runs == 3
    assert same
    assert any("in place" in str(w.message) for w in caught), [str(w.message) for w in caught]


class Slotted:
    __slots__ = ("k", "coef_")

    def __init__(self):
        self.k = 3


class Guarded(Est):
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class Shadowed(Est):
    @property
    def coef_(self):
        return self.__dict__["coef_"]

    @coef_.setter
    def coef_(self, value):
        self.__dict__["coef_"] = value


def fit_coef(self, X):
    self.coef_ = X.sum(0)
    return 1


@pytest.mark.parametrize("kind", [Guarded, Shadowed], ids=["own __setattr__", "a property of the name"])
def test_a_class_that_controls_its_attributes_runs_every_time(disk_cash, kind):
    runs, same, _ = _check(disk_cash, kind, fit_coef)
    assert runs == 3
    assert same


def test_slots_run_every_time(disk_cash):
    runs = []

    @disk_cash.cache
    def fit(obj, X):
        runs.append(1)
        obj.coef_ = X.sum(0)

    for _ in range(3):
        obj = Slotted()
        fit(obj, np.ones(3))
        assert obj.coef_ == 3.0
    assert len(runs) == 3


class Stored(Est):
    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        self._total = value


def fit_setter(self, X):
    self.total = float(X.sum())
    return 1


SET_CALLS = []


class Counted(Est):
    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        SET_CALLS.append(value)
        self._total = value


def test_a_property_setter_that_only_binds_is_replayed(disk_cash):
    """A setter whose body is only ``self._total = value`` changes nothing
    but the instance ``__dict__``, so the binding it makes is replayed."""
    runs, same, _ = _check(disk_cash, Stored, fit_setter)
    assert (runs, same) == (1, True)


def test_a_property_setter_that_runs_other_code_runs_every_time(disk_cash):
    """A setter that does more than bind is code a hit would not run, so
    a class that has one keeps the call running every time."""
    runs, same, _ = _check(disk_cash, Counted, fit_setter)
    assert (runs, same) == (3, True)


class Dropping(Est):
    """Its pickle leaves ``cache_`` out."""

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("cache_", None)
        return state


def fit_dropping(self, X):
    inner = Dropping()
    inner.cache_ = [1]
    self.model_ = inner
    return 1


@dataclasses.dataclass(frozen=True)
class Frozen:
    k: int = 3


def fit_frozen(self, X):
    object.__setattr__(self, "coef_", X.sum(0))
    return 1


def test_a_value_whose_pickle_leaves_state_out_runs_every_time(disk_cash):
    runs, same, _ = _check(disk_cash, Est, fit_dropping)
    assert (runs, same) == (3, True)


def test_a_frozen_dataclass_runs_every_time(disk_cash):
    runs, same, _ = _check(disk_cash, Frozen, fit_frozen)
    assert (runs, same) == (3, True)


GLOBAL_PATH = types.SimpleNamespace(target=None)


def test_an_object_the_function_reaches_through_a_global_runs_every_time(disk_cash):
    """The body writes through ``GLOBAL_PATH.target``, which is the argument
    when it is stored: a hit would bind on the argument handed in, where a
    plain run writes on whatever the global holds."""
    runs = []

    @disk_cash.cache
    def fit(est, n):
        runs.append(1)
        GLOBAL_PATH.target.x = n * 2
        return 1

    for _ in range(2):
        est = Est()
        GLOBAL_PATH.target = est
        fit(est, 3)
        assert est.x == 6
    assert len(runs) == 2


@pytest.mark.skipif(rewrite_in_place.cwatch is None, reason="the thread a change came from is told by the C watcher")
def test_a_change_from_another_thread_runs_every_time(disk_cash):
    runs = []

    @disk_cash.cache
    def fit(est, X):
        runs.append(1)
        est.coef_ = X.sum(0)
        helper = threading.Thread(target=setattr, args=(est, "seen_", True))
        helper.start()
        helper.join()
        return 1

    for _ in range(2):
        est = Est()
        fit(est, np.ones(2))
        assert est.seen_ is True and est.coef_ == 2.0
    assert len(runs) == 2


def test_an_array_argument_written_in_place_is_still_seen(disk_cash):
    runs = []

    @disk_cash.cache
    def fit(obj, X):
        runs.append(1)
        X *= 2
        obj.n_ = 1

    for _ in range(3):
        X = np.ones(3)
        fit(Est(), X)
        assert X.tolist() == [2.0, 2.0, 2.0]
    assert len(runs) == 3


class Estimator:
    """Shaped like a scikit-learn estimator: parameters set in ``__init__``,
    ``fit`` binding trailing-underscore attributes and returning ``self``."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def get_params(self):
        return {"alpha": self.alpha}

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0] * self.alpha
        self.sampling_strategy_ = {int(c): int((y == c).sum()) for c in self.classes_}
        return self

    def predict(self, X):
        return X @ self.coef_


def test_an_estimator_fit_is_replayed(disk_cash):
    runs = []

    @disk_cash.cache
    def fit(est, X, y):
        runs.append(1)
        return est.fit(X, y)

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0])
    plain = Estimator(2.0).fit(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            est = Estimator(2.0)
            assert fit(est, X.copy(), y.copy()) is est
            assert list(vars(est)) == list(vars(plain))
            assert est.sampling_strategy_ == plain.sampling_strategy_
            np.testing.assert_array_equal(est.predict(X), plain.predict(X))
    assert len(runs) == 1
    assert not [w for w in caught if "in place" in str(w.message)], [str(w.message) for w in caught]


class Model:
    def __init__(self):
        self.k = 2

    def fit(self, X):
        self.coef_ = X * self.k
        return self


def test_a_method_setting_its_own_attributes_says_nothing(full_warning_text, disk_cash):
    """The source shows ``self.coef_ = ...``: once the call is replayed, no
    warning says it changes ``self`` or runs every time."""
    fit = disk_cash.cache(Model.fit)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            m = Model()
            assert fit(m, np.ones(2)) is m
            assert m.coef_.tolist() == [2.0, 2.0]
    assert fit.cache_info()["hits"] == 1
    assert not [w for w in caught if "'self'" in str(w.message)], [str(w.message) for w in caught]


def test_without_the_c_watcher_the_attributes_are_compared(disk_cash, monkeypatch):
    """Below Python 3.12, or without the C extension, the ``__dict__`` is
    copied before the body and compared after it."""
    monkeypatch.setattr(rewrite_in_place, "cwatch", None)
    for method in (fit_fresh, fit_alias, fit_delete):
        runs, same, _ = _check(disk_cash, Est, method)
        assert (runs, same) == (1, True), method
    runs, same, _ = _check(disk_cash, History, fit_append)
    assert (runs, same) == (3, True)


@pytest.mark.timeout(120)
def test_a_second_process_replays_what_the_first_stored(tmp_path):
    script = tmp_path / "fit.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import sys
            import numpy as np
            import cash

            c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)
            LOG = sys.argv[2]
            DEFAULT = object()

            class Est:
                def __init__(self):
                    self.k = 3

            @c.cache
            def fit(est, X):
                fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
                os.write(fd, b"ran\\n")
                os.close(fd)
                est.coef_ = X * est.k
                est.alias_ = est.coef_
                est.X_ = X
                est.default_ = DEFAULT
                del est.k
                return est

            for _ in range(2):
                X = np.arange(4.0)
                est = Est()
                assert fit(est, X) is est
                assert list(vars(est)) == ["coef_", "alias_", "X_", "default_"], vars(est)
                assert est.coef_.tolist() == [0.0, 3.0, 6.0, 9.0]
                assert est.alias_ is est.coef_ and est.X_ is X and est.default_ is DEFAULT
            print("ok")
            """
        ),
        encoding="utf-8",
    )
    log = tmp_path / "runs.log"
    for _ in range(2):
        out = run_python(script, tmp_path / ".cash", log, cwd=tmp_path)
        assert out.stdout.strip() == "ok", out.stderr
    assert log.read_text(encoding="utf-8").count("ran") == 1


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="the fallback without dict watchers is below Python 3.12")
def test_below_3_12_the_copy_compare_path_is_taken():
    assert rewrite_in_place.cwatch is None


def test_a_cached_bound_method_replays_on_its_instance(disk_cash):
    """``c.cache(est.fit)``: the instance is no argument, the key folds it;
    a hit binds the attributes on the instance the callable is bound to and
    hands back that instance where the call returned ``self``."""
    runs = []

    class_fit = Model.fit

    def counted(self, X):
        runs.append(1)
        return class_fit(self, X)

    for _ in range(3):
        m = Model()
        fit = disk_cash.cache(counted.__get__(m))
        assert fit(np.ones(2)) is m
        assert list(vars(m)) == ["k", "coef_"]
        assert m.coef_.tolist() == [2.0, 2.0]
    assert len(runs) == 1


class Keyed:
    """Keyed by what ``__cash_key__`` says when it is an argument; held
    inside another object, by all it holds."""

    def __init__(self):
        self.k = 2

    def __cash_key__(self):
        return "keyed"


class HoldsKeyed(Est):
    def __init__(self):
        super().__init__()
        self.part = Keyed()


def test_an_instance_holding_a_cash_key_object_is_replayed(disk_cash):
    runs, same, kept = _check(disk_cash, HoldsKeyed, fit_coef)
    assert (runs, same) == (1, True)
    assert kept[-1].coef_.tolist() == [6.0, 9.0]
