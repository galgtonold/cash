"""An attribute a traced call binds on an object it made and dropped before
returning changes nothing a hit skips, so the call is stored and served.

``clone(sampler).fit_resample(X, y)`` (imbalanced-learn's walk, here a
stand-in library of the user's whose ``fit_resample`` a decorator wraps)
binds ``est.sampling_strategy_`` on the clone. The strict check of a call
that picks code by name refused that binding as a change to an argument a
hit would skip, though no one can see the clone once the call returns: every
call missed. The recorder now holds each object such a binding was made on
by a weak reference, and a binding whose every object is gone when the call
is judged is no change. One still alive -- returned, kept in a global or on
an argument, held by a cycle no collection has freed yet -- or one no weak
reference can reach keeps the refusal, and each run matches plain Python.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

_SAMPLERS = """
import copy
import functools
from collections import OrderedDict

import numpy as np

try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = object


def _context(fit):
    @functools.wraps(fit)
    def wrapper(estimator, *args, **kwargs):
        return fit(estimator, *args, **kwargs)

    return wrapper


def clone(est):
    return copy.deepcopy(est)


class Sampler:
    def __init__(self, kind="plain"):
        self.kind = kind

    @_context
    def fit_resample(self, X, y):
        _set_strategy(self, y)
        return _dispatch(self, "_fit_" + self.kind, X, y)

    def _fit_plain(self, X, y):
        return np.vstack([X, X * 2]), np.concatenate([y, y])


class SlotSampler:
    # No __weakref__ slot: nothing tells whether a clone of it is gone.
    __slots__ = ("kind", "sampling_strategy_")

    __init__ = Sampler.__init__
    fit_resample = Sampler.__dict__["fit_resample"]
    _fit_plain = Sampler._fit_plain


class SkSampler(Sampler, BaseEstimator):
    pass


def _dispatch(obj, name, *args):
    return getattr(obj, name)(*args)


def _set_strategy(est, y):
    est.sampling_strategy_ = OrderedDict((int(c), int((y == c).sum())) for c in np.unique(y))


class Pipeline:
    def __init__(self, *steps):
        self.steps = list(steps)

    def fit_resample(self, X, y):
        for step in self.steps:
            X, y = step.fit_resample(X, y)
        return X, y
"""

_WALK = """
import gc
import json
import sys
import warnings

import numpy as np

import cash
import samplers

how, cls = sys.argv[1], getattr(samplers, sys.argv[2])
if how == "cycle":
    # Nothing collects the cycle before the call is judged.
    gc.disable()
if cls is samplers.SkSampler:
    from sklearn.base import clone
else:
    clone = samplers.clone

c = cash.Cash(register_magic=False)
RUNS = []
KEPT = []


def body(sampler, X, y):
    if how == "pipeline":
        return samplers.Pipeline(clone(sampler)).fit_resample(X, y)
    est = clone(sampler)
    out = est.fit_resample(X, y)
    if how == "returned":
        return out, est
    if how == "global":
        KEPT.append(est)
    if how == "argument":
        sampler.child_ = est
    if how == "cycle":
        est.me_ = est
    return out


@c.cache
def resample(sampler, X, y):
    RUNS.append(1)  # @cash:assume-safe
    return body(sampler, X, y)


def state(obj):
    if hasattr(obj, "__dict__"):
        return {k: state(v) for k, v in vars(obj).items() if k != "me_"}
    if hasattr(obj, "__slots__"):
        return {k: state(getattr(obj, k)) for k in obj.__slots__ if hasattr(obj, k)}
    if isinstance(obj, tuple):
        return [state(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


X = np.random.default_rng(0).random((20, 3))
y = np.array([0] * 15 + [1] * 5)
warnings.simplefilter("ignore")
for _ in range(3):
    plain = cls()
    want = body(plain, X.copy(), y.copy())
    handed = cls()
    got = resample(handed, X.copy(), y.copy())
    assert state(got) == state(want), (state(got), state(want))
    assert state(handed) == state(plain), (state(handed), state(plain))
why = resample.explain(cls(), X.copy(), y.copy()).details.get("why")
print(json.dumps({"runs": len(RUNS), "hits": resample.cache_info()["hits"], "why": why}))
"""


@pytest.fixture
def walk(tmp_path):
    (tmp_path / "samplers.py").write_text(textwrap.dedent(_SAMPLERS), encoding="utf-8")
    (tmp_path / "walk.py").write_text(textwrap.dedent(_WALK), encoding="utf-8")

    def _run(how: str, cls: str = "Sampler") -> dict:
        out = run_python("walk.py", how, cls, cwd=tmp_path)
        return json.loads(out.stdout.strip().splitlines()[-1])

    return _run


REFUSED = "est.sampling_strategy_ = ... - attribute mutation that changes the argument 'est'"


@pytest.mark.parametrize("how", ["dropped", "pipeline"])
def test_a_clone_made_and_dropped_inside_the_call_hits_on_repeat(walk, how):
    assert walk(how) == {"runs": 1, "hits": 2, "why": None}


@pytest.mark.parametrize("how", ["returned", "global", "argument"])
def test_a_clone_that_outlives_the_call_still_runs_every_time(walk, how):
    got = walk(how)
    assert (got["runs"], got["hits"]) == (3, 0)
    assert REFUSED in got["why"]


def test_a_clone_a_cycle_keeps_alive_still_runs_every_time(walk):
    got = walk("cycle")
    assert (got["runs"], got["hits"]) == (3, 0)
    assert REFUSED in got["why"]


def test_a_dropped_clone_no_weak_reference_reaches_still_runs_every_time(walk):
    got = walk("dropped", "SlotSampler")
    assert (got["runs"], got["hits"]) == (3, 0)
    assert REFUSED in got["why"]


def test_a_sklearn_clone_made_and_dropped_inside_the_call_hits_on_repeat(walk):
    pytest.importorskip("sklearn")
    assert walk("dropped", "SkSampler") == {"runs": 1, "hits": 2, "why": None}
