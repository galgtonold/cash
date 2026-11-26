"""A cached driver whose sampler binds attributes on itself deep inside a
library of the user's is stored and replayed, as when the method is cached.

``resample(sampler, X, y)`` calls ``sampler.fit_resample(X, y)``; the library
(an imbalanced-learn checkout, here a stand-in written in the test folder)
wraps ``fit_resample`` in a decorator, so the key does not fold its code, and
picks a method by a run-time name, so the call is served on a trace whose
strict check judges the code that ran (`cash.decorator.code_trace`). That
check refused ``est.sampling_strategy_ = ...`` two calls deep as a change to
an argument a hit would skip, before attribute replay was considered: every
call missed. An attribute bound on an object every start of that function
was handed the cached call's argument is now left to the attribute verdict
(`cash.decorator.attribute_replay`), which stores and replays it or says why
not; so is ``self.x = ...`` in the ``__init__`` of an object ``Cls(...)``
made inside the call. A change in place, a binding on an object that is no
argument, or a bound value out of the replay scope still runs every time,
with its reason.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

_SAMPLERS = """
import functools
from collections import OrderedDict

import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None


def _context(fit):
    @functools.wraps(fit)
    def wrapper(estimator, *args, **kwargs):
        return fit(estimator, *args, **kwargs)

    return wrapper


class Props:
    def __init__(self, X):
        self.rows = X.shape[0]


class Sampler:
    def __init__(self, kind):
        self.kind = kind
        self.params = {"k": 3}
        self.history = []

    @_context
    def fit_resample(self, X, y):
        props = Props(X)
        _validate(self, X, y)
        if NearestNeighbors is not None:
            self.nn_k_ = NearestNeighbors(n_neighbors=3).fit(X.copy())
        out = _dispatch(self, "_fit_" + self.kind, X, y)
        return out[0].reshape(props.rows * 2, -1), out[1]

    def _fit_plain(self, X, y):
        return np.vstack([X, X * 2]), np.concatenate([y, y])

    _fit_logged = _fit_held = _fit_view = _fit_plain


def _dispatch(obj, name, *args):
    return getattr(obj, name)(*args)


def _validate(est, X, y):
    _set_strategy(est, y)
    if est.kind == "logged":
        _log(est, y)
    elif est.kind == "held":
        _remember(HELD, y)
    elif est.kind == "view":
        _keep_view(est, X)


def _set_strategy(est, y):
    est.sampling_strategy_ = OrderedDict((int(c), int((y == c).sum())) for c in np.unique(y))
    est.params_ = est.params


def _log(est, y):
    est.history.append(len(y))


def _remember(holder, y):
    holder.last_ = len(y)


def _keep_view(est, X):
    est.view_ = X[:10]


HELD = Sampler("plain")
"""

_WALK = """
import json
import sys
import warnings
from collections import OrderedDict

import numpy as np

import cash
import samplers

c = cash.Cash(register_magic=False)
RUNS = []


@c.cache
def resample(sampler, X, y):
    RUNS.append(1)  # @cash:assume-safe
    return sampler.fit_resample(X, y)


kind = sys.argv[1]
X = np.random.default_rng(0).random((60, 3))
y = np.array([0] * 45 + [1] * 15)
warnings.simplefilter("ignore")
for _ in range(3):
    plain = samplers.Sampler(kind)
    want = plain.fit_resample(X.copy(), y.copy())
    # Named as the parameter: the module global it leaves is not what the
    # body calls, and is not keyed.
    sampler = handed = samplers.Sampler(kind)
    got = resample(sampler, X.copy(), y.copy())
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    assert list(vars(handed)) == list(vars(plain)), (list(vars(handed)), list(vars(plain)))
    assert type(handed.sampling_strategy_) is OrderedDict
    assert handed.sampling_strategy_ == plain.sampling_strategy_
    assert handed.params_ is handed.params
    assert handed.history == plain.history
    if samplers.NearestNeighbors is not None:
        assert handed.nn_k_ is not plain.nn_k_
        for left, right in zip(handed.nn_k_.kneighbors(X[:5]), plain.nn_k_.kneighbors(X[:5])):
            np.testing.assert_array_equal(left, right)
why = resample.explain(samplers.Sampler(kind), X.copy(), y.copy()).details.get("why")
print(json.dumps({"runs": len(RUNS), "hits": resample.cache_info()["hits"], "why": why}))
"""


@pytest.fixture
def walk(tmp_path):
    (tmp_path / "samplers.py").write_text(textwrap.dedent(_SAMPLERS), encoding="utf-8")
    (tmp_path / "walk.py").write_text(textwrap.dedent(_WALK), encoding="utf-8")

    def _run(kind: str) -> dict:
        out = run_python("walk.py", kind, cwd=tmp_path)
        return json.loads(out.stdout.strip().splitlines()[-1])

    return _run


def test_the_driver_hits_on_repeat_and_leaves_the_sampler_as_plain_python_does(walk):
    assert walk("plain") == {"runs": 1, "hits": 2, "why": None}


def test_a_second_process_hits_the_stored_entry(walk):
    assert walk("plain")["runs"] == 1
    assert walk("plain") == {"runs": 0, "hits": 3, "why": None}


def test_an_in_place_change_deep_inside_still_runs_every_time(walk):
    got = walk("logged")
    assert (got["runs"], got["hits"]) == (3, 0)
    assert "changed something else in its arguments in place" in got["why"]


def test_a_binding_on_an_object_that_is_no_argument_still_runs_every_time(walk):
    got = walk("held")
    assert (got["runs"], got["hits"]) == (3, 0)
    assert "holder.last_ = ... - attribute mutation that changes the argument 'holder'" in got["why"]


def test_a_bound_value_out_of_the_replay_scope_still_runs_every_time(walk):
    got = walk("view")
    assert (got["runs"], got["hits"]) == (3, 0)
    assert "sampler.view_ shares memory with the argument 'X'" in got["why"]
