"""A setting the call itself changes around a served cached call is checked on its entry value.

scikit-learn's ``_fit_context`` runs a ``fit`` inside
``config_context(skip_parameter_validation=True)`` and puts the setting back
as it leaves. A cached call served inside it (imbalanced-learn's
``SMOTE._fit_resample`` under a decorator walk, decorated in the project's
own code) brought the settings its entry was checked on to the call around
it, as that entry read them: ``skip_parameter_validation=True``. The outer
call had read the setting itself before ``_fit_context`` changed it, and kept
both values. Its next lookup, outside the context, could never match the
inner one, so ``driver.resample`` and ``driver.evaluate`` missed on every
commit of the walk with identical keys ("library setting changed:
sklearn._config._threadlocal.global_config") and were stored again.

The rule: the earliest read of a place in a call is what its result came of.
An entry served after the call had read that place itself was served under
what the call's own code set there, and the outer entry is checked on what
the call first read; an entry served before the call read it, or a place the
call never read itself, still brings its value.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo
from tests._scripts import run_python

np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")
sk_base = pytest.importorskip("sklearn.base")
sk_utils = pytest.importorskip("sklearn.utils")

pytestmark = [
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.skipif(
        sys.version_info < (3, 12), reason="a body's reads are recorded with sys.monitoring, Python 3.12 or newer"
    ),
]

X = np.arange(10.0)


@pytest.fixture(autouse=True)
def sklearn_config():
    """Whatever a test sets is put back as it leaves."""
    with sklearn.config_context():
        yield


@pytest.fixture(params=["ordinary", "state memo", "fast"])
def path(request, monkeypatch):
    if request.param == "ordinary":
        monkeypatch.setattr(hit_memo, "ENABLED", False)
        monkeypatch.setattr(state_memo, "ENABLED", False)
    elif request.param == "state memo":
        if not state_memo.ENABLED:
            pytest.skip("the state memo needs the C extension")
        monkeypatch.setattr(hit_memo, "ENABLED", False)
    elif not hit_memo.ENABLED:
        pytest.skip("the fast hit path needs the C extension")
    return request.param


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _pair(c):
    """``resample(X, scale)``: a sampler whose ``fit_resample`` runs in
    ``_fit_context`` and calls the cached ``inner``, which reads the config
    (``check_array``)."""

    @c.cache
    def inner(X):
        return sk_utils.check_array(X.reshape(-1, 1))[::2, 0]

    class Sampler(sk_base.BaseEstimator):
        _parameter_constraints: dict = {}

        def __init__(self, scale=1):
            self.scale = scale

        @sk_base._fit_context(prefer_skip_nested_validation=True)
        def fit_resample(self, X):
            return inner(X) * self.scale

    @c.cache
    def resample(X, scale):
        return Sampler(scale=scale).fit_resample(X)

    return inner, resample


def _hits(f):
    return f.cache_info()["hits"]


def test_an_entry_served_inside_fit_context_does_not_hold_the_outer_call_to_it(c, path):
    inner, resample = _pair(c)
    resample(X, 1)
    # ``inner(X)`` is served inside the second call's ``_fit_context``.
    hits = _hits(inner)
    first = resample(X, 2)
    assert _hits(inner) == hits + 1, "positive control: the inner entry was not served"
    hits = _hits(resample)
    assert np.array_equal(resample(X, 2), first)
    assert _hits(resample) == hits + 1, resample.cache_info()["miss_reasons"]


def test_the_outer_call_is_still_checked_on_the_config_it_entered_with(c, path):
    inner, resample = _pair(c)
    resample(X, 1)
    resample(X, 2)
    resample(X, 2)
    hits = _hits(resample)
    sklearn.set_config(assume_finite=True)
    resample(X, 2)
    assert _hits(resample) == hits, "served the entry made under assume_finite=False"
    assert "library setting changed" in resample.cache_info()["miss_reasons"]


def test_a_call_made_inside_the_user_s_own_config_context_is_served_only_inside_it(c, path):
    inner, resample = _pair(c)
    with sklearn.config_context(skip_parameter_validation=True):
        resample(X, 1)
        resample(X, 2)
        hits = _hits(resample)
        resample(X, 2)
        assert _hits(resample) == hits + 1, resample.cache_info()["miss_reasons"]
    hits = _hits(resample)
    resample(X, 2)
    assert _hits(resample) == hits, "served outside the block what was stored inside it"
    hits = _hits(resample)
    resample(X, 2)
    assert _hits(resample) == hits + 1, resample.cache_info()["miss_reasons"]
    with sklearn.config_context(skip_parameter_validation=True):
        # One entry a key: the one stored outside the block is not served in it.
        hits = _hits(resample)
        resample(X, 2)
        assert _hits(resample) == hits, "served inside the block what was stored outside it"
        resample(X, 2)
        assert _hits(resample) == hits + 1, resample.cache_info()["miss_reasons"]


def test_a_setting_only_the_served_entry_read_is_still_checked(c, path):
    @c.cache
    def output():
        return sklearn.get_config()["transform_output"]

    @c.cache
    def outer(n):
        return [output()] * n

    outer(1)
    outer(2)  # ``output()`` served: the outer call read no config itself
    hits = _hits(outer)
    assert outer(2) == ["default"] * 2
    assert _hits(outer) == hits + 1, outer.cache_info()["miss_reasons"]
    sklearn.set_config(transform_output="pandas")
    assert outer(2) == ["pandas"] * 2


def test_an_entry_served_inside_a_pandas_option_context_does_not_hold_the_outer_call_to_it(c, path):
    pd = pytest.importorskip("pandas")

    @c.cache
    def precision():
        return pd.get_option("display.precision")

    @c.cache
    def outer(n):
        with pd.option_context("display.precision", 2):
            return [precision()] * n

    outer(1)
    outer(2)
    hits = _hits(outer)
    assert outer(2) == [2, 2]
    assert _hits(outer) == hits + 1, outer.cache_info()["miss_reasons"]
    with pd.option_context("display.max_rows", 7):
        # Another entry of the options than the call read: still served.
        hits = _hits(outer)
        outer(2)
        assert _hits(outer) == hits + 1, outer.cache_info()["miss_reasons"]


SCRIPT = """
import sys
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils import check_array
from cash import Cash

c = Cash(cache_dir={cache!r}, register_magic=False)


@c.cache
def inner(X):
    return check_array(X.reshape(-1, 1))[::2, 0]


class Sampler(BaseEstimator):
    _parameter_constraints = {{}}

    def __init__(self, scale=1):
        self.scale = scale

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_resample(self, X):
        return inner(X) * self.scale


@c.cache
def resample(X, scale):
    return Sampler(scale=scale).fit_resample(X)


X = np.arange(10.0)
resample(X, 1)
resample(X, 2)
print(resample.cache_info()["hits"])
"""


def test_the_outer_call_is_served_in_the_next_process(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(SCRIPT.format(cache=str(tmp_path / "cache"))), encoding="utf-8")

    def run():
        return run_python(script, cwd=tmp_path, cache_dir=tmp_path / "cache").stdout.split()

    assert run() == ["0"]
    assert run() == ["2"]
    assert run() == ["2"]
