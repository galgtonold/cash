"""A cached function returning an unseeded fitted estimator must say so.

Round-14 gate finding (WRONG). Using the docs' own ML recipe, three runs
returned the identical model -- same internal `random_state`, same
feature-importance fingerprint -- with no `CashRandomnessWarning` and no
`unseeded` badge marker. The reporter's control showed the signal exists on the
bare-statement path and is absent on the decorator path, i.e. present on the
route the docs discourage and missing on the one they recommend.

`decorator.md` is right that source-based detection cannot see randomness inside
sklearn's compiled `.fit()`. The notebook's statement path solves that by asking
the LIVE object instead of the source; this gives the decorator the same answer.

The harm is not a corrupted number, it is a corrupted conclusion -- in the
reporter's words, "I would have written 'the model is completely stable across
random seeds' in a report."
"""
from __future__ import annotations

import warnings

import pytest

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash import CashRandomnessWarning


class _Estimator:
    """Duck-types the one method the check uses. No sklearn needed."""

    def __init__(self, random_state):
        self._rs = random_state

    def get_params(self, deep=True):
        return {"n_estimators": 100, "random_state": self._rs}


class _Deterministic:
    """An estimator with no randomness at all, e.g. LinearRegression."""

    def get_params(self, deep=True):
        return {"fit_intercept": True}


@pytest.fixture
def cash_instance():
    c = Cash(backend=InMemoryBackend(), register_magic=False)
    yield c
    c.backend.clear()


def _warnings_from(fn, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args)
    return [str(w.message) for w in caught
            if issubclass(w.category, CashRandomnessWarning)]


def test_an_unseeded_returned_estimator_warns(cash_instance):
    @cash_instance.cache
    def train(n):
        return _Estimator(random_state=None)

    got = _warnings_from(train, 1)
    assert got, "no CashRandomnessWarning for a returned random_state=None estimator"
    assert "random_state=None" in got[0]
    assert "frozen" in got[0]


def test_a_seeded_estimator_is_silent(cash_instance):
    @cash_instance.cache
    def train(n):
        return _Estimator(random_state=42)

    assert _warnings_from(train, 1) == []


def test_an_estimator_with_no_random_state_is_silent(cash_instance):
    """`LinearRegression` has no such parameter; warning about it is noise."""
    @cash_instance.cache
    def train(n):
        return _Deterministic()

    assert _warnings_from(train, 1) == []


def test_allow_random_silences_it(cash_instance):
    @cash_instance.cache(allow_random=True)
    def train(n):
        return _Estimator(random_state=None)

    assert _warnings_from(train, 1) == []


def test_an_ordinary_return_value_is_untouched(cash_instance):
    """The check must not fire -- or cost anything visible -- on normal results."""
    @cash_instance.cache
    def add(a, b):
        return a + b

    assert _warnings_from(add, 1, 2) == []
    assert add(1, 2) == 3


def test_a_broken_get_params_cannot_break_the_call(cash_instance):
    """An advisory must never take down a user's function."""
    class _Hostile:
        def get_params(self, deep=True):
            raise RuntimeError("nope")

    @cash_instance.cache
    def train(n):
        return _Hostile()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert isinstance(train(1), _Hostile)
