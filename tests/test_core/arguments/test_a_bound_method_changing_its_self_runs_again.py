"""``c.cache(est.fit)``: the instance a cached callable is bound to is no
argument of the call, but a body that changes it does what a hit would
not. An estimator whose ``fit`` stores ``self.result_`` was stored, and a
second, equal estimator got the hit and no ``result_``. Such a call is
not stored, as one changing an argument in place is not.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from cash import Cash, FileBackend


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def _set(obj, name, value):
    # Out of the body's sight, as a library's estimator code is.
    setattr(obj, name, value)


class Estimator:
    def __init__(self, k):
        self.k = k

    def fit(self, x):
        _set(self, "result_", np.asarray(x) * self.k)
        return "fitted"

    def score(self, x):
        return float(np.sum(x)) * self.k


def test_a_second_equal_estimator_gets_its_attribute(tmp_path):
    c = _cash(tmp_path)
    first = Estimator(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert c.cache(first.fit)([1, 2, 3]) == "fitted"
    assert any("'self'" in str(w.message) for w in caught), [str(w.message) for w in caught]
    second = Estimator(2)
    assert c.cache(second.fit)([1, 2, 3]) == "fitted"
    np.testing.assert_array_equal(second.result_, [2, 4, 6])


SCORED = []


class Counted(Estimator):
    def score(self, x):
        SCORED.append(x)
        return super().score(x)


def test_a_bound_method_that_leaves_self_alone_still_hits(tmp_path):
    c = _cash(tmp_path)
    SCORED.clear()
    assert c.cache(Counted(3).score)([1, 2]) == 9.0
    assert c.cache(Counted(3).score)([1, 2]) == 9.0
    assert len(SCORED) == 1


def test_an_instance_keyed_by_identity_still_hits(tmp_path):
    """A local class cannot be hashed: keyed by the instance's identity, and
    compared across the body by its attributes' identities."""
    c = _cash(tmp_path)
    calls = []

    class Local:
        k = 2

        def score(self, x):
            calls.append(x)
            return x * self.k

    inst = Local()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cached = c.cache(inst.score)
        assert cached(4) == 8
        assert cached(4) == 8
    assert len(calls) == 1


class Registry:
    @classmethod
    def load(cls, x):
        _set(cls, "loaded_", list(x))
        return len(x)


def test_a_classmethod_setting_a_class_attribute_runs_again(tmp_path):
    c = _cash(tmp_path)
    assert c.cache(Registry.load)([1, 2]) == 2
    del Registry.loaded_
    assert c.cache(Registry.load)([1, 2]) == 2
    assert Registry.loaded_ == [1, 2]


def _fit(est, x):
    _set(est, "result_", list(x))
    return "fitted"


def test_a_partial_binding_an_object_it_changes_runs_again(tmp_path):
    c = _cash(tmp_path)
    c.cache(functools.partial(_fit, Estimator(1)))([1])
    second = Estimator(1)
    c.cache(functools.partial(_fit, second))([1])
    assert second.result_ == [1]


def test_a_partial_around_a_bound_method_runs_again(tmp_path):
    c = _cash(tmp_path)
    c.cache(functools.partial(Estimator(2).fit, [1, 2]))()
    second = Estimator(2)
    c.cache(functools.partial(second.fit, [1, 2]))()
    np.testing.assert_array_equal(second.result_, [2, 4])


def test_a_c_method_is_keyed_on_its_instance(tmp_path):
    c = _cash(tmp_path)
    assert c.cache({1: "a"}.get)(1) == "a"
    assert c.cache({1: "b"}.get)(1) == "b"


def test_a_method_wrapper_is_keyed_on_its_instance(tmp_path):
    c = _cash(tmp_path)
    assert c.cache([1, 2].__len__)() == 2
    assert c.cache([1, 2, 3].__len__)() == 3


def test_a_c_method_changing_its_instance_runs_again(tmp_path):
    c = _cash(tmp_path)
    c.cache([].extend)([1, 2])
    second = []
    c.cache(second.extend)([1, 2])
    assert second == [1, 2]
