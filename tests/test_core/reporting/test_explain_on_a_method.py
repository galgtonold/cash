"""``explain()`` on a cached method answers for the call it names, or refuses.

``m.score.explain(2)`` returned ``no_entry`` for a call ``m.score(2)`` that
then hit: ``explain`` is an attribute of the function, not of the bound
method, so the instance was never passed and ``2`` was taken as ``self``.
The method spelling is ``Model.score.explain(m, 2)``; the other one now
raises instead of answering for another call.
"""

from __future__ import annotations

import pytest


class Model:
    def __init__(self, k):
        self.k = k

    def score(self, x):
        return self.k * x

    def price(self, x=1):
        return self.k * x

    @classmethod
    def build(cls, x=1):
        return x


@pytest.fixture
def cached_model(cash_instance):
    Model.score = cash_instance.cache(Model.__dict__["score"])
    yield Model
    Model.score = Model.score.__wrapped__


def test_the_bound_spelling_raises_naming_the_right_one(cached_model):
    m = cached_model(3)
    m.score(2)
    with pytest.raises(TypeError, match=r"Model\.score\.explain\(obj, \.\.\.\)"):
        m.score.explain(2)


def test_the_class_spelling_explains_the_call(cached_model):
    m = cached_model(3)
    m.score(2)
    e = cached_model.score.explain(m, 2)
    assert e.would_hit, e.reason


def test_a_wrong_first_argument_raises_even_when_the_count_fits(cash_instance):
    """``def price(self, x=1)``: ``m.price.explain(2)`` binds, with 2 as self."""
    Model.price = cash_instance.cache(Model.__dict__["price"])
    try:
        with pytest.raises(TypeError, match="not the Model instance"):
            Model(3).price.explain(2)
        assert Model.price.explain(Model(3), 2).reason == "no_entry"
    finally:
        Model.price = Model.price.__wrapped__


def test_a_classmethod_takes_the_class(cash_instance):
    original = Model.__dict__["build"]
    Model.build = classmethod(cash_instance.cache(original.__func__))
    try:
        with pytest.raises(TypeError, match=r"not the Model class.*Model\.build\.__func__\.explain\(Model"):
            Model.build.explain(5)
        assert Model.build.__func__.explain(Model, 5).reason == "no_entry"
    finally:
        Model.build = original


def test_arguments_that_cannot_call_a_function_raise(cash_instance):
    @cash_instance.cache
    def f(x):
        return x

    with pytest.raises(TypeError, match="cannot call it"):
        f.explain(1, 2)
    assert f.explain(1).reason == "no_entry"
