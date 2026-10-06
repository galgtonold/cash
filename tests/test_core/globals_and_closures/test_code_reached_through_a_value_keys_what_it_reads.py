"""The module constants a helper reads reach the key when the cached function
reaches that helper through a value.

``apply(partial(price, k=1), x)``, ``STEPS = [lambda x: price(x)]``, a
dispatch dict of lambdas, and a factory's captured list of lambdas: the code
of ``price`` was keyed, the ``RATE`` it reads was not, and editing it served
the old result. The same helper passed bare, or read by name, recomputed.
"""

from __future__ import annotations

import functools
import sys

import pytest

RATE = 2


def price(x, k=1):
    return x * RATE * k


STEPS = [lambda x: price(x)]
DISPATCH = {"price": lambda x: price(x)}


@pytest.fixture
def rate(monkeypatch):
    module = sys.modules[__name__]

    def set_rate(value):
        monkeypatch.setattr(module, "RATE", value)

    return set_rate


def _recomputes(rate, call):
    first = call()
    rate(3)
    assert call() == first // 2 * 3


def test_a_partial_argument(cash_instance, rate):
    @cash_instance.cache
    def apply(f, x):
        return f(x)

    _recomputes(rate, lambda: apply(functools.partial(price, k=1), 10))


def test_a_lambda_in_a_global_list(cash_instance, rate):
    @cash_instance.cache
    def run_steps(x):
        return STEPS[0](x)

    _recomputes(rate, lambda: run_steps(10))


def test_a_lambda_in_a_global_dict(cash_instance, rate):
    @cash_instance.cache
    def dispatch(name, x):
        return DISPATCH[name](x)

    _recomputes(rate, lambda: dispatch("price", 10))


def test_a_lambda_in_a_captured_list(cash_instance, rate):
    def make_runner():
        steps = [lambda x: price(x)]

        @cash_instance.cache
        def run(x):
            return steps[0](x)

        return run

    run = make_runner()
    _recomputes(rate, lambda: run(10))
