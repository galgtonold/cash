"""Two cached functions that share a module and qualified name each keep the
``key=`` and ``ignore=`` they were decorated with.

One function wrapped twice, the closures a factory makes and two lambdas in a
module all share one name, and cash keeps one record per name: every wrapper
built its key with the ``key=`` of the one decorated LAST. ``exact(2,
factor=5)`` returned 6, the result of ``exact(2, factor=3)``, because the
other wrapper's key function left ``factor`` out.
"""

from __future__ import annotations


def _scale(x, factor=1):
    return x * factor


def test_one_function_wrapped_twice(cash_instance):
    exact = cash_instance.cache(_scale)
    by_x = cash_instance.cache(key=lambda x, factor=1: x)(_scale)

    assert exact(2, factor=3) == 6
    assert exact(2, factor=5) == 10, "the second wrapper's key= keyed the first one"
    assert by_x(2, factor=3) == 6
    assert by_x(2, factor=5) == 6  # its own key= still leaves factor out


def test_the_closures_a_factory_makes(cash_instance):
    def make_pricer(currency):
        if currency == "JPY":

            @cash_instance.cache(key=lambda amount: round(amount))
            def price(amount):
                return f"{round(amount)} {currency}"

        else:

            @cash_instance.cache
            def price(amount):
                return f"{amount:.2f} {currency}"

        return price

    usd = make_pricer("USD")
    jpy = make_pricer("JPY")
    assert usd(1.25) == "1.25 USD"
    assert usd(1.40) == "1.40 USD", "the JPY closure's key= keyed the USD one"
    assert jpy(1.40) == "1 JPY"
    assert jpy(1.25) == "1 JPY" and jpy.cache_info()["hits"] == 1


def test_two_lambdas_in_one_module(cash_instance):
    add = cash_instance.cache(lambda a, b: a + b)
    first_x10 = cash_instance.cache(ignore=["b"])(lambda a, b: a * 10)

    assert add(1, 2) == 3
    assert add(1, 5) == 6, "the other lambda's ignore= keyed this one"
    assert first_x10(1, 2) == 10
    assert first_x10(1, 5) == 10 and first_x10.cache_info()["hits"] == 1
