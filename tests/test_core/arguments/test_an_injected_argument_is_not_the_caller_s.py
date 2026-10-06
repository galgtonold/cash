"""A ``functools.wraps`` decorator that injects the first argument (a logger,
a DB session, click's ``pass_obj``) does not shift how a call is keyed.

Cash read the signature of the wrapped function through every ``*args,
**kwargs`` wrapper, assuming each passes the call straight on. Under
``f(LOG, *args, **kwargs)`` the caller's ``price(10, 2.0)`` was bound as
``log=10, amount=2.0``, so leaving the injected ``log`` out of the key left
the caller's real amount out: ``price(99, 2.0)`` was served ``20.0``.
"""

from __future__ import annotations

import functools
import logging

import cash

LOG = logging.getLogger("etl")


def with_logger(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(LOG, *args, **kwargs)

    return wrapper


def with_session(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, session=LOG, **kwargs)

    return wrapper


def dropping_first(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args[1:], **kwargs)

    return wrapper


def test_ignoring_an_injected_first_argument(cash_instance):
    @cash_instance.cache
    @with_logger
    def price(log: cash.Ignore[logging.Logger], amount, rate=1.0):
        return amount * rate

    @cash_instance.cache(ignore=["log"])
    @with_logger
    def price2(log, amount, rate=1.0):
        return amount * rate

    for f in (price, price2):
        assert f(10, 2.0) == 20.0
        assert f(99, 2.0) == 198.0, "the caller's first argument was left out of the key"
        assert f(99, rate=2.0) == 198.0 and f.cache_info()["hits"] == 1


def test_a_key_function_gets_the_caller_s_arguments(cash_instance):
    seen = []

    def key(amount, rate=1.0):
        seen.append((amount, rate))
        return amount

    @cash_instance.cache(key=key)
    @with_logger
    def price(log, amount, rate=1.0):
        return amount * rate

    assert price(10, 2.0) == 20.0
    assert seen == [(10, 2.0)]


def test_an_injected_keyword(cash_instance):
    @cash_instance.cache(ignore=["session"])
    @with_session
    def load(table, session=None):
        return table.upper()

    assert load("a") == "A"
    assert load("b") == "B"


def test_a_wrapper_that_reshapes_the_call_keys_every_argument(cash_instance):
    """The control: a wrapper whose source does not show it passing the call
    straight on is not looked through, so every argument stays keyed."""

    @cash_instance.cache
    @dropping_first
    def price(amount, rate=1.0):
        return amount * rate

    assert price("ignored", 10, 2.0) == 20.0
    assert price("ignored", 99, 2.0) == 198.0
