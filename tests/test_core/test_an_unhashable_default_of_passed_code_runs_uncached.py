"""A default of passed code that cannot be keyed means no key.

A function passed to a cached function is keyed by its code and its
defaults. A default that cannot be hashed and whose repr is only an
address (an object holding a lock) cannot be keyed at all: the call runs
uncached with KEY-UNHASHABLE-DEFAULT, as for the cached function's own
default, rather than keying the default by its type name.
"""

from __future__ import annotations

import threading
import warnings

from cash.exceptions import CashCacheIneffectiveWarning


class Holder:
    def __init__(self, rate):
        self.rate = rate
        self.lock = threading.Lock()


class Mask:
    """Unpicklable, with a value-based repr that holds a hex literal."""

    def __init__(self, bits):
        self.bits = bits
        self.lock = threading.Lock()

    def __repr__(self):
        return f"Mask(bits={self.bits:#x})"


HOLDER = Holder(1)


def scorer(x, cfg=HOLDER):
    return x * cfg.rate


def masked(x, m=Mask(0xFF)):
    return x & m.bits


def test_a_default_with_only_an_address_runs_uncached(cash_instance):
    @cash_instance.cache
    def run(fn, x):
        return fn(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert run(scorer, 2) == 2
        HOLDER.rate = 5
        assert run(scorer, 2) == 10, "a default the key cannot see served the old result"
    HOLDER.rate = 1
    codes = [str(w.message) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert any("KEY-UNHASHABLE-DEFAULT" in c and "Holder" in c for c in codes), codes


def test_a_value_based_repr_with_a_hex_literal_still_keys(cash_instance):
    @cash_instance.cache
    def run(fn, x):
        return fn(x)

    assert run(masked, 0x1234) == 0x34
    assert run(masked, 0x1234) == 0x34
    assert run.cache_info()["hits"] == 1
    masked.__defaults__ = (Mask(0x0F),)
    assert run(masked, 0x1234) == 0x4


class Plain:
    def __init__(self, rate):
        self.rate = rate


PLAIN = Plain(1)


def plain_scorer(x, cfg=PLAIN):
    return x * cfg.rate


def test_a_default_changed_in_place_or_rebound_reaches_the_key(cash_instance):
    @cash_instance.cache
    def run(fn, x):
        return fn(x)

    assert run(plain_scorer, 2) == 2
    PLAIN.rate = 5
    assert run(plain_scorer, 2) == 10
    plain_scorer.__defaults__ = (Plain(7),)
    assert run(plain_scorer, 2) == 14
