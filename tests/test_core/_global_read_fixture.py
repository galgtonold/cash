"""Fixture module for mutable-global-read detection tests.

Lives as a real importable module (not collected by pytest) so the analyzer can
read its source and scan it for module-global mutations.
"""
import cash

c = cash.Cash(backend=cash.InMemoryBackend())

CONFIG = {"rate": 0.10}          # mutated below -> footgun
TABLE = {"a": 1, "b": 2}         # never written -> constant
LIMIT = 100                      # constant int
COUNTER = 0                      # reassigned via `global` -> footgun

REGISTRY = {}                    # populated at import time (top-level) only...
for _name in ("alpha", "beta"):
    REGISTRY[_name] = len(_name)  # ...so it is constant during runtime -> no flag


def set_rate(r):
    CONFIG["rate"] = r           # in-place mutation of a global


def bump():
    global COUNTER
    COUNTER += 1                 # reassignment of a global


def unrelated_local():
    TABLE = []                   # LOCAL shadowing a global name...
    TABLE.append(1)              # ...mutating the local, not the global
    return TABLE


@c.cache
def price(amount):
    return amount * (1 + CONFIG["rate"])     # reads mutated CONFIG -> flag


@c.cache
def lookup(k):
    return TABLE[k] * LIMIT                   # reads constants -> no flag


@c.cache
def from_registry(k):
    return REGISTRY[k]                        # reads import-time-only REGISTRY -> no flag


@c.cache
def with_counter(x):
    return x + COUNTER                        # reads reassigned COUNTER -> flag


@c.cache(strict=True)
def strict_price(amount):
    return amount * (1 + CONFIG["rate"])      # strict -> raises
