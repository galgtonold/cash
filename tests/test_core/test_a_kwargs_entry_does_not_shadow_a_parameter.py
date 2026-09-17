"""A ``**kwargs`` entry never displaces the parameter of the same name.

Found while attacking the decorator before round 26: canonicalising a call put
named parameters and ``**kwargs`` items into one dict, so ``def request(url, /,
**params)`` called as ``request("/a", url="x")`` lost ``url``'s positional
value -- every such call shared one entry, and ``request("/b", url="x")``
returned ``GET /a``. Same for a method whose caller passes ``self=`` through
``**kwargs``.
"""
from __future__ import annotations

import pytest

from cash import Cash
from cash.backends import InMemoryBackend


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


def test_a_positional_only_parameter_is_not_shadowed(cash):
    @cash.cache
    def request(url, /, **params):
        return f"GET {url} ? {sorted(params.items())}"

    assert request("/a", url="x") == "GET /a ? [('url', 'x')]"
    assert request("/b", url="x") == "GET /b ? [('url', 'x')]"


def test_a_kwargs_self_does_not_shadow_the_receiver(cash):
    class Client:
        def __init__(self, base):
            self.base = base

        @cash.cache
        def fetch(self, /, **kw):
            return f"{self.base} ? {sorted(kw.items())}"

    assert Client("host1").fetch(self="q") == "host1 ? [('self', 'q')]"
    assert Client("host2").fetch(self="q") == "host2 ? [('self', 'q')]"


def test_the_same_call_still_hits(cash):
    ran = []

    @cash.cache
    def request(url, /, **params):
        ran.append(url)
        return f"GET {url} ? {sorted(params.items())}"

    request("/a", url="x")
    request("/a", url="x")
    assert len(ran) == 1


def test_spelling_a_call_two_ways_still_shares_an_entry(cash):
    ran = []

    @cash.cache
    def add(a, b=2, **rest):
        ran.append((a, b))
        return a + b + sum(rest.values())

    assert add(1) == add(1, 2) == add(a=1, b=2) == add(b=2, a=1) == 3
    assert len(ran) == 1
