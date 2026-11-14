"""A library that decorates a new callable object on every call (nilearn's
``cache(_ExtractionFunctor(...), ...)(imgs)``) moves nothing a remembered
state segment or fast hit reads.

Each new instance is another object to `Cash._decorated_before`, so it was
registered anew, which moved the state memo's generation and the cash
epoch, and setting the new wrapper's defaults moved the world epoch: every
call threw away what every cached function had remembered. Another
instance of the same class, with the same options and code, now takes the
registration's place quietly. Each instance is still keyed by what it
holds, and another class, other options or a function decorated again
still register as before.
"""

from __future__ import annotations

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


class Adder:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        RUNS["n"] += 1
        return x + self.k


class Scaler:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return x * self.k


def double(x):
    return 2 * x


def _epochs():
    return state_memo.GEN[0], hit_memo.world_epochs()


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def test_a_new_instance_each_call_moves_no_epoch(c):
    for _ in range(3):
        assert c.cache(Adder(1))(5) == 6
    before = _epochs()
    for _ in range(5):
        assert c.cache(Adder(1))(5) == 6
    assert _epochs() == before


def test_each_instance_is_still_keyed_by_what_it_holds(c):
    assert c.cache(Adder(1))(5) == 6
    runs = RUNS["n"]
    assert c.cache(Adder(2))(5) == 7
    assert RUNS["n"] == runs + 1  # not served from the first instance's entry
    assert c.cache(Adder(1))(5) == 6
    assert c.cache(Adder(2))(5) == 7


def test_another_class_or_other_options_still_register(c):
    c.cache(Adder(1))(5)
    c.cache(Adder(1))(5)
    gen = state_memo.GEN[0]
    assert c.cache(Scaler(3))(5) == 15
    assert state_memo.GEN[0] != gen
    gen = state_memo.GEN[0]
    assert c.cache(Adder(1), ttl=60)(5) == 6
    assert state_memo.GEN[0] != gen


def test_a_function_is_never_swapped_quietly(c):
    c.cache(double)(1)
    gen = state_memo.GEN[0]
    assert c._other_instance_before(double, ([],) + (None,) * 12) is None
    assert c.cache(double, ttl=5)(1) == 2
    assert state_memo.GEN[0] != gen


def test_an_instance_another_cached_function_depends_on_registers(c):
    c.cache(Adder(1))(5)
    (name,) = [n for n in c._registry.cached if n.endswith("Adder[instance]")]
    c._registry.graph.add_dependency("someone.else", name)
    gen = state_memo.GEN[0]
    assert c.cache(Adder(1))(5) == 6
    assert state_memo.GEN[0] != gen
