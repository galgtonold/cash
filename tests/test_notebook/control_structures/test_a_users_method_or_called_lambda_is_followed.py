"""A method of the user's class, and a ``lambda`` a helper calls at once,
are followed for what they hand to code whose effect is unknown.

``Mixer().go(SEED)`` with ``def go(self, seed):
random.Random(seed).shuffle(DECK)``, and ``def mix(seed): (lambda:
random.Random(seed).shuffle(DECK))()``, change ``DECK``: a statement or a
loop calling either watches it. A library object's method is not followed,
and costs no source read.
"""

from __future__ import annotations

import ast
import random
from collections import OrderedDict

from cash.analysis.mutation_effects import live_function_source, loop_watch_candidates
from cash.analysis.namespace_effects import call_arguments, users_method_call


class Mixer:
    def go(self, seed):
        random.Random(seed).shuffle(DECK)

    def total(self, seed):
        return sum(DECK) + seed


class Holder:
    def __init__(self, items):
        self.items = items

    def go(self, seed):
        random.Random(seed).shuffle(self.items)


def mix(seed):
    (lambda: random.Random(seed).shuffle(DECK))()


def through_method(seed):
    Mixer().go(seed)


DECK = [1, 2, 3]
NAMESPACE = {
    "random": random,
    "Mixer": Mixer,
    "Holder": Holder,
    "mix": mix,
    "through_method": through_method,
    "m": Mixer(),
    "h": Holder([4, 5]),
    "od": OrderedDict(),
    "DECK": DECK,
    "SEED": 1,
}


def _source(name: str) -> str | None:
    return live_function_source(name, NAMESPACE)


def _statement(code: str) -> frozenset[str]:
    return call_arguments(ast.parse(code), dict(NAMESPACE), _source)


def _loop(code: str) -> frozenset[str]:
    return loop_watch_candidates(ast.parse(code).body[0], NAMESPACE, lambda name: name == "random", _source)


def test_a_method_that_shuffles_a_global_watches_it():
    assert "DECK" in _statement("Mixer().go(SEED)")
    assert "DECK" in _statement("m.go(SEED)")
    assert "DECK" in _statement("n = m.go(SEED)")
    assert "DECK" in _loop("for _ in range(2):\n    Mixer().go(SEED)")


def test_a_method_changing_what_self_holds_watches_the_receiver():
    assert "h" in _statement("h.go(SEED)")


def test_a_helper_calling_such_a_method_watches_the_global():
    assert "DECK" in _statement("through_method(SEED)")


def test_a_lambda_called_at_once_is_followed():
    assert "DECK" in _statement("mix(SEED)")
    assert "DECK" in _loop("for _ in range(2):\n    mix(SEED)")


def test_a_method_that_only_reads_watches_nothing():
    assert "DECK" not in _statement("s = Mixer().total(SEED)")


def test_only_a_users_class_is_followed():
    call = ast.parse("od.move_to_end(1)").body[0].value
    assert users_method_call(call, NAMESPACE) is None
    call = ast.parse("random.shuffle(DECK)").body[0].value
    assert users_method_call(call, NAMESPACE) is None
    name, as_called = users_method_call(ast.parse("m.go(SEED)").body[0].value, NAMESPACE)
    assert name == "m.go"
    assert [ast.unparse(a) for a in as_called.args] == ["m", "SEED"]
