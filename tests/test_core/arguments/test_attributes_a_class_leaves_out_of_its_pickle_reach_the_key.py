"""An attribute a class of yours leaves out of its pickled state is in the key.

Arguments are keyed by what pickle stores for them, and a class's own
``__getstate__`` or ``__reduce__`` decides that. A model that leaves a runtime
setting out of its saved state (``precision``) still reads it, so
``predict(Model(w, precision=5), xs)`` was served the ``precision=2`` result.
The attributes left out are keyed beside the pickled state; one that cannot be
pickled (a lock, which is often why a class leaves it out) is keyed by its type.
"""

from __future__ import annotations

import threading

import pytest

from cash import Cash
from cash.backends import InMemoryBackend


class Model:
    def __init__(self, weights, precision=2):
        self.weights = weights
        self.precision = precision
        self.lock = threading.Lock()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("precision")
        state.pop("lock")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.precision = 2
        self.lock = threading.Lock()


class Reduced:
    def __init__(self, weights, precision=2):
        self.weights = weights
        self.precision = precision

    def __reduce__(self):
        return Reduced, (self.weights,)


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


@pytest.mark.parametrize("cls", [Model, Reduced])
def test_a_setting_left_out_of_the_pickle_keys_apart(cash, cls):
    runs = []

    @cash.cache
    def predict(model, xs):
        runs.append(1)
        return [round(sum(w * x for w in model.weights), model.precision) for x in xs]

    assert predict(cls([0.123456, 0.2]), [1.0, 2.0]) == [0.32, 0.65]
    assert predict(cls([0.123456, 0.2], precision=5), [1.0, 2.0]) == [0.32346, 0.64691]
    assert predict(cls([0.123456, 0.2], precision=5), [1.0, 2.0]) == [0.32346, 0.64691]
    assert len(runs) == 2, "an equal instance did not hit"


def test_a_set_inside_keys_the_setting_too(cash):
    """An object holding a set is keyed through its reduce parts instead."""

    @cash.cache
    def first(model):
        return round(min(model.weights), model.precision)

    assert first(Model({0.123456, 0.2})) == 0.12
    assert first(Model({0.123456, 0.2}, precision=5)) == 0.12346
