"""Settings a class keeps out of its pickle reach the key however it does so.

Beside a ``__getstate__`` that drops an attribute from ``__dict__``, a class
can keep a setting in ``__slots__``, save the attribute's name with a portable
default in place of its live value, or register its reducer with
``copyreg.pickle``. Each one used to key two settings alike and serve one
setting's result for the other.
"""

from __future__ import annotations

import copyreg

import pytest

from cash import Cash
from cash.backends import InMemoryBackend


class Slotted:
    __slots__ = ("w", "precision")

    def __init__(self, w, precision):
        self.w, self.precision = w, precision

    def __getstate__(self):
        return {"w": self.w}

    def __setstate__(self, state):
        self.w = state["w"]
        self.precision = 2


class Portable:
    def __init__(self, w, scale):
        self.w, self.scale = w, scale

    def __getstate__(self):
        state = self.__dict__.copy()
        state["scale"] = 1  # saved with a portable default
        return state


class Registered:
    def __init__(self, w, mode):
        self.w, self.mode = w, mode


def _rebuild_registered(w):
    return Registered(w, "fast")


copyreg.pickle(Registered, lambda o: (_rebuild_registered, (o.w,)))


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


def _runs_twice_then_hits(cash, body, make, a, b):
    runs = []

    @cash.cache
    def f(m):
        runs.append(1)
        return body(m)

    assert f(make(a)) == body(make(a))
    assert f(make(b)) == body(make(b)), "the other setting was served the first one's result"
    assert f(make(b)) == body(make(b))
    assert len(runs) == 2, "an equal instance did not hit"


def test_a_setting_kept_in_slots_and_left_out_keys_apart(cash):
    _runs_twice_then_hits(cash, lambda m: round(m.w / 3, m.precision), lambda p: Slotted(1.0, p), 2, 5)


def test_a_setting_saved_under_its_name_with_another_value_keys_apart(cash):
    _runs_twice_then_hits(cash, lambda m: m.w * m.scale, lambda s: Portable(2, s), 1, 10)


def test_a_setting_a_copyreg_reducer_leaves_out_keys_apart(cash):
    _runs_twice_then_hits(cash, lambda m: (m.w, m.mode), lambda mode: Registered(1, mode), "fast", "exact")
