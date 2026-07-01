"""Hidden mutation through a custom in-place operator dunder must reset on
isolated re-run (CAS-78). An augmented assignment ``obj <op>= x`` on a custom
instance dispatches to ``__iadd__`` / ``__isub__`` / ``__imul__`` (or falls back
to ``__add__`` etc.) whose body mutates hidden state; on an isolated re-run it
accumulates.

Each operator runs ONCE, so the correct value is identical on the first run and
every re-run.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


# --- must reset ---------------------------------------------------------------

def test_iadd_free_var(nb_runner):
    _rerun(nb_runner,
           "log = []\nclass Box:\n    def __iadd__(self, x):\n        log.append(x)\n        return self\nb = Box()",
           "b += 7\nprint('log', len(log))", "log 1")


def test_isub_free_var(nb_runner):
    _rerun(nb_runner,
           "events = []\nclass Meter:\n    def __isub__(self, x):\n        events.append(x)\n        return self\nm = Meter()",
           "m -= 3\nprint('events', len(events))", "events 1")


def test_imul_class_var(nb_runner):
    _rerun(nb_runner,
           "class Scaler:\n    seen = []\n    def __imul__(self, x):\n        Scaler.seen.append(x)\n        return self\ns = Scaler()",
           "s *= 2\nprint('seen', len(Scaler.seen))", "seen 1")


def test_add_fallback_free_var(nb_runner):
    # Vec defines __add__ (no __iadd__): v += 4 becomes v = v.__add__(4), which
    # reassigns v (idempotent) but the free-var side effect must still reset.
    _rerun(nb_runner,
           "calls = []\nclass Vec:\n    def __init__(self, n):\n        self.n = n\n    def __add__(self, x):\n        calls.append(x)\n        return Vec(self.n + x)\nv = Vec(0)",
           "v += 4\nprint('calls', len(calls))", "calls 1")


def test_iadd_mutates_self(nb_runner):
    # __iadd__ mutates self in place (returns self) — reset the receiver.
    _rerun(nb_runner,
           "class Bag:\n    def __init__(self):\n        self.items = []\n    def __iadd__(self, x):\n        self.items.append(x)\n        return self\nbag = Bag()",
           "bag += 'a'\nprint('items', len(bag.items))", "items 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_int_augassign_not_over_invalidated(nb_runner):
    # the COMMON case: plain int += must not be treated as a hidden mutation.
    _rerun(nb_runner,
           "base = 10",
           "x = base\nx += 5\nprint('x', x)", "x 15")


def test_pure_iadd_not_over_invalidated(nb_runner):
    # __iadd__ that returns a fresh value with no side effect.
    _rerun(nb_runner,
           "class Money:\n    def __init__(self, c):\n        self.c = c\n    def __iadd__(self, x):\n        return Money(self.c + x)\nm = Money(100)",
           "m += 50\nprint('c', m.c)", "c 150")
