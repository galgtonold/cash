"""``explain()`` predicts the key the next call actually looks up.

It used to rebuild the key by hand, and the copy lacked two of the steps a real
call takes: the random-seed epoch of a function seen drawing random numbers,
and the class members a method reaches through ``self``. For those functions
``explain()`` reported ``no_entry`` for a call that would hit. It also resolved
``dynamic_depends_on`` through a separate "silent" copy that ignored a return
value the real call refuses.
"""

from __future__ import annotations

import random
import warnings

from cash import Cash, CashCacheIneffectiveWarning
from cash.backends.memory_backend import InMemoryBackend


class Model:
    RATE = 2

    def run(self, x):
        return x * self.rate()

    def rate(self):
        return self.RATE


def _cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


def test_a_function_that_draws_random_numbers(monkeypatch):
    from cash.notebook import randomness

    # What a seeded session records (the notebook's seed tracking fills it).
    monkeypatch.setattr(randomness, "_ACTIVE_SEED_EPOCHS", {"random": "epoch-1"})
    c = _cash()

    @c.cache(allow_random=True)
    def draw(n):
        return random.random() + n

    random.seed(1)
    draw(1)  # learns that it draws; not stored
    draw(1)  # stored under the seed epoch
    explanation = draw.explain(1)
    assert explanation.would_hit, explanation


def test_a_method_that_reaches_class_members():
    c = _cash()
    run = c.cache(Model.run)
    m = Model()
    run(m, 3)
    explanation = run.explain(m, 3)
    assert explanation.would_hit, explanation


def test_a_resolver_the_call_refuses_is_uncomputable_and_silent():
    c = _cash()

    @c.cache(dynamic_depends_on=lambda: "not a DataSource")
    def f():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = f.explain()
    assert explanation.reason == "key_uncomputable", explanation
    assert not [w for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    # Held back while explaining, so the real call still warns.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f()
    assert [w for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
