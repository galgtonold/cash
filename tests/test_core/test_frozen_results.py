"""`@cash.cache(frozen=True)`: "this function's result is not modified afterwards".

Since round 18 a cached result passed on is keyed by its CONTENT, because
nothing keeps the producer's tag current when the object is mutated in place.
That is correct and cheap for small objects and pandas 3 frames, and costs a
full hash per call for everything else -- a model, a large custom object, a
numpy array. `frozen=True` is the explicit way back to the fast path for those:

* the result is keyed downstream by its producer's identity again -- no hash,
  the same in every process, and usable for objects that cannot be pickled;
* where it can be enforced cheaply, it is: a numpy result comes back
  read-only;
* where it cannot, it is audited: the object is re-hashed at an occasional
  use (every use under CASH_DEBUG), and a change warns KEY-FROZEN-MUTATED
  and falls back to content hashing for that object.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import warnings

import pytest

from cash import Cash

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


class Model:
    """A stand-in for a fitted model: state that is expensive to hash."""

    def __init__(self, weights):
        self.weights = list(weights)


def test_a_frozen_result_is_keyed_without_hashing_it(c, monkeypatch):
    @c.cache(frozen=True)
    def train():
        return Model(range(100_000))

    @c.cache
    def score(model, x):
        return sum(model.weights[:x])

    model = train()
    score(model, 10)
    # The key's payload is pickled; a frozen model enters it as its tag, so
    # the bytes pickled stay tiny whatever the model holds.
    import pickle
    sizes = []
    real = pickle.dumps

    def spy(obj, *a, **k):
        data = real(obj, *a, **k)
        sizes.append(len(data))
        return data

    monkeypatch.setattr(pickle, "dumps", spy)
    assert score(model, 10) == score.__wrapped__(model, 10)
    assert sizes and max(sizes) < 4096, f"the frozen model was pickled into the key: {sizes}"


def test_an_unfrozen_result_is_keyed_by_content(c):
    """The control: without the declaration, a mutation reaches the key."""
    @c.cache
    def train():
        return Model([1, 2, 3])

    @c.cache
    def total(model):
        return sum(model.weights)

    m = train()
    assert total(m) == 6
    m.weights.append(4)
    assert total(m) == 10


def test_a_frozen_result_hits_across_processes(tmp_path):
    """Keyed by the producer's identity, which is the same in every process."""
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent('''
        import sys, time
        import cash

        class Model:
            def __init__(self, n):
                self.weights = list(range(n))

        @cash.cache(frozen=True)
        def train(n):
            time.sleep(0.2)  # @cash:assume-safe
            return Model(n)

        @cash.cache
        def score(model):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            time.sleep(0.2)  # @cash:assume-safe
            return sum(model.weights)

        print(score(train(1000)))
    '''), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONDONTWRITEBYTECODE="1")
    runs = [subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           env=env, timeout=120) for _ in range(2)]
    assert [r.stdout.strip() for r in runs] == ["499500", "499500"]
    assert "[RUN]" in runs[0].stderr and "[RUN]" not in runs[1].stderr


def test_an_unpicklable_frozen_result_can_still_be_passed_on(c):
    """Content hashing cannot key an object holding a lock; its producer's
    identity can, which is what made this work before round 18."""
    class Holder:
        def __init__(self):
            self.lock = threading.Lock()
            self.n = 3

    @c.cache(frozen=True)
    def make():
        return Holder()

    calls = []

    @c.cache
    def use(h):
        calls.append(1)
        return h.n * 2

    h = make()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert use(h) == 6
        assert use(h) == 6
    assert len(calls) == 1
    assert not [w for w in rec if "KEY-UNHASHABLE-ARG" in str(w.message)]


def test_a_frozen_numpy_result_comes_back_read_only(c):
    @c.cache(frozen=True)
    def grid(n):
        return np.arange(float(n))

    a = grid(5)
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0] = 99.0
    b = grid(5)                                  # the restored copy too
    assert not b.flags.writeable


def test_a_frozen_numpy_result_is_hashed_once(c, monkeypatch):
    calls = []
    real = Cash._try_hash_numpy

    def counting(value):
        calls.append(1)
        return real(value)

    @c.cache(frozen=True)
    def grid(n):
        return np.arange(float(n))

    @c.cache
    def total(a):
        return float(a.sum())

    a = grid(1000)
    monkeypatch.setattr(Cash, "_try_hash_numpy", staticmethod(counting))
    for _ in range(5):
        total(a)
    assert len(calls) == 1


def test_mutating_a_frozen_result_is_caught_by_the_audit(c, monkeypatch):
    """The declaration can be wrong: `model.fit(...)` downstream on a model from a
    frozen step. Under CASH_DEBUG every use is audited."""
    monkeypatch.setenv("CASH_DEBUG", "1")

    @c.cache(frozen=True)
    def train():
        return Model([1, 2, 3])

    @c.cache
    def total(model):
        return sum(model.weights)

    m = train()
    assert total(m) == 6
    m.weights.append(4)                          # breaks the promise
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = total(m)
    assert got == 10, "after the audit fails, the object is keyed by content"
    found = [w for w in rec if "KEY-FROZEN-MUTATED" in str(w.message)]
    assert found, [str(w.message) for w in rec]
    assert "train" in str(found[0].message)


def test_explain_names_an_argument_keyed_as_frozen(c):
    @c.cache(frozen=True)
    def train():
        return Model([1, 2, 3])

    @c.cache
    def total(model):
        return sum(model.weights)

    m = train()
    total(m)
    explanation = total.explain(m)
    assert explanation.would_hit
    assert "model" in str(explanation.details.get("frozen_args", ""))
