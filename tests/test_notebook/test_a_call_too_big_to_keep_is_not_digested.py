"""A call result too big to be worth its bytes is not pickled to be digested.

Measured before round 29 on r28s5's own cells (pandas 2.3): after its 2.8 s
of compute, ``net_returns(orders, 12)`` spent 2.7 s pickling its 1.7 GiB
result for a digest -- the digest that lets a statement store a reference to
the call's entry instead of a copy -- and the statement was then refused for
disk, as not worth its bytes, on the size that digest reported. An estimate
read off the arrays gives that answer without the pickle.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from cash._sizing import pickled_size_estimate
from cash.notebook import call_unit as call_unit_mod
from cash.notebook.call_refs import DIGEST_FIELD, ESTIMATED_FIELD, UNHASHED_PREFIX


def _spy(monkeypatch):
    calls = []
    real = call_unit_mod.digest_and_size

    def spy(value):
        calls.append(1)
        return real(value)

    monkeypatch.setattr(call_unit_mod, "digest_and_size", spy)
    return calls


def _meta(unit, key):
    return unit._cash.backend.get_metadata(key) or {}


def test_a_result_far_over_the_ceiling_is_not_digested(call_unit_harness, monkeypatch):
    calls = _spy(monkeypatch)
    unit = call_unit_harness(lineage={}, user_ns={})
    big = pd.DataFrame({"x": np.zeros(4_000_000), "s": ["a"] * 4_000_000})  # ~40 MB
    unit._store("call:big", big, 0.1)  # 0.1 s is worth 12.8 MiB at most
    assert calls == [], "pickled a result the size estimate already refuses"
    meta = _meta(unit, "call:big")
    assert meta.get(ESTIMATED_FIELD) is True
    assert str(meta.get(DIGEST_FIELD)).startswith(UNHASHED_PREFIX)


def test_a_result_worth_its_bytes_is_still_digested(call_unit_harness, monkeypatch):
    calls = _spy(monkeypatch)
    unit = call_unit_harness(lineage={}, user_ns={})
    unit._store("call:small", pd.DataFrame({"x": np.zeros(1000)}), 0.2)
    assert calls == [1]
    meta = _meta(unit, "call:small")
    assert DIGEST_FIELD in meta and ESTIMATED_FIELD not in meta


def _shapes():
    n = 50_000
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": np.arange(n, dtype=float), "s": [f"order-{i}" for i in range(n)]})
    return {
        "numbers": pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n)}),
        "distinct strings": pd.DataFrame({"s": pd.array([f"o{i:08d}" for i in range(n)], dtype=object)}),
        "one string repeated": pd.DataFrame({"s": pd.array(["EU"] * n, dtype=object)}),
        "a few strings": pd.DataFrame({"s": pd.Series(rng.choice(["EU", "NA", "APAC"], n)).astype(object)}),
        "default strings": frame,
        "categorical": pd.DataFrame({"c": pd.Categorical(rng.choice(["a", "b"], n))}),
        "a tuple sharing a column": (frame, frame["s"], None),
        "an array": np.ones(n),
    }


@pytest.mark.parametrize("name", list(_shapes()))
def test_the_estimate_is_close_and_never_far_over(name):
    value = _shapes()[name]
    real = len(pickle.dumps(value, protocol=5))
    ratio = pickled_size_estimate(value) / real
    # Over is the side that costs: a digest skipped for a value worth keeping.
    assert 0.5 <= ratio <= 1.2, f"{name}: estimate/real = {ratio:.2f}"


def test_anything_else_is_not_estimated():
    assert pickled_size_estimate(object()) == 0
    assert pickled_size_estimate({"a": [1, 2]}) == 0


def test_estimating_does_not_pickle_a_column(monkeypatch):
    """pandas 2 hands out 2-D blocks; sampling one as a single item pickled
    the whole column -- the cost the estimate exists to avoid."""
    import cash._sizing as sizing

    frame = pd.DataFrame(
        {
            "s": pd.array([f"o{i}" for i in range(10_000)], dtype=object),
            "t": pd.array([f"p{i}" for i in range(10_000)], dtype=object),
        }
    )
    biggest = []
    real = sizing.pickle.dumps

    def spy(obj, *a, **k):
        out = real(obj, *a, **k)
        biggest.append(len(out))
        return out

    monkeypatch.setattr(sizing.pickle, "dumps", spy)
    pickled_size_estimate(frame)
    assert max(biggest, default=0) < 1000, "an estimate pickled a whole column"
