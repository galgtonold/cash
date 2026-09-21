"""A call result too big to be worth its bytes is not pickled to be digested.

Measured before round 29 on r28s5's own cells (pandas 2.3): after its 2.8 s
of compute, ``net_returns(orders, 12)`` spent 2.7 s pickling its 1.7 GiB result
for a digest -- a digest that lets a statement store a reference to the call's
entry instead of a copy. The statement was then refused for disk, as not worth
its bytes, on the size that digest reported. Numeric columns pickle to at
least their ``nbytes``, so a bound read off the arrays gives the same answer
here without the pickle.
"""
import numpy as np
import pandas as pd

from cash._sizing import pickled_lower_bound
from cash.notebook import call_unit as call_unit_mod
from cash.notebook.call_refs import DIGEST_FIELD


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


def test_a_result_over_the_ceiling_is_not_digested(call_unit_harness, monkeypatch):
    calls = _spy(monkeypatch)
    unit = call_unit_harness(lineage={}, user_ns={})
    big = pd.DataFrame({"x": np.zeros(4_000_000), "s": ["a"] * 4_000_000})   # >= 36 MB
    unit._store("call:big", big, 0.2)                    # 0.2 s is worth 25.6 MiB at most
    assert calls == [], "pickled a result the size bound already refuses"
    assert DIGEST_FIELD not in _meta(unit, "call:big")


def test_a_result_worth_its_bytes_is_still_digested(call_unit_harness, monkeypatch):
    calls = _spy(monkeypatch)
    unit = call_unit_harness(lineage={}, user_ns={})
    small = pd.DataFrame({"x": np.zeros(1000)})
    unit._store("call:small", small, 0.2)
    assert calls == [1]
    assert DIGEST_FIELD in _meta(unit, "call:small")


def test_the_bound_is_below_the_pickled_size():
    import pickle
    frame = pd.DataFrame({"x": np.arange(10_000, dtype=float), "s": ["ab"] * 10_000,
                          "c": pd.Categorical(["u", "v"] * 5_000)})
    for value in (frame, frame["x"], (frame, frame["s"], None), {"a": frame}, np.ones(500)):
        assert 0 < pickled_lower_bound(value) <= len(pickle.dumps(value, protocol=5)), type(value)
    assert pickled_lower_bound(object()) == 0
