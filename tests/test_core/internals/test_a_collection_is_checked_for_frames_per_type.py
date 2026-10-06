"""A collection is checked for frames and arrays once per item TYPE.

``g = [f(i) for ...]`` left a list of 200,000 ints, and hashing it asked of
every int whether it was a frame (`_is_bulky`): 0.9 s, 30x the pickle that
then hashed the list. The answer depends on the type alone.
"""

from __future__ import annotations

import numpy as np

from cash import value_hash


def test_a_list_of_ints_asks_once(monkeypatch):
    asked: list[type] = []
    real = value_hash.builtin_hash_family

    def counting(t):
        asked.append(t)
        return real(t)

    monkeypatch.setattr(value_hash, "builtin_hash_family", counting)
    value_hash.compute_hash(list(range(10_000)) + ["x"])
    assert len(asked) <= 3, f"asked {len(asked)} times"


def test_an_array_in_a_list_is_still_hashed_on_its_own():
    a = np.arange(10)
    b = np.arange(10)
    b[5] = 99
    assert value_hash.compute_hash([1, a]) != value_hash.compute_hash([1, b])
    assert value_hash.compute_hash({"k": a}) == value_hash.compute_hash({"k": np.arange(10)})
