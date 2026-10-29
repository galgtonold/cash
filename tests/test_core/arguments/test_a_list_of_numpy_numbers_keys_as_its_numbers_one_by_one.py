"""A list of numpy numbers of one type (``list(series)`` of an ``int64``
column) is keyed from one array of them, to the bytes the walk gives one
number at a time: 1.1 us an item before, 4.6 s a key for seven lists of
140,000 numbers."""

from __future__ import annotations

import numpy as np
import pytest

from cash import canonical_form
from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT, _object_items_bytes

CASES = {
    "int64": list(np.arange(500)),
    "float64": list(np.random.default_rng(0).random(500)),
    "float32 tuple": tuple(np.random.default_rng(1).random(500).astype(np.float32)),
    "int8": list(np.arange(300, dtype=np.int8)),
    "bool": list(np.array([True, False] * 200)),
    "complex": list(np.arange(200) * 1j),
    "float16": list(np.random.default_rng(2).random(200).astype(np.float16)),
    "two types": list(np.arange(200)) + [np.float64(1.0)],
    "with a python int": list(np.arange(200)) + [3],
}


@pytest.mark.parametrize("value", CASES.values(), ids=CASES.keys())
def test_the_key_is_the_one_read_number_by_number(monkeypatch, value):
    fast = (canonical_bytes([value, value[:3]], BUILTIN_CONTENT), _object_items_bytes([value]))
    monkeypatch.setattr(canonical_form, "_numbers_form", lambda items: None)
    assert fast == (canonical_bytes([value, value[:3]], BUILTIN_CONTENT), _object_items_bytes([value]))


def test_one_list_held_twice_is_still_marked():
    row = list(np.arange(200))
    assert canonical_bytes([row, row], BUILTIN_CONTENT) != canonical_bytes([row, list(row)], BUILTIN_CONTENT)
