"""A key pickle's ``persistent_id`` (`kept_state.key_aliases`) at C speed
(``cwalk.c``'s ``key_alias_id``) writes the bytes the Python one writes.

The pickle asks it of every object a key holds: keying a column of seven
lists of 140,000 timestamps took 3.6 s, most of it in the Python hook. The
same marks come out, so every key is the same.
"""

from __future__ import annotations

import datetime
import types

import numpy as np
import pandas as pd
import pytest

from cash import _native, _plain_data, kept_state
from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT

pytestmark = pytest.mark.skipif(
    getattr(_native.kernels, "key_alias_id", None) is None, reason=f"no C key hook: {_native.REASON}"
)


class Rec:
    def __init__(self, x):
        self.x = x


def _values():
    shared = [1, 2]
    rec = Rec(shared)
    loop = Rec(None)
    loop.x = loop
    stamps = list(pd.date_range("2020-01-01", periods=300, freq="min"))
    zoned = list(pd.date_range("2020-01-01", periods=50, freq="h", tz="Europe/Berlin"))
    yield "records", [Rec(i) for i in range(50)]
    yield "one record twice", [rec, rec, shared]
    yield "a loop", [loop]
    yield "timestamps", [stamps, stamps[:3]]
    yield "zoned timestamps", zoned
    yield "one timestamp twice", [stamps[0], stamps[0]]
    yield "numpy in objects", [Rec(np.float64(1.5)), Rec(np.dtype("int8")), Rec(np.arange(3))]
    yield "dates", [Rec(datetime.date(2020, 1, 2)), datetime.timedelta(3)]
    yield "functions and types", [Rec(len), Rec(Rec), Rec(sorted)]
    yield "a frame inside", [Rec(pd.DataFrame({"a": [1, 2]}))]


def _python_hook(monkeypatch):
    monkeypatch.setattr(kept_state, "_native", types.SimpleNamespace(kernels=None))


@pytest.mark.parametrize(("name", "value"), list(_values()), ids=[n for n, _ in _values()])
def test_a_key_is_the_same_bytes_with_either_hook(monkeypatch, name, value):
    fast = canonical_bytes(value, BUILTIN_CONTENT)
    with monkeypatch.context() as m:
        _python_hook(m)
        slow = canonical_bytes(value, BUILTIN_CONTENT)
    assert fast == slow


def test_a_container_held_outside_is_marked_by_either_hook(monkeypatch):
    held = [1, 2, 3]
    value = [Rec(held), Rec(held)]
    outside = {id(held): ("arg", 0)}
    fast = _plain_data.key_dumps(value, outside)
    with monkeypatch.context() as m:
        _python_hook(m)
        slow = _plain_data.key_dumps(value, outside)
    assert fast == slow
    assert b"__cash_held__" in fast
