"""numpy ``datetime64`` and ``timedelta64`` scalars key as their exact type,
their unit and their 8 bytes: the same number in two units keys apart, a
changed value keys apart, and a long list of them keys at C speed (a list
of 868,000 from ``list(frame['t'].diff())`` was pickled item by item)."""

from __future__ import annotations

import numpy as np
import pytest

from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT


def _key(value):
    return canonical_bytes(value, BUILTIN_CONTENT)


@pytest.mark.parametrize("kind", ["timedelta64", "datetime64"])
def test_a_unit_or_a_value_changed_keys_apart(kind):
    make = getattr(np, kind)
    assert _key(make(3, "s")) == _key(make(3, "s"))
    assert _key(make(3, "s")) != _key(make(3, "ms"))
    assert _key(make(3, "s")) != _key(make(4, "s"))
    assert _key(make("NaT", "s")) != _key(make("NaT", "ns"))


def test_a_long_list_keys_its_items():
    values = list(np.arange(1_000).astype("timedelta64[us]"))
    changed = list(values)
    changed[500] = np.timedelta64(7, "us")
    in_ms = [value.astype("timedelta64[ms]") for value in values]
    assert _key(values) == _key(list(np.arange(1_000).astype("timedelta64[us]")))
    assert len({_key(values), _key(changed), _key(in_ms), _key(values[:999])}) == 4


def test_a_time_does_not_key_as_the_integer_of_its_bytes():
    assert _key(np.timedelta64(3, "ns")) != _key(np.int64(3))
    assert _key([np.timedelta64(3, "ns")] * 300) != _key([np.int64(3)] * 300)
