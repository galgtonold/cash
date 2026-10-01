"""Equal immutable values key alike, whether they are one object or two.

A key pickled its canonical form through pickle's memo, which writes a
second reference to one object as a back-reference. So a config dict whose
two dates were one string literal keyed apart from the equal dict parsed
from JSON, and ``f(s, s)`` apart from ``f(s, t)`` with ``s == t``: a miss and
a second entry for one value. Sharing that matters -- a list, dict or array
held twice -- is still keyed (test_a_container_two_arguments_share_keys_apart).
"""

from __future__ import annotations

import datetime
import decimal
import json

import pytest

from cash import Cash, FileBackend


@pytest.fixture
def key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    return lambda *args, **kwargs: c._args.hash_payload(args, kwargs)


def _two(make):
    return make(), make()


@pytest.mark.parametrize(
    "make",
    [
        lambda: "2020-" + str(1).zfill(2),
        lambda: b"x" * 3 + bytes([1]),
        lambda: datetime.date(2020, 1, 1),
        lambda: datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc),
        lambda: decimal.Decimal("1.50"),
        lambda: (1, "a" + str(2)),
    ],
    ids=["str", "bytes", "date", "datetime", "Decimal", "tuple"],
)
def test_one_object_or_two_keys_alike(key, make):
    one = make()
    first, second = _two(make)
    assert first is not second
    assert key(one, one) == key(first, second)
    assert key({"a": one, "b": one}) == key({"a": first, "b": second})
    assert key([one, one, {1}]) == key([first, second, {1}])


def test_a_json_config_hits_the_literal_one(key):
    literal = {"start": "2020-01-01", "end": "2020-01-01"}
    parsed = json.loads('{"start": "2020-01-01", "end": "2020-01-01"}')
    assert literal["start"] is literal["end"]
    assert parsed["start"] is not parsed["end"]
    assert key(literal) == key(parsed)


def test_equal_decimals_of_different_precision_still_key_apart(key):
    assert key(decimal.Decimal("1.0")) != key(decimal.Decimal("1.00"))
