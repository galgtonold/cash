"""The pickled-size estimate passes over the plain values of a list.

`pickled_size_estimate` looks into lists, tuples and dicts for the arrays
they hold, and the statement store asks it about every value a statement
binds. It called itself for each item, so ``a = [json.dumps({'i': i}) for i
in range(20_000)]`` paid 20,000 calls that each came to 0 -- 8 ms on a 60 ms
cell (``nb_cheap_calls/json_dumps``), and a list of 50,000 record dicts paid
200,000. A string, number or other primitive holds no array, and nor does a
container at the depth limit: they are passed over without a call.
"""

from __future__ import annotations

import json

import pytest

from cash import sizing

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.core]


@pytest.fixture
def calls(monkeypatch):
    counted = []
    real = sizing.pickled_size_estimate

    def counting(value, *args, **kwargs):
        counted.append(type(value))
        return real(value, *args, **kwargs)

    monkeypatch.setattr(sizing, "pickled_size_estimate", counting)
    return counted


def test_a_list_of_strings_is_one_call(calls):
    values = [json.dumps({"i": i}) for i in range(20_000)]
    assert sizing.pickled_size_estimate(values) == 0
    assert len(calls) == 1


def test_record_dicts_are_a_call_each_and_their_fields_none(calls):
    recs = [{"id": i, "user": "u", "amount": i * 0.5, "tags": ["a"]} for i in range(5_000)]
    assert sizing.pickled_size_estimate(recs) == 0
    assert len(calls) == 1 + len(recs)


def test_arrays_among_plain_values_are_still_counted():
    arr = np.zeros(1_000, dtype=np.float64)
    assert sizing.pickled_size_estimate(["x", 1, arr, None]) == arr.nbytes
    assert sizing.pickled_size_estimate({"a": "x", "b": [1, arr]}) == arr.nbytes
    assert sizing.pickled_size_estimate([{"x": arr, "y": 2.0}, ("s", arr.copy())]) == 2 * arr.nbytes
    # Below the depth limit an array is still read; a container is not looked into.
    assert sizing.pickled_size_estimate([[arr]]) == arr.nbytes
    assert sizing.pickled_size_estimate([[[arr]]]) == 0
