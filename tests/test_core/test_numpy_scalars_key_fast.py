"""A list of numpy numbers keys at C speed, and by exact value and type.

``list(arr)`` gives numpy scalars. They were not leaves of plain data, so each
was walked for sets (an MRO walk), walked again for its canonical form and
pickled on its own: about 7 us apiece, and a hit on 200k of them took 1.4 s
where the body took 7 ms. They are now leaves, keyed one array per type.
"""

from __future__ import annotations

import pytest

from cash import Cash, object_hashing
from cash.decorator import arg_hashing

np = pytest.importorskip("numpy")


@pytest.fixture
def key(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    return lambda *args: c._args.hash_payload(args, {})


def test_a_list_of_numpy_numbers_takes_the_fast_path(key, monkeypatch):
    values = [np.float64(i) for i in range(1000)]
    assert arg_hashing.plain_census(values)[0] == "plain_numpy"
    walked = []
    real = object_hashing.contains_set
    monkeypatch.setattr(object_hashing, "contains_set", lambda *a, **k: walked.append(1) or real(*a, **k))
    key(values)
    assert walked == []


def test_numpy_numbers_in_a_dict_are_not_walked(key, monkeypatch):
    calls = []
    real = object_hashing.object_state
    monkeypatch.setattr(object_hashing, "object_state", lambda v: calls.append(1) or real(v))
    key({i: np.float64(i) for i in range(100)})
    assert calls == []


PAIRS = {
    "value": ([np.float64(1.0)], [np.float64(2.0)]),
    "numpy type": ([np.float64(1.0)], [np.float32(1.0)]),
    "int width": ([np.int64(1)], [np.int32(1)]),
    "numpy against python": ([np.float64(1.0)], [1.0]),
    "order of mixed types": ([np.int64(1), 1], [1, np.int64(1)]),
    "nesting": ([[np.int64(1), np.int64(2)], [np.int64(3)]], [[np.int64(1)], [np.int64(2), np.int64(3)]]),
    "list or tuple": ([(np.int64(1),)], [[np.int64(1)]]),
    "negative zero": ([np.float64(0.0)], [np.float64(-0.0)]),
    "in a dict": ({"a": np.float64(1.0)}, {"a": np.float32(1.0)}),
    "bool": ([np.bool_(True)], [np.int64(1)]),
}


@pytest.mark.parametrize("left, right", list(PAIRS.values()), ids=list(PAIRS))
def test_different_numbers_key_apart(key, left, right):
    assert key(left) != key(right)


def test_equal_numbers_key_alike(key):
    """Positive control: the same numbers built twice, NaN included."""
    build = lambda: [np.float64(i / 3) for i in range(50)] + [np.float64("nan"), np.int8(3), "tag"]
    assert key(build()) == key(build())
    assert key([[np.int64(1)], (np.float32(2),)]) == key([[np.int64(1)], (np.float32(2),)])
    assert key({"a": np.float64(1.0)}) == key({"a": np.float64(1.0)})


def test_a_hit_serves_the_right_value(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)

    @c.cache
    def total(xs):
        return float(sum(xs))

    assert total([np.float64(i) for i in range(10)]) == 45.0
    assert total([np.float64(i) for i in range(10)]) == 45.0
    assert total([np.float64(i) for i in range(11)]) == 55.0
    assert total.cache_info()["hits"] == 1
