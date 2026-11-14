"""A call with a numpy number argument is served from the memo too.

Iterating an array gives numpy scalars: ``for n in np.array([512, 1024]):
get_window("hann", n)`` passes ``np.int64``. The memo key took only Python
atoms, so each such hit built its key the ordinary way, ~160 us where a hit
on ``3`` costs well under a microsecond. A fixed-size numpy number is now
keyed by its exact type and the bytes of its value: exactly as fine as the
ordinary key, which keeps ``np.int64(3)``, ``np.int32(3)``, ``3`` and
``3.0`` apart, ``-0.0`` from ``0.0`` and one NaN from another.
"""

from __future__ import annotations

import struct
import sys

import pytest

from cash import Cash
from cash.decorator import hit_memo

np = pytest.importorskip("numpy")

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


def described(x):
    RUNS["n"] += 1
    return f"{type(x).__name__}:{x!r}"


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _key(*args):
    hit_memo.numpy_scalars_keyed()
    return hit_memo._cw.any_key(args, {})


def _nan(bits: int):
    return np.frombuffer(struct.pack("<Q", bits), dtype=np.float64)[0]


def _served_fast(f, *args):
    for _ in range(4):
        result = f(*args)
    before, runs = hit_memo.status()["fast_hits"], RUNS["n"]
    assert f(*args) == result
    assert RUNS["n"] == runs, "not served"
    assert hit_memo.status()["fast_hits"] > before, "the memo never served it"
    return result


@pytest.mark.parametrize("value", [np.int64(3), np.float64(2.5), np.int32(7), np.bool_(True), np.complex128(1j)])
def test_a_numpy_number_argument_is_served_from_the_memo(c, value):
    assert _served_fast(c.cache(described), value) == described(value)


def test_numpy_numbers_from_iterating_an_array_are_served(c):
    f = c.cache(described)
    sizes = np.array([512, 1024, 2048])
    expected = [described(n) for n in sizes]
    for _ in range(4):
        assert [f(n) for n in sizes] == expected
    before, runs = hit_memo.status()["fast_hits"], RUNS["n"]
    assert [f(n) for n in sizes] == expected
    assert RUNS["n"] == runs
    assert hit_memo.status()["fast_hits"] - before == len(sizes)


def test_each_kind_of_three_is_served_its_own_result(c):
    """Warmed together, then each served fast: a coarser key would hand one
    of them another's result."""
    f = c.cache(described)
    values = [
        np.int64(3),
        np.int32(3),
        np.longlong(3),
        np.uint8(3),
        3,
        3.0,
        np.float64(3),
        np.float32(3),
        True,
        np.bool_(True),
    ]
    for _ in range(4):
        for v in values:
            f(v)
    before, runs = hit_memo.status()["fast_hits"], RUNS["n"]
    assert [f(v) for v in values] == [described(v) for v in values]
    assert RUNS["n"] == runs + len(values)  # the expected values above, not f
    assert hit_memo.status()["fast_hits"] - before == len(values)


def test_signed_zeros_and_nans_are_served_their_own_results(c):
    f = c.cache(lambda x: struct.pack("<d", float(x)))
    values = [np.float64(0.0), np.float64(-0.0), _nan(0x7FF8000000000000), _nan(0x7FF8000000000001)]
    for _ in range(4):
        got = [f(v) for v in values]
    assert got == [struct.pack("<d", float(v)) for v in values]
    assert len(set(got)) == len(values)
    before = hit_memo.status()["fast_hits"]
    assert [f(v) for v in values] == got
    assert hit_memo.status()["fast_hits"] - before == len(values)


def test_the_memo_key_is_the_exact_type_and_the_bytes():
    values = [
        np.int64(3),
        np.int32(3),
        np.longlong(3),
        3,
        3.0,
        np.float64(3),
        np.float32(3),
        np.float16(3),
        True,
        np.bool_(True),
        np.complex128(3),
        np.float64(0.0),
        np.float64(-0.0),
        _nan(0x7FF8000000000000),
        _nan(0x7FF8000000000001),
    ]
    keys = [_key(v) for v in values]
    assert None not in keys
    # np.longlong is np.int64 itself where C long is 32 bits (Windows): the
    # same value of the same type, which must key alike, not a collision.
    distinct = {(type(v), v.tobytes() if isinstance(v, np.generic) else repr(v)) for v in values}
    assert len(set(keys)) == len(distinct)
    assert _key(np.int64(3)) == _key(np.int64(3))
    assert _key(_nan(0x7FF8000000000001)) == _key(_nan(0x7FF8000000000001))
    assert _key([np.int64(1), np.float64(2)]) is not None  # in a small list too


def test_what_its_bytes_do_not_fully_say_keeps_the_ordinary_path():
    """A subclass may carry state; longdouble's bytes carry padding; an
    object, string or date scalar is not a fixed-size number."""

    class Tagged(np.int64):
        pass

    for v in [Tagged(3), np.longdouble(3), np.str_("a"), np.datetime64("2020-01-01"), np.timedelta64(1, "D")]:
        assert _key(v) is None, type(v)


def test_a_numpy_subclass_argument_is_not_served_fast(c):
    class Tagged(np.int64):
        pass

    f = c.cache(described)
    for _ in range(4):
        f(Tagged(3))
    before = hit_memo.status()["fast_hits"]
    f(Tagged(3))
    assert hit_memo.status()["fast_hits"] == before


def test_numpy_is_not_imported_to_key_without_it(monkeypatch):
    monkeypatch.delitem(sys.modules, "numpy")
    assert hit_memo.numpy_scalars_keyed() == 0
    assert "numpy" not in sys.modules


def test_each_numpy_number_type_keys_three_apart_and_one_type_alike():
    """Every type the extension takes, on this platform's own mapping of C
    types (``np.longlong is np.int64`` and ``np.long`` a separate 32-bit
    type on Windows): two keys are equal exactly when the type object and
    the bytes are."""
    from cash._plain_data import numpy_scalar_types

    hit_memo.numpy_scalars_keyed()
    types = numpy_scalar_types()
    assert np.int64 in types and np.int32 in types
    keys = {}
    for t in types:
        three = t(3) if t is not np.bool_ else t(True)
        k = _key(three)
        assert k is not None, t
        assert _key(t(three)) == k
        assert k not in keys, (t, keys.get(k))
        keys[k] = t
    assert (_key(np.longlong(3)) == _key(np.int64(3))) == (np.longlong is np.int64)
    assert (_key(np.intc(3)) == _key(np.int32(3))) == (np.intc is np.int32)
    assert _key(np.int64(3)) != _key(np.int64(4))
