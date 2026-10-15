"""A value that cannot change is read once while it lives, then looked up.

An Arrow table, batch, array or chunked array in memory Arrow allocated
(`ArrowDigests`), and a frozenset of primitives (its sorted form, the slow
part of its key): each is remembered by identity, behind a weak reference,
and dropped when it is collected. Arrow over a numpy array, and a frozenset
holding an object that can change, are read every time.
"""

from __future__ import annotations

import gc

import pytest

from cash import canonical_form, content_hashers
from cash.backends import frame_sharing
from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT, builtin_hash

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")


@pytest.fixture
def reads(monkeypatch):
    """How often `hash_pyarrow` reads a value."""
    counted = []
    original = content_hashers.hash_pyarrow

    def counting(value):
        counted.append(type(value).__name__)
        return original(value)

    monkeypatch.setattr(content_hashers, "hash_pyarrow", counting)
    return counted


def _owned():
    values = pa.array([float(i) for i in range(10_000)])
    return {
        "table": pa.table({"x": values, "s": pa.array([str(i) for i in range(10_000)])}),
        "batch": pa.record_batch({"x": values}),
        "array": values,
        "chunked": pa.chunked_array([values.slice(0, 5000), values.slice(5000)]),
        "slice": pa.table({"x": values}).slice(100, 5000),
        "dictionary": pa.table({"d": pa.array(["a", "b", "a"] * 1000).dictionary_encode()}),
    }


@pytest.mark.parametrize("kind", list(_owned()))
def test_arrow_in_its_own_memory_is_read_once(reads, kind):
    value = _owned()[kind]
    assert frame_sharing.arrow_data_owned(value)
    first = builtin_hash(value)
    assert first is not None
    assert builtin_hash(value) == first
    assert builtin_hash(value) == first
    assert len(reads) == 1
    assert first == content_hashers.ArrowDigests(4).digest(value)  # a fresh read gives it too


@pytest.mark.parametrize(
    "build",
    [
        lambda a: pa.table({"x": a}),
        lambda a: pa.array(a),
        lambda a: pa.chunked_array([a]),
        lambda a: pa.record_batch({"x": a}),
        lambda a: pa.py_buffer(a) and pa.Array.from_buffers(pa.float64(), len(a), [None, pa.py_buffer(a)]),
    ],
)
def test_arrow_over_a_numpy_array_is_read_every_time_and_sees_a_write(reads, build):
    a = np.arange(10_000.0)
    value = build(a)
    assert not frame_sharing.arrow_data_owned(value)
    before = builtin_hash(value)
    assert builtin_hash(value) == before
    a[5000] = -1.0
    after = builtin_hash(value)
    assert after != before
    assert len(reads) == 3


def test_a_remembered_digest_goes_with_its_value(reads):
    table = _owned()["table"]
    builtin_hash(table)
    key = id(table)
    assert key in content_hashers.ARROW_DIGESTS._entries
    del table
    gc.collect()
    assert key not in content_hashers.ARROW_DIGESTS._entries


def test_a_new_value_at_a_remembered_address_is_read():
    """The weak reference, not the address, says it is the same value."""
    memo = content_hashers.ArrowDigests(16)
    first = pa.table({"x": pa.array([1.0, 2.0])})
    memo.digest(first)
    entry = memo._entries.get(id(first))
    other = pa.table({"x": pa.array([3.0, 4.0])})
    memo._entries[id(other)] = entry  # as if `other` took `first`'s address
    assert memo.digest(other) == content_hashers.hash_pyarrow(other) != content_hashers.hash_pyarrow(first)


def test_polars_values_are_never_remembered(reads):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"x": [float(i) for i in range(10_000)]})
    builtin_hash(frame)
    builtin_hash(frame)
    assert len(content_hashers.ARROW_DIGESTS._entries.get(id(frame)) or ()) == 0


def _sorted_calls(monkeypatch):
    counted = []
    original = canonical_form._plain_data.content_dumps

    def counting(value):
        counted.append(1)
        return original(value)

    monkeypatch.setattr(canonical_form._plain_data, "content_dumps", counting)
    return counted


def test_a_frozenset_of_primitives_is_sorted_once(monkeypatch):
    sorts = _sorted_calls(monkeypatch)
    value = frozenset(range(1000)) | {"a", b"b", 1.5, None}
    first = canonical_bytes(value, BUILTIN_CONTENT)
    once = len(sorts)
    assert once >= len(value)
    assert canonical_bytes(value, BUILTIN_CONTENT) == first
    assert len(sorts) - once < 10  # not sorted again
    assert canonical_bytes(frozenset(list(value)), BUILTIN_CONTENT) == first  # an equal set keys alike
    assert len(sorts) - once >= len(value)  # the copy was sorted
    key = id(value)
    del value
    gc.collect()
    assert canonical_form._SORTED_SETS.get(key) is None


class _Box:
    def __init__(self, n):
        self.n = n


def test_a_frozenset_holding_an_object_that_can_change_is_walked_every_time(monkeypatch):
    sorts = _sorted_calls(monkeypatch)
    box = _Box(1)
    value = frozenset(range(1000)) | {box}
    first = canonical_bytes(value, BUILTIN_CONTENT)
    once = len(sorts)
    box.n = 2
    assert canonical_bytes(value, BUILTIN_CONTENT) != first
    assert len(sorts) - once >= len(value)


def test_a_set_is_walked_every_time(monkeypatch):
    sorts = _sorted_calls(monkeypatch)
    value = set(range(1000))
    first = canonical_bytes(value, BUILTIN_CONTENT)
    once = len(sorts)
    value.add(5000)
    assert canonical_bytes(value, BUILTIN_CONTENT) != first
    assert len(sorts) - once >= len(value)
