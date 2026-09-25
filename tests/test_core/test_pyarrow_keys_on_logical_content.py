"""A PyArrow argument keys on the rows it holds, not on the buffers under it.

A slice shares its parent's buffers and differs only in an offset, and a
dictionary column's buffers are its indices alone. Keyed on the buffers, every
batch of ``to_batches()`` was served the first batch's result, and two dictionary
columns with the same indices over different words shared one entry.
"""

from __future__ import annotations

import pickle

import pytest

from cash import Cash, FileBackend
from cash.object_hashing import hash_pyarrow

pa = pytest.importorskip("pyarrow")


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def test_each_batch_of_a_table_gets_its_own_result(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def values(batch):
        return batch.column("x").to_pylist()

    batches = pa.table({"x": [1, 2, 3, 4], "s": ["a", "b", "c", "d"]}).to_batches(max_chunksize=2)
    assert values(batches[0]) == [1, 2]
    assert values(batches[1]) == [3, 4]


@pytest.mark.parametrize(
    "column",
    [
        [1, 2, 3, 4],
        ["a", "bb", "c", "dd"],
        [[1], [2, 3], [4], [5, 6]],
        [1, None, 3, None],
    ],
)
def test_equal_length_slices_of_one_table_key_apart(column):
    table = pa.table({"x": column})
    assert hash_pyarrow(table.slice(0, 2)) != hash_pyarrow(table.slice(2, 2))


def test_a_slice_keys_like_its_pickled_copy():
    """A slice a cached function returned is restored by pickle; the restored
    value must key as the original did, or every consumer re-runs."""
    table = pa.table({"x": [1, None, 3, None, 5], "s": ["a", "b", "c", "d", "e"]})
    part = table.slice(1, 2)
    assert hash_pyarrow(part) == hash_pyarrow(pickle.loads(pickle.dumps(part)))


def test_dictionary_columns_key_on_their_dictionary(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def labels(t):
        return t.column("c").to_pylist()

    assert labels(pa.table({"c": pa.array(["red", "blue"]).dictionary_encode()})) == ["red", "blue"]
    assert labels(pa.table({"c": pa.array(["cat", "dog"]).dictionary_encode()})) == ["cat", "dog"]


def test_chunks_with_different_dictionaries_hash():
    chunked = pa.chunked_array([pa.array(["a", "b"]).dictionary_encode(), pa.array(["c"]).dictionary_encode()])
    other = pa.chunked_array([pa.array(["a", "b"]).dictionary_encode(), pa.array(["d"]).dictionary_encode()])
    one, two = hash_pyarrow(pa.table({"c": chunked})), hash_pyarrow(pa.table({"c": other}))
    assert one is not None and two is not None
    assert one != two


def test_a_table_and_a_batch_of_the_same_rows_key_apart():
    table = pa.table({"x": [1, 2]})
    assert hash_pyarrow(table) != hash_pyarrow(table.to_batches()[0])
