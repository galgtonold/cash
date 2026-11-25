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
from cash.content_hashers import hash_pyarrow

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


def _sizes(indices, entries):
    return pa.DictionaryArray.from_arrays(indices, entries)


def test_a_dictionary_columns_order_and_unused_entries_are_keyed(tmp_path):
    """``to_pandas()`` turns the dictionary into the categories of a
    Categorical, and ``.dictionary`` shows it: two tables of equal values
    over different dictionaries are not one argument."""
    c = _cash(tmp_path)

    @c.cache
    def categories(t):
        return list(t.to_pandas()["size"].cat.categories)

    short = pa.table({"size": _sizes([0, 1, 0], ["S", "L"])})
    long = pa.table({"size": _sizes([1, 0, 1], ["L", "S", "XL"])})
    assert short.column("size").to_pylist() == long.column("size").to_pylist()
    assert categories(short) == ["S", "L"]
    assert categories(long) == ["L", "S", "XL"]


@pytest.mark.parametrize(
    "other",
    [
        _sizes([1, 0, 1], ["L", "S"]),  # another order
        _sizes([0, 1, 0], ["S", "L", "XL"]),  # an entry no row uses
        _sizes([0, 2, 0], ["S", "S2", "L"]),
    ],
    ids=["order", "unused entry", "unused entry between"],
)
def test_equal_values_over_another_dictionary_hash_apart(other):
    base = _sizes([0, 1, 0], ["S", "L"])
    assert base.to_pylist() == other.to_pylist()
    for wrap in (lambda a: a, lambda a: pa.chunked_array([a]), lambda a: pa.table({"c": a})):
        assert hash_pyarrow(wrap(base)) != hash_pyarrow(wrap(other))
        assert hash_pyarrow(wrap(base)) == hash_pyarrow(wrap(_sizes([0, 1, 0], ["S", "L"])))


def test_a_dictionary_column_keys_alike_in_any_chunking_and_as_a_slice():
    whole = _sizes([0, 1, 0, 2, None, 1], ["S", "L", "XL"])
    one = pa.chunked_array([whole])
    split = pa.chunked_array([whole.slice(0, 2), whole.slice(2, 1), whole.slice(3)])
    assert hash_pyarrow(one) == hash_pyarrow(split)
    # Chunks built over dictionaries of their own, as two files give them:
    # keyed as the one dictionary ``to_pandas`` gives them.
    parts = pa.chunked_array([_sizes([0, 1, 0], ["S", "L"]), _sizes([0, 1], ["XL", "L"])])
    joined = _sizes([0, 1, 0, 2, 1], ["S", "L", "XL"])
    assert parts.to_pylist() == joined.to_pylist()
    assert hash_pyarrow(parts) == hash_pyarrow(pa.chunked_array([joined]))
    tail = _sizes([9, 9, 0, 2, None, 1], ["S", "L", "XL"] + [f"x{i}" for i in range(7)]).slice(2)
    assert hash_pyarrow(tail) != hash_pyarrow(whole.slice(2))  # other entries
    assert hash_pyarrow(whole.slice(2)) == hash_pyarrow(_sizes([0, 2, None, 1], ["S", "L", "XL"]))
    assert hash_pyarrow(whole) != hash_pyarrow(_sizes([0, 1, 0, 2, 0, 1], ["S", "L", "XL"]))  # the missing row


def test_a_pandas_column_over_an_arrow_dictionary_keys_its_dictionary():
    pd = pytest.importorskip("pandas")
    from cash.content_hashers import hash_pandas

    def series(array):
        return pd.Series(pd.arrays.ArrowExtensionArray(array))

    assert hash_pandas(series(_sizes([0, 1], ["S", "L"]))) != hash_pandas(series(_sizes([1, 0], ["L", "S"])))
    assert hash_pandas(series(_sizes([0, 1], ["S", "L"]))) == hash_pandas(series(_sizes([0, 1], ["S", "L"])))
