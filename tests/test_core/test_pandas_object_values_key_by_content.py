"""Python objects in a pandas column or index key by what they are, not by str().

``hash_pandas_object`` stringifies an array of Python objects as soon as one
of them is not a string, so ``1`` and ``'1'``, ``True`` and ``'True'``, a date
and its ISO string, or ``b'a'`` and ``'a'`` hashed alike, and a function
doubling the column was served ``2`` where ``'11'`` was due.
"""

from __future__ import annotations

import datetime
import decimal

import pytest

from cash.object_hashing import hash_pandas

pd = pytest.importorskip("pandas")


LOOKALIKES = {
    "int and str": (1, "1"),
    "bool and str": (True, "True"),
    "bytes and str": (b"a", "a"),
    "decimal and str": (decimal.Decimal("1.5"), "1.5"),
    "date and str": (datetime.date(2020, 1, 2), "2020-01-02"),
    "tuple and str": ((1, 2), "(1, 2)"),
}


@pytest.mark.parametrize("left, right", list(LOOKALIKES.values()), ids=list(LOOKALIKES))
def test_an_object_column_keys_on_each_value(left, right):
    one = pd.Series([left, "A7"], dtype=object)
    two = pd.Series([right, "A7"], dtype=object)
    assert hash_pandas(one) != hash_pandas(two)
    frame_one, frame_two = pd.DataFrame({"x": [1.0, 2.0], "o": one}), pd.DataFrame({"x": [1.0, 2.0], "o": two})
    assert hash_pandas(frame_one) != hash_pandas(frame_two)


@pytest.mark.parametrize("left, right", list(LOOKALIKES.values()), ids=list(LOOKALIKES))
def test_an_object_index_keys_on_each_label(left, right):
    one = pd.Series([1.0, 2.0], index=pd.Index([left, "A7"], dtype=object))
    two = pd.Series([1.0, 2.0], index=pd.Index([right, "A7"], dtype=object))
    assert hash_pandas(one) != hash_pandas(two)


def test_mixed_type_categories_key_on_the_codes():
    dtype = pd.CategoricalDtype([1, "1"])
    assert hash_pandas(pd.Series([1], dtype=dtype)) != hash_pandas(pd.Series(["1"], dtype=dtype))


def test_equal_values_key_alike_whatever_their_storage():
    """Positive control: equal columns built twice, and a column of sets
    whose iteration order differs, share a key."""
    rows = [1, "x", None, 2.5, {"b", "a"}]
    one = pd.DataFrame({"o": pd.Series(rows, dtype=object), "s": ["p", "q", None, "r", "s"]})
    two = pd.DataFrame({"o": pd.Series([1, "x", None, 2.5, {"a", "b"}], dtype=object), "s": ["p", "q", None, "r", "s"]})
    assert hash_pandas(one) == hash_pandas(two)


def test_the_decorator_serves_each_its_own_result(tmp_path):
    from cash import Cash, FileBackend

    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def doubled(s):
        return list(s * 2)

    assert doubled(pd.Series([1, "A7"], dtype=object)) == [2, "A7A7"]
    assert doubled(pd.Series(["1", "A7"], dtype=object)) == ["11", "A7A7"]
