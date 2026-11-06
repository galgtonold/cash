"""A column of long strings, each its own object, is pickled as it is.

Joining a column's strings into one buffer (`text_columns.encode`) pays
when they are short: pickle writes each string as an object of its own. A
long string pickle writes as a length and its bytes, and the buffer only
adds passes: 50,000 unique review texts (66 MB) were written in 81 ms
against plain pickle's 42 ms, and read back slower too. Such a column now
goes to pickle as it is; short strings, and long ones held again and again
(each written once, with codes), still go through the buffer.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from cash.backends import text_columns
from cash.backends.serialization import PickleSerializer, reduce_text_objects

ROWS = 5_000


def _objects(items: list) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def _long_unique() -> list:
    return [f"{i:06d} " + "review text " * 25 for i in range(ROWS)]  # ~300 characters


def test_long_unique_strings_are_not_joined():
    objects = _objects(_long_unique())
    assert text_columns.encode(objects) is None
    assert reduce_text_objects(objects) is None


def test_they_come_back_as_they_were_stored():
    items = _long_unique()
    items[7] = None
    column = pd.array(items, dtype=pd.StringDtype("python"))
    stream, buffers = PickleSerializer().serialize_split(pd.Series(column))
    back = pickle.loads(stream, buffers=buffers)
    pd.testing.assert_series_equal(back, pd.Series(column))
    assert type(back.array) is pd.arrays.StringArray


def test_short_unique_strings_are_still_joined():
    encoded = text_columns.encode(_objects([f"id-{i:07d}" for i in range(ROWS)]))
    assert encoded is not None and encoded.codes is None


@pytest.mark.parametrize("length", [10, 300])
def test_strings_held_again_and_again_are_still_written_once(length):
    names = [f"{n}-" + "x" * length for n in range(20)]
    encoded = text_columns.encode(_objects([names[i % 20] for i in range(ROWS)]))
    assert encoded is not None and encoded.codes is not None and encoded.strings == 20


def test_the_mean_length_decides_not_the_longest():
    items = [f"{i:06d}" if i % 50 else f"{i:06d} " + "long " * 100 for i in range(ROWS)]
    assert text_columns.encode(_objects(items)) is not None
