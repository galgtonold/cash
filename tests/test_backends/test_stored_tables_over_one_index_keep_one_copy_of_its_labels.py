"""Tables stored over one index keep one copy of its labels.

Every result of a loop over one table carries that table's index: a
Series of floats over a year of minutes holds as many bytes of dates as of
values. The RAM tier copied the labels of each table it stored -- twice,
once with the table and again when it froze the copy -- so the tier held a
copy of the same dates per entry, and paged in fresh memory for each.

The tier now freezes the labels as it copies them, and a later table over
the same caller's labels takes that frozen copy, once its bytes, type,
dtype and ``freq`` are checked to be the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend
from tests._label_writes import label_array

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="tables are shared under copy-on-write only")

META = {"execution_time": 10.0}


def _labels(value):
    return value.index._data._ndarray


def _stored(ram, key):
    return ram.get(key)[1]


def test_series_over_one_index_share_its_frozen_labels():
    index = pd.date_range("2024-01-01", periods=50_000, freq="min", tz="UTC")
    ram = InMemoryBackend()
    ram.set("a", pd.Series(np.arange(50_000.0), index=index), META)
    ram.set("b", pd.Series(np.arange(50_000.0) * 2, index=index), META)
    ram.set("c", pd.DataFrame({"x": np.ones(50_000), "y": np.zeros(50_000)}, index=index), META)
    a, b, c = _stored(ram, "a"), _stored(ram, "b"), _stored(ram, "c")
    assert np.shares_memory(_labels(a), _labels(b))
    assert np.shares_memory(_labels(a), _labels(c))
    assert not np.shares_memory(_labels(a), index._data._ndarray)
    assert a.index.equals(index) and c.index.equals(index)
    assert a.index.freq == index.freq and c.index.freq == index.freq


def test_labels_written_in_place_since_are_copied_again():
    index = pd.date_range("2024-01-01", periods=1000, freq="h")
    original = index.copy(deep=True)
    ram = InMemoryBackend()
    ram.set("a", pd.Series(np.arange(1000.0), index=index), META)
    label_array(index)[0] = pd.Timestamp("1999-01-01")
    ram.set("b", pd.Series(np.arange(1000.0), index=index), META)
    assert _stored(ram, "a").index.equals(original)
    assert _stored(ram, "b").index[0] == pd.Timestamp("1999-01-01")
    assert not np.shares_memory(_labels(_stored(ram, "a")), _labels(_stored(ram, "b")))


def test_the_same_dates_in_another_zone_keep_their_zone():
    utc = pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC")
    berlin = utc.tz_convert("Europe/Berlin")  # over the very same int64 memory
    ram = InMemoryBackend()
    ram.set("utc", pd.Series(np.arange(1000.0), index=utc), META)
    ram.set("berlin", pd.Series(np.arange(1000.0), index=berlin), META)
    assert str(_stored(ram, "utc").index.tz) == "UTC"
    assert str(_stored(ram, "berlin").index.tz) == "Europe/Berlin"
    assert _stored(ram, "berlin").index.equals(berlin)


def test_a_hit_cannot_write_the_shared_labels():
    index = pd.date_range("2024-01-01", periods=1000, freq="h")
    ram = InMemoryBackend()
    ram.set("a", pd.Series(np.arange(1000.0), index=index), META)
    ram.set("b", pd.Series(np.arange(1000.0), index=index), META)
    with pytest.raises(ValueError):
        label_array(_stored(ram, "a").index)[0] = pd.Timestamp("1999-01-01")
    assert _stored(ram, "b").index.equals(index)
