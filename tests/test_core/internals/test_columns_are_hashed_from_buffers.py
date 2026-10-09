"""Work counts of the content hashes: columns are read from their buffers.

pandas' per-row ``hash_pandas_object`` ran over every numeric column, text
columns of pandas' ``str`` type and object columns were pickled item by
item, and polars frames went through ``hash_rows``: 0.4-0.65 s per 100 MB
before the digest itself. Each test counts the slow path and asserts it is
not taken. Counters: ``tests/_work_counts.py``.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from cash.content_hashers import builtin_hash
from tests._work_counts import method_calls, pickled_bytes

pytestmark = pytest.mark.core


def test_numbers_dates_and_nullable_columns_skip_the_per_row_hash():
    n = 1000
    frame = pd.DataFrame(
        {
            "f": np.arange(n, dtype=np.float64),
            "i": np.arange(n),
            "b": np.arange(n) % 2 == 0,
            "d": pd.date_range("2020-01-01", periods=n, freq="min"),
            "tz": pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC"),
            "m": pd.array(list(range(n - 1)) + [None], dtype="Int64"),
        },
        index=pd.Index(np.arange(n) * 2),
    )
    with method_calls(pd.util, "hash_pandas_object") as calls:
        assert builtin_hash(frame) is not None
    assert calls.calls == 0, "a numeric column went through pandas' per-row hash"


def test_a_text_column_is_not_pickled_item_by_item():
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {"a": np.arange(10_000.0), "s": pd.array([f"name_{i % 500}" for i in range(10_000)], dtype="string[pyarrow]")}
    )
    with pickled_bytes() as pickled:
        assert builtin_hash(frame) is not None
    assert pickled.bytes < 1000, f"{pickled.bytes} bytes pickled to key an Arrow text column"


def test_an_object_table_of_floats_and_bools_is_not_pickled():
    """``X.to_numpy()`` of floats and dummies: the object table sklearn gets."""
    table = np.empty((5000, 4), dtype=object)
    table[:, 0] = np.arange(5000.0).tolist()
    table[:, 1] = [i % 2 == 0 for i in range(5000)]
    table[:, 2] = np.linspace(0, 1, 5000).tolist()
    table[:, 3] = list(range(5000))
    with pickled_bytes() as pickled:
        assert builtin_hash(table) is not None
        assert builtin_hash(pd.DataFrame(table)) is not None
    assert pickled.bytes < 1000, f"{pickled.bytes} bytes pickled to key floats, bools and ints"


def test_a_polars_frame_skips_the_row_hash():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"a": np.arange(1000.0), "s": [f"x{i}" for i in range(1000)]})
    with method_calls(pl.DataFrame, "hash_rows") as calls:
        assert builtin_hash(frame) is not None
    assert calls.calls == 0


def test_a_big_array_is_hashed_on_several_threads(monkeypatch):
    from cash import bulk_digest

    monkeypatch.setattr(bulk_digest, "_THREADS", 4)
    started: list[str] = []
    real_start = threading.Thread.start

    def start(self):
        started.append(self.name)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    assert builtin_hash(np.zeros(4 * 1024 * 1024)) is not None  # 32 MiB
    assert started.count("cash-digest") >= 2, started
