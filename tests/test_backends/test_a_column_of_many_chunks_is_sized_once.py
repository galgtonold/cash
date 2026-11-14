"""A frame grown by ``df = pd.concat([df, new])`` is sized and kept cheaply.

Its text columns come in thousands of Arrow chunks. Arrow works out a
ChunkedArray's ``nbytes`` afresh at every read (13 ms for 6,000 chunks)
and a statement read it up to eight times; the store then proved every
chunk's buffers Arrow's own one buffer at a time from Python, 80 ms more.
The size is now read once per array (`sizing.chunked_nbytes`), and a
column's chunks are proved in one call to the C extension. These pin that
the size is Arrow's own, how often it is read per statement, that the
statement's value is plain Python's, and that the stored frame keeps only
memory nothing else writes.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")

from cash import _native, sizing
from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend
from tests._cell_driver import run_cash_cell

ArrowExtensionArray = pytest.importorskip("pandas.core.arrays.arrow").ArrowExtensionArray

#: A frame of many small reads, as a loop of ``pd.concat`` leaves it.
SETUP = (
    "import numpy as np, pandas as pd\n"
    "def part(i):\n"
    "    return pd.DataFrame({\n"
    "        'name': pd.array([f's{i}_{j}' for j in range(3)], dtype='string[pyarrow]'),\n"
    "        'grp': pd.array(['a', 'b', 'c'], dtype='string[pyarrow]'),\n"
    "        'v': np.arange(3.0) + i,\n"
    "    })\n"
    "df = pd.concat([part(i) for i in range(400)])\n"
    "new = part(9999)\n"
)
GROW = "df = pd.concat([df, new])"


def _frame(n: int = 400) -> pd.DataFrame:
    parts = [
        pd.DataFrame({"name": pd.array([f"s{i}_{j}" for j in range(3)], dtype="string[pyarrow]"), "v": [1.0, 2.0, i]})
        for i in range(n)
    ]
    return pd.concat(parts)


def test_the_size_is_arrows_own():
    df = _frame()
    column = df["name"].array._pa_array
    assert column.num_chunks >= sizing._ARROW_MEMO_MIN_CHUNKS
    assert sizing.chunked_nbytes(column) == column.nbytes
    assert sizing.chunked_nbytes(column) == column.nbytes  # from the memo
    assert sizing.array_nbytes(df["name"].array) == df["name"].array.nbytes
    assert sizing.pandas_nbytes(df) == int(df.memory_usage(deep=True).sum())
    assert frame_sharing._total_bytes(df) == sum(block.values.nbytes for block in df._mgr.blocks)


def test_a_sliced_or_shared_chunk_is_counted_as_arrow_counts_it():
    big = pa.array([f"value {i}" for i in range(10_000)])
    shared = pa.chunked_array([big.slice(i * 10, 10) for i in range(50)] + [big] * 3)
    assert sizing.chunked_nbytes(shared) == shared.nbytes
    assert sizing._data_size(shared) == shared.nbytes


def test_the_memo_goes_with_the_array():
    column = _frame()["name"].array._pa_array
    sizing.chunked_nbytes(column)
    key = id(column)
    assert key in sizing._ARROW_SIZES
    del column
    gc.collect()
    assert key not in sizing._ARROW_SIZES


def test_a_write_into_the_column_is_sized_afresh():
    df = _frame()
    before = sizing.pandas_nbytes(df)
    df.iloc[0, df.columns.get_loc("name")] = "a much longer value than any of the others" * 20
    assert sizing.pandas_nbytes(df) == int(df.memory_usage(deep=True).sum()) > before


def test_a_statement_reads_each_new_columns_size_once(cash_magics, cash_instance, monkeypatch):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [SETUP, GROW]
    run_cash_cell(cash_magics, SETUP, cells=cells)
    ns = cash_magics.shell.user_ns
    expected = pd.concat([ns["df"], ns["new"]])

    read: list[int] = []
    pandas_reads: list[int] = []
    real = sizing._chunked_nbytes
    monkeypatch.setattr(sizing, "_chunked_nbytes", lambda chunked: read.append(id(chunked)) or real(chunked))
    monkeypatch.setattr(
        ArrowExtensionArray, "nbytes", property(lambda self: pandas_reads.append(1) or self._pa_array.nbytes)
    )
    run_cash_cell(cash_magics, GROW, cells=cells)

    pd.testing.assert_frame_equal(ns["df"], expected, check_exact=True)
    assert ns["df"]["name"].array._pa_array.num_chunks > 400
    # The two new text columns, each read once; nothing through pandas.
    assert len(read) <= 2 and len(set(read)) == len(read), read
    assert not pandas_reads


@pytest.mark.parametrize("many", [False, True], ids=["one at a time", "C extension"])
def test_only_chunks_over_memory_arrow_does_not_own_are_copied(many, monkeypatch):
    if frame_sharing._arrow_layout() is None:
        pytest.skip("Arrow's buffer owners cannot be told here")
    if many and getattr(_native.kernels, "pool_buffers_unproven", None) is None:
        pytest.skip("no C extension")
    if not many:
        monkeypatch.setattr(frame_sharing, "_MANY_CHUNKS", 10**9)
    numbers = np.arange(64, dtype=np.int64)
    pooled = pa.concat_arrays([pa.array(numbers)])
    chunks = [pooled] * 10 + [pa.array(numbers), pooled.slice(3, 20)] + [pooled] * 10 + [pa.array(numbers[:5])]
    column = pa.chunked_array(chunks)

    owned = frame_sharing._owned_arrow(column)

    assert frame_sharing._unowned_chunks(column.type, column.chunks) == [10, 22]
    kept = [c.buffers()[1].address == d.buffers()[1].address for c, d in zip(column.chunks, owned.chunks)]
    assert [i for i, same in enumerate(kept) if not same] == [10, 22]  # Arrow's own chunks shared, not copied
    assert owned.equals(column)
    assert frame_sharing.arrow_data_owned(owned)
    assert not frame_sharing.arrow_data_owned(column)
    numbers[:] = -1  # the numpy memory the two chunks wrap
    assert owned.to_pylist() == [*range(64)] * 10 + [*range(64), *range(3, 23)] + [*range(64)] * 10 + [*range(5)]


def test_a_stored_frame_is_not_changed_by_writes_into_the_live_one():
    df = _frame()
    numbers = np.arange(len(df), dtype=np.int64)
    df["wrapped"] = pd.arrays.ArrowExtensionArray(pa.chunked_array([pa.array(numbers)]))
    # Built afresh: pandas' own deep copy of an Arrow column shares its
    # buffers, the numpy memory under "wrapped" too.
    expected = df.drop(columns="wrapped").copy(deep=True)
    expected["wrapped"] = pd.array(numbers.tolist(), dtype="int64[pyarrow]")
    ram = InMemoryBackend()
    ram.set("k", df, {"execution_time": 1.0})

    df.iloc[0, df.columns.get_loc("name")] = "changed"
    df.iloc[1, df.columns.get_loc("v")] = -5.0
    numbers[:] = -1

    pd.testing.assert_frame_equal(ram.get("k")[1], expected, check_exact=True)


def test_the_numbers_a_text_column_spells_are_read_the_same_joined(monkeypatch):
    from cash.notebook.statement import identity_holders

    values = [f"{i:09d}" if i % 3 == 0 else (f"0x{i:07x}" if i % 3 == 1 else f"w{i}") for i in range(3000)]
    column = pd.array(values, dtype="string[pyarrow]")
    column = pd.concat([pd.Series(column[i : i + 5]) for i in range(0, 3000, 5)]).array
    assert column._pa_array.num_chunks == 600

    def read() -> list:
        identity_holders._ARROW_READS.clear()
        parts = identity_holders._arrow_text_numbers(column)
        return sorted(int(n) for part in parts for n in (part if isinstance(part, np.ndarray) else [part]))

    joined = read()  # one run of all 600 chunks
    assert len(identity_holders._joined_runs(pa, column._pa_array)) == 1
    monkeypatch.setattr(identity_holders, "_RUN_ROWS", 7)
    assert len(identity_holders._joined_runs(pa, column._pa_array)) == 300
    in_runs = read()
    monkeypatch.setattr(identity_holders, "_SMALL_CHUNK_ROWS", 0)
    assert identity_holders._joined_runs(pa, column._pa_array) == [column._pa_array]
    assert read() == joined == in_runs == sorted(i for i in range(3000) if i % 3 != 2)
    identity_holders._ARROW_READS.clear()
