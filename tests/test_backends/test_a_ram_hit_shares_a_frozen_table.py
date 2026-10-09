"""The RAM tier shares a pandas table's data instead of copying it.

Under copy-on-write, the tier copies a table once, when storing it, and
hands out shallow copies of that copy: a hit of a 250 MB table costs
microseconds instead of a 250 MB copy. The copy is frozen first
(`cash.backends.frame_sharing`): read-only, and marked shared for good, so
pandas copies a block before every write. These tests pin that no holder of
a stored or returned table -- the caller who stored it, a hit, another hit,
a pickle of either -- can change what another holder sees, and that the
caller's own table is left as writable as it was.
"""

from __future__ import annotations

import gc
import pickle

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")


def _frame(n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "k": np.arange(n) % 7,
            "when": pd.date_range("2024-01-01", periods=n, freq="min"),
            "name": [f"u{i % 13}" for i in range(n)],
            "day": [pd.Timestamp("2024-01-01").date()] * n,  # object column of immutable cells
        },
        index=pd.Index(np.arange(n) * 2, name="row"),
    )


#: Tables the tier copies once and shares: without and with an object column
#: (dates as Python objects: the store checks that its cells cannot change).
MAKERS = {"numbers_and_text": lambda: _frame().drop(columns=["day"]), "with_date_objects": _frame}


@pytest.fixture(params=MAKERS, autouse=False)
def make(request):
    return MAKERS[request.param]


def _store(backend: InMemoryBackend, value) -> None:
    backend.set("k", value, {"execution_time": 1.0})


def _hit(backend: InMemoryBackend):
    return backend.get("k")[1]


class TestOnlyTheStoreCopies:
    def test_a_store_copies_the_callers_data_once(self):
        df = _frame().drop(columns=["day"])
        backend = InMemoryBackend()
        _store(backend, df)
        stored = backend._store["k"][1]
        assert not np.shares_memory(stored["x"].to_numpy(), df["x"].to_numpy())

    def test_a_hit_shares_the_stored_data(self):
        backend = InMemoryBackend()
        _store(backend, _frame())
        first, second = _hit(backend), _hit(backend)
        assert np.shares_memory(first["x"].to_numpy(), second["x"].to_numpy())
        assert np.shares_memory(first["when"].to_numpy(), second["when"].to_numpy())

    def test_a_table_inside_an_entry_is_shared_too(self):
        backend = InMemoryBackend()
        df = _frame().drop(columns=["day"])
        _store(backend, {"variables": {"df": df, "n": [1, 2]}, "rng": (1, 2, 3)})
        first, second = _hit(backend)["variables"]["df"], _hit(backend)["variables"]["df"]
        assert np.shares_memory(first["x"].to_numpy(), second["x"].to_numpy())
        assert not np.shares_memory(first["x"].to_numpy(), df["x"].to_numpy())


#: Writes made through pandas, each of which must stay in the table written.
PANDAS_WRITES = {
    "loc": lambda d: d.loc.__setitem__((d.index[0], "x"), -1.0),
    "iloc": lambda d: d.iloc.__setitem__((1, 1), -5),
    "at": lambda d: d.at.__setitem__((d.index[2], "when"), pd.Timestamp("1999-01-01")),
    "column": lambda d: d.__setitem__("x", d["x"] * 0),
    "fillna_inplace": lambda d: d.fillna({"x": 0.0}, inplace=True),
    "replace_inplace": lambda d: d.replace(3.0, -3.0, inplace=True),
    "mask_inplace": lambda d: d["x"].mask(d["x"] > 5, 0, inplace=True),
    "sort_inplace": lambda d: d.sort_values("x", ascending=False, inplace=True),
    "update": lambda d: d.update(pd.DataFrame({"x": [-7.0]}, index=d.index[:1])),
    "object_cell": lambda d: d.loc.__setitem__((d.index[0], "name"), None),
    "text_cell": lambda d: d.loc.__setitem__((d.index[0], "name"), "zz"),
    "series_setitem": lambda d: d["k"].__setitem__(d.index[0], 99),
    "index_name": lambda d: setattr(d.index, "name", "renamed"),
    "attrs": lambda d: d.attrs.__setitem__("note", "x"),
}


class TestAPandasWriteStaysInTheTableWritten:
    @pytest.mark.parametrize("write", PANDAS_WRITES.values(), ids=PANDAS_WRITES.keys())
    def test_writing_a_hit_reaches_neither_the_entry_nor_another_hit(self, write, make):
        backend = InMemoryBackend()
        _store(backend, make())
        hit, other = _hit(backend), _hit(backend)
        write(hit)
        pd.testing.assert_frame_equal(other, make())
        pd.testing.assert_frame_equal(_hit(backend), make())

    @pytest.mark.parametrize("write", PANDAS_WRITES.values(), ids=PANDAS_WRITES.keys())
    def test_writing_the_stored_original_does_not_reach_the_entry(self, write, make):
        df = make()
        backend = InMemoryBackend()
        _store(backend, df)
        hit = _hit(backend)
        write(df)
        pd.testing.assert_frame_equal(_hit(backend), make())
        pd.testing.assert_frame_equal(hit, make())

    @pytest.mark.parametrize("write", PANDAS_WRITES.values(), ids=PANDAS_WRITES.keys())
    def test_a_hit_is_writable_after_the_entry_is_gone(self, write, make):
        backend = InMemoryBackend()
        _store(backend, make())
        hit, other = _hit(backend), _hit(backend)
        backend.clear()
        gc.collect()
        write(hit)  # pandas still copies first: the data stays frozen
        pd.testing.assert_frame_equal(other, make())

    def test_the_written_table_sees_its_own_write(self, make):
        backend = InMemoryBackend()
        _store(backend, make())
        hit = _hit(backend)
        hit.loc[hit.index[0], "x"] = -1.0
        hit["k"] += 1
        assert hit["x"].iloc[0] == -1.0 and hit["k"].iloc[0] == 1


#: Writes through a handle to the memory itself, past copy-on-write.
HANDLE_WRITES = {
    "series_array": lambda d: d["x"].array.__setitem__(0, -1.0),
    "values_made_writable": lambda d: (setattr((v := d["x"].values).flags, "writeable", True), v.__setitem__(0, -1)),
    "to_numpy_made_writable": lambda d: (setattr((v := d["k"].to_numpy()).flags, "writeable", True), v.__setitem__(0, -1)),
    "asarray_of_array": lambda d: (setattr((v := np.asarray(d["x"].array)).flags, "writeable", True), v.__setitem__(0, -1)),
    "date_array": lambda d: d["when"].array.__setitem__(0, pd.Timestamp("1999-01-01")),
    "object_array": lambda d: d.iloc[:, -1].array.__setitem__(0, None),
    "index_array": lambda d: d.index.array.__setitem__(0, -1),
    "columns_array": lambda d: d.columns.array.__setitem__(0, "renamed"),
    "frame_values_made_writable": lambda d: (setattr((v := d[["x"]].values).flags, "writeable", True), v.__setitem__(0, -1)),
}


class TestAHandleWriteNeverReachesAnotherHolder:
    @pytest.mark.parametrize("write", HANDLE_WRITES.values(), ids=HANDLE_WRITES.keys())
    @pytest.mark.parametrize("holder", ["hit", "original"])
    def test_a_write_through_a_handle_is_refused_or_stays_put(self, write, holder, make):
        df = make()
        backend = InMemoryBackend()
        _store(backend, df)
        hit, other = _hit(backend), _hit(backend)
        target = hit if holder == "hit" else df
        if holder == "hit":
            try:
                write(target)
            except ValueError:
                pass  # read-only: the memory is shared
        else:
            write(target)  # the caller's own table: as writable as before the store
        pd.testing.assert_frame_equal(other, make())
        pd.testing.assert_frame_equal(_hit(backend), make())
        if holder == "hit":
            pd.testing.assert_frame_equal(df, make())


class TestAPickleOfASharedTableIsAnOrdinaryTable:
    @pytest.mark.parametrize("protocol", [4, 5])
    @pytest.mark.parametrize("out_of_band", [False, True])
    def test_it_loads_writable_and_without_cash(self, protocol, out_of_band, make):
        backend = InMemoryBackend()
        _store(backend, make())
        hit = _hit(backend)
        buffers: list = []
        kwargs = {"buffer_callback": buffers.append} if out_of_band and protocol == 5 else {}
        data = pickle.dumps(hit, protocol=protocol, **kwargs)
        assert b"cash" not in data
        loaded = pickle.loads(data, buffers=[bytearray(b.raw()) for b in buffers])
        pd.testing.assert_frame_equal(loaded, make())
        loaded.loc[loaded.index[0], "x"] = -1.0  # no copy-on-write mark survives a pickle
        loaded["when"].array[0] = pd.Timestamp("1999-01-01")
        pd.testing.assert_frame_equal(_hit(backend), make())

    def test_a_series_pickles_writable_too(self):
        backend = InMemoryBackend()
        backend.set("s", pd.Series(np.arange(100_000, dtype=float)), {"execution_time": 1.0})
        loaded = pickle.loads(pickle.dumps(backend.get("s")[1], protocol=5))
        loaded.iloc[0] = -1.0
        assert backend.get("s")[1].iloc[0] == 0.0


class TestWhatIsNotSharedIsCopied:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: pd.DataFrame({"n": pd.array([1, 2, None], dtype="Int64"), "w": [1.0, 2.0, 3.0]}),
            lambda: pd.DataFrame({"c": pd.Categorical(["a", "b", "a"]), "w": [1.0, 2.0, 3.0]}),
            lambda: pd.DataFrame({"w": [1.0, 2.0, 3.0]}, index=pd.MultiIndex.from_tuples([(1, 2), (1, 3), (2, 2)])),
        ],
        ids=["nullable", "categorical", "multiindex"],
    )
    def test_a_table_cash_does_not_freeze_is_copied_and_stays_writable(self, make):
        df = make()
        backend = InMemoryBackend()
        _store(backend, df)
        hit = _hit(backend)
        assert not np.shares_memory(hit["w"].to_numpy(), df["w"].to_numpy())
        hit["w"].array[0] = -1.0  # writable as ever
        assert _hit(backend)["w"].iloc[0] == 1.0

    def test_a_table_over_an_array_the_caller_still_holds_is_copied(self):
        data = np.arange(12, dtype=float).reshape(4, 3)
        df = pd.DataFrame(data, columns=["a", "b", "c"], copy=False)
        backend = InMemoryBackend()
        _store(backend, df)
        data[0, 0] = -1.0  # straight past pandas
        assert _hit(backend)["a"].iloc[0] == 0.0
        data[1, 1] = -2.0  # the caller's array was not frozen
        assert df["b"].iloc[1] == -2.0

    def test_a_subclass_is_copied(self):
        class Sub(pd.DataFrame):
            @property
            def _constructor(self):
                return Sub

        df = Sub({"w": [1.0, 2.0]})
        backend = InMemoryBackend()
        _store(backend, df)
        assert not np.shares_memory(_hit(backend)["w"].to_numpy(), df["w"].to_numpy())


def test_a_frozen_table_is_still_checked_by_the_decorators_frame_memo():
    """The shared mark is not a reference from outside pandas: a table cash
    froze is as memoable as before (`frame_borrows_its_data`)."""
    from cash.decorator.arg_hashing import frame_borrows_its_data

    backend = InMemoryBackend()
    _store(backend, _frame().drop(columns=["day"]))
    assert not frame_borrows_its_data(_hit(backend), since=1 << 62)


class TestTheCallersTableStaysAsItWas:
    """Cash never changes what an object the caller holds allows: the table
    stored, and every view of it, are as writable after the store as before."""

    @pytest.mark.parametrize("make", MAKERS.values(), ids=MAKERS.keys())
    def test_its_columns_and_views_stay_writable(self, make):
        df = make()
        series = df["k"]  # a view of the block, taken before the store
        handles = [df["x"].to_numpy(), df["when"].to_numpy(), df.index.to_numpy(), df.values]
        before = [h.flags.writeable for h in handles]
        backend = InMemoryBackend()
        _store(backend, df)
        _store(backend, {"variables": {"df": df}})
        assert [h.flags.writeable for h in handles] == before
        df["x"].array[0] = -1.0
        series.array[1] = -2
        df["when"].array[0] = pd.Timestamp("1999-01-01")
        df.index.array[0] = -4
        assert df["x"].iloc[0] == -1.0 and df["k"].iloc[1] == -2 and df.index[0] == -4
        pd.testing.assert_frame_equal(_hit(backend)["variables"]["df"], make())

    def test_a_table_kept_with_bytes_leaves_the_callers_other_columns_writable(self):
        df = pd.DataFrame({"tags": [[1], [2], [3]], "w": [1.0, 2.0, 3.0]})
        backend = InMemoryBackend()
        _store(backend, df)
        df["w"].array[0] = -1.0
        assert df["w"].iloc[0] == -1.0 and _hit(backend)["w"].iloc[0] == 1.0
