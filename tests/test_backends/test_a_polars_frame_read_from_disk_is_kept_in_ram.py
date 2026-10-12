"""A polars frame read from disk is kept in RAM on its first read.

The RAM tier keeps a value restored from disk at once only when keeping it
costs no copy, and it knew that only of pandas tables. A polars frame, whose
store is a clone sharing its immutable buffers, waited for its second read:
after a restart an 80 MB frame was read from disk twice (272 ms, 221 ms)
before a RAM hit (0.4 ms). Whatever the reader does to a frame it was
handed never reaches a later read.
"""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


def _tiers(path) -> TieredBackend:
    return TieredBackend([InMemoryBackend(), FileBackend(str(path), flush_interval=0)])


def _written(path, key, value) -> None:
    writer = _tiers(path)
    writer.set(key, value, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()


def _frame():
    return pl.DataFrame({"x": list(range(10_000)), "s": [f"s{i % 7}" for i in range(10_000)]})


def _edit_cell(df):
    df[0, "x"] = -1


def _insert_column(df):
    df.insert_column(0, pl.Series("new", [0] * df.height))


def _replace_column(df):
    df.replace_column(0, pl.Series("x", [-1] * df.height))


def _drop_in_place(df):
    df.drop_in_place("s")


def _rename(df):
    df.columns = ["a", "b"]


def _extend(df):
    df.extend(df.head(3))


def _vstack(df):
    df.vstack(df.head(3), in_place=True)


def _hstack(df):
    df.hstack([pl.Series("y", [1] * df.height)], in_place=True)


MUTATORS = [_edit_cell, _insert_column, _replace_column, _drop_in_place, _rename, _extend, _vstack, _hstack]


@pytest.mark.parametrize(
    "value",
    [_frame(), _frame()["x"], {"variables": {"df": _frame(), "n": 3}, "rng": (1, 2)}],
    ids=["frame", "series", "notebook-entry"],
)
def test_is_kept_in_ram_on_its_first_read(tmp_path, value):
    _written(tmp_path, "k", value)
    reader = _tiers(tmp_path)
    meta, first = reader.get("k")
    assert meta["source"] == "DISK"
    meta, second = reader.get("k")
    assert meta["source"] == "RAM"
    reader.shutdown()
    if isinstance(value, dict):
        assert second["variables"]["df"].equals(value["variables"]["df"])
    else:
        assert second.equals(value)


@pytest.mark.parametrize("mutate", MUTATORS, ids=lambda f: f.__name__.strip("_"))
def test_what_the_first_reader_does_to_it_never_reaches_a_later_read(tmp_path, mutate):
    _written(tmp_path, "k", _frame())
    reader = _tiers(tmp_path)
    _meta, first = reader.get("k")
    mutate(first)
    meta, second = reader.get("k")
    assert meta["source"] == "RAM"
    assert second.equals(_frame())
    mutate(second)
    _meta, third = reader.get("k")
    reader.shutdown()
    assert third.equals(_frame())
