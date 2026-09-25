"""A reader bound by name -- ``from pyarrow.parquet import read_table`` -- is
tracked like the module attribute.

The wrappers go onto the module attributes while a tracker is open, but a name
imported at the top of a module was bound to the ORIGINAL function before any
tracker opened. The read went through unwrapped, no dependency was recorded,
and after the data changed the cached function kept returning the old value,
while ``pq.read_table(...)`` beside it recomputed. ``ParquetFile`` and the
Arrow IPC readers were not tracked under any spelling.
"""

from __future__ import annotations

import importlib
import os
import sqlite3
import sys
import textwrap
from sqlite3 import connect

import pytest

from cash import Cash

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
from pyarrow.parquet import read_table

read_parquet_table = pq.read_table  # an alias under another name


def _settle(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def _parquet(path, value):
    pq.write_table(pa.table({"x": [value]}), path)
    _settle(path)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _recomputes_after_edit(c, path, write, read):
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def load(p):
        runs.append(1)
        return read(p)

    write(path, 1)
    assert load(str(path)) == 1
    assert load(str(path)) == 1
    assert len(runs) == 1
    write(path, 2)
    return load(str(path))


def test_a_from_imported_reader_is_tracked(c, tmp_path):
    got = _recomputes_after_edit(c, tmp_path / "d.parquet", _parquet, lambda p: read_table(p).column("x")[0].as_py())
    assert got == 2


def test_an_alias_under_another_name_is_tracked(c, tmp_path):
    got = _recomputes_after_edit(
        c, tmp_path / "d.parquet", _parquet, lambda p: read_parquet_table(p).column("x")[0].as_py()
    )
    assert got == 2


def test_a_from_imported_sqlite_connect_is_tracked(c, tmp_path):
    def write(path, value):
        conn = sqlite3.connect(path)
        conn.execute("create table if not exists t (x int)")
        conn.execute("delete from t")
        conn.execute("insert into t values (?)", (value,))
        conn.commit()
        conn.close()

    def read(p):
        conn = connect(p)
        try:
            return conn.execute("select x from t").fetchone()[0]
        finally:
            conn.close()

    assert _recomputes_after_edit(c, tmp_path / "d.db", write, read) == 2


def test_a_reader_imported_by_a_helper_module_is_tracked(c, tmp_path, monkeypatch):
    (tmp_path / "helper_by_name.py").write_text(
        textwrap.dedent("""
            from pyarrow.parquet import read_table

            def first(p):
                return read_table(p).column("x")[0].as_py()
        """),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    helper = importlib.import_module("helper_by_name")
    try:
        assert _recomputes_after_edit(c, tmp_path / "d.parquet", _parquet, helper.first) == 2
    finally:
        sys.modules.pop("helper_by_name", None)


def test_parquet_file_is_tracked_and_still_a_class(c, tmp_path):
    def read(p):
        f = pq.ParquetFile(p)
        assert isinstance(f, pq.ParquetFile)
        return f.read().column("x")[0].as_py()

    assert _recomputes_after_edit(c, tmp_path / "d.parquet", _parquet, read) == 2


def test_an_arrow_ipc_file_read_through_a_memory_map_is_tracked(c, tmp_path):
    def write(path, value):
        table = pa.table({"x": [value]})
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        _settle(path)

    def read(p):
        with pa.memory_map(p) as source:
            return pa.ipc.open_file(source).read_all().column("x")[0].as_py()

    assert _recomputes_after_edit(c, tmp_path / "d.arrow", write, read) == 2


def test_the_name_keys_the_same_inside_another_cached_call(c, tmp_path):
    """While a tracker is open the name holds cash's wrapper; a call nested in
    another cached call must still find the entry a top-level call stored."""
    path = tmp_path / "d.parquet"
    _parquet(path, 1)
    runs: list[str] = []

    @c.cache(assume_safe=True)
    def inner(p):
        runs.append("inner")
        return read_table(p).column("x")[0].as_py()

    @c.cache(assume_safe=True)
    def outer(p):
        runs.append("outer")
        return inner(p) + 1

    assert inner(str(path)) == 1
    assert outer(str(path)) == 2
    assert runs == ["inner", "outer"]
