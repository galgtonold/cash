"""A reader given a directory, a glob or a list reads every file in it, and
each of those files is a dependency.

``pd.read_parquet("dd")``, ``ds.dataset("dd")`` and
``pl.read_parquet("dd/*.parquet")`` were recorded as the directory alone, or,
for a glob, as a path that does not exist and was then dropped. An in-place
rewrite of a partition does not move its directory's mtime, so all three
served the old total; the glob missed a new partition too.
"""

from __future__ import annotations

import os

import pytest

from cash import Cash

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write(path, value):
    pq.write_table(pa.table({"x": [value]}), path)
    st = os.stat(path)
    # Settled, so a rewrite within the same second is still a new mtime and
    # the directory keeps its own.
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def _readers():
    import pandas as pd
    import pyarrow.dataset as ds

    found = {
        "pandas.read_parquet(dir)": lambda d: int(pd.read_parquet(d)["x"].sum()),
        "pyarrow.dataset(dir)": lambda d: int(ds.dataset(d).to_table().column("x").to_pandas().sum()),
        "pyarrow.parquet.read_table(dir)": lambda d: int(pq.read_table(d).column("x").to_pandas().sum()),
        "pyarrow.dataset([files])": lambda d: int(
            ds.dataset(sorted(os.path.join(d, n) for n in os.listdir(d))).to_table().column("x").to_pandas().sum()
        ),
    }
    try:
        import polars as pl
    except ImportError:
        return found
    found["polars.read_parquet(glob)"] = lambda d: int(pl.read_parquet(os.path.join(d, "*.parquet"))["x"].sum())
    found["polars.scan_parquet(dir)"] = lambda d: int(pl.scan_parquet(d + os.sep).collect()["x"].sum())
    return found


@pytest.mark.parametrize("reader", sorted(_readers()))
def test_an_edit_and_a_new_file_both_recompute(tmp_path, reader):
    read = _readers()[reader]
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    dd = tmp_path / "dd"
    dd.mkdir()
    _write(dd / "a.parquet", 1)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def total(d):
        runs.append(1)
        return read(d)

    assert total(str(dd)) == 1
    assert total(str(dd)) == 1
    assert len(runs) == 1

    _write(dd / "a.parquet", 5)
    assert total(str(dd)) == 5

    if reader == "pyarrow.dataset([files])":
        return  # a list names its files; a new one is not in it
    _write(dd / "b.parquet", 10)
    assert total(str(dd)) == 15


def test_bookkeeping_files_are_not_dependencies(tmp_path):
    """``_SUCCESS`` and dot-files are what the readers skip; a job rewriting
    its marker does not invalidate the read."""
    import pandas as pd

    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    dd = tmp_path / "dd"
    dd.mkdir()
    _write(dd / "a.parquet", 1)
    (dd / "_SUCCESS").write_text("", encoding="utf-8")
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def total(d):
        runs.append(1)
        return int(pd.read_parquet(d)["x"].sum())

    assert total(str(dd)) == 1
    (dd / "_SUCCESS").write_text("done", encoding="utf-8")
    assert total(str(dd)) == 1
    assert len(runs) == 1
