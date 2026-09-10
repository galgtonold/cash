"""pyarrow's readers are tracked, and a reader called by keyword still works.

CAS-115, round-17 tester r17s1 (#6). pyarrow reads files in C++, so nothing
passes through a patched ``open()``, and none of its readers were registered.
A cached function that switched from pandas to ``pyarrow.csv.read_csv`` for
speed recorded no file dependency at all; a whole new export returned
yesterday's numbers, 3/3, with no warning.

Found while adding them: the generic path wrapper demanded the path as its
first POSITIONAL parameter. The wrappers are installed process-wide on the
first cached call, so from then on ``pd.read_csv(filepath_or_buffer=p)``,
``np.load(file=p)`` or ``pq.read_table(source=p)`` raised TypeError anywhere
in the process -- inside cached code or not.
"""
from __future__ import annotations

import time

import pytest

from cash import Cash

pa = pytest.importorskip("pyarrow")

pytestmark = pytest.mark.core


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _write_csv(path, values):
    path.write_text("v\n" + "\n".join(str(v) for v in values) + "\n", encoding="utf-8")


def _reader_suite():
    import pyarrow.csv as pacsv
    import pyarrow.feather as feather
    import pyarrow.json as pajson
    import pyarrow.parquet as pq
    return {
        "pyarrow.csv.read_csv": (".csv", lambda p: pacsv.read_csv(p)),
        "pyarrow.parquet.read_table": (".parquet", lambda p: pq.read_table(p)),
        "pyarrow.feather.read_table": (".feather", lambda p: feather.read_table(p)),
        "pyarrow.json.read_json": (".jsonl", lambda p: pajson.read_json(p)),
    }


def _write(path, suffix, values):
    import pyarrow.feather as feather
    import pyarrow.parquet as pq
    table = pa.table({"v": values})
    if suffix == ".csv":
        _write_csv(path, values)
    elif suffix == ".parquet":
        pq.write_table(table, str(path))
    elif suffix == ".feather":
        feather.write_feather(table, str(path))
    else:
        path.write_text("\n".join(f'{{"v": {v}}}' for v in values) + "\n", encoding="utf-8")


@pytest.mark.parametrize("reader", sorted(_reader_suite()))
def test_a_new_file_through_a_pyarrow_reader_recomputes(c, tmp_path, reader):
    """THE BUG: a new export read through pyarrow returned the old total."""
    suffix, read = _reader_suite()[reader]
    path = tmp_path / f"data{suffix}"
    _write(path, suffix, [1, 2, 3])
    runs = []

    @c.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        time.sleep(0.2)
        return int(read(p).column("v").to_pandas().sum())

    assert total(str(path)) == 6
    assert total(str(path)) == 6
    assert len(runs) == 1, "an unchanged file recomputed"

    _write(path, suffix, [10, 20, 30, 40])
    assert total(str(path)) == 100, f"{reader} served the old file's answer"
    assert len(runs) == 2


def test_a_reader_called_by_keyword_works_everywhere(c, tmp_path):
    """The wrapper bug: keyword calls raised TypeError once any call had run."""
    import pandas as pd
    import pyarrow.parquet as pq

    csv = tmp_path / "x.csv"
    _write_csv(csv, [1, 2])
    parquet = tmp_path / "x.parquet"
    pq.write_table(pa.table({"v": [3, 4]}), str(parquet))

    @c.cache(assume_safe=True)
    def warm(p):
        time.sleep(0.2)
        return int(pd.read_csv(p)["v"].sum())

    warm(str(csv))                                    # installs the wrappers

    assert int(pd.read_csv(filepath_or_buffer=str(csv))["v"].sum()) == 3
    assert pq.read_table(source=str(parquet)).num_rows == 2


def test_a_keyword_path_is_still_tracked(c, tmp_path):
    """And the keyword form records the dependency, not just survives."""
    import pyarrow.parquet as pq

    parquet = tmp_path / "x.parquet"
    pq.write_table(pa.table({"v": [1, 2]}), str(parquet))
    runs = []

    @c.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        time.sleep(0.2)
        return int(pq.read_table(source=p).column("v").to_pandas().sum())

    assert total(str(parquet)) == 3
    pq.write_table(pa.table({"v": [5, 6, 7]}), str(parquet))
    assert total(str(parquet)) == 18
    assert len(runs) == 2
