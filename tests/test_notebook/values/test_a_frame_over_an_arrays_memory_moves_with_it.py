"""A frame built over another library's memory without a copy moves with it.

``s = pl.Series(arr)``, ``pl.DataFrame({'a': arr})``, ``pa.array(arr)`` and
``pd.Series(arr, copy=False)`` read ``arr``'s buffer, so ``arr[:10] = 0``
changes them; ``pl.from_pandas(df)`` reads ``df``'s blocks, and
``df['a'].to_numpy()`` is a view of one. Lineage froze each at creation, so a
cached call over the frame kept answering with the old data after the write.
Each notebook is checked against plain Python running the same cells.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement.derivation_edges import is_uncacheable_alias
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

SETUP = (
    "import os, time\n"
    "import numpy as np, pandas as pd, polars as pl, pyarrow as pa, pyarrow.compute as pc\n"
    "def total(x):\n"
    "    os.write(COUNT, b'x')\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    if isinstance(x, pl.LazyFrame): x = x.collect()\n"
    "    if isinstance(x, (pa.Array, pa.ChunkedArray)): return float(pc.sum(x).as_py())\n"
    "    if isinstance(x, (pa.Table, pa.RecordBatch)): return float(pc.sum(x.column(0)).as_py())\n"
    "    if isinstance(x, (pl.DataFrame, pd.DataFrame)): x = x[x.columns[0]]\n"
    "    return float(np.asarray(x).sum())\n"
    "def make(a):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return pl.Series(a)"
)
ARR = "arr = np.arange(1000.0)"
CALL = "r = total(s)"
WRITE = "arr[:10] = 1000.0"

#: name -> the cells after SETUP; the last one reads ``r``.
NOTEBOOKS = {
    "polars Series": [ARR, "s = pl.Series(arr)", CALL, WRITE, CALL],
    "polars int Series": ["arr = np.arange(1000)", "s = pl.Series(arr)", CALL, "arr[:10] = 7", CALL],
    "polars DataFrame": [ARR, "s = pl.DataFrame({'a': arr})", CALL, WRITE, CALL],
    "polars Array column": [ARR, "s = pl.Series(arr.reshape(-1, 2))", CALL, WRITE, CALL],
    "polars from Arrow": [ARR, "s = pl.from_arrow(pa.array(arr))", CALL, WRITE, CALL],
    "polars slice of a borrower": [ARR, "d = pl.DataFrame({'a': arr})", "s = d.slice(0, 500)", CALL, WRITE, CALL],
    "polars over a view": [ARR, "v = arr[:500]", "s = pl.Series(v)", CALL, WRITE, CALL],
    "polars written through a view": [ARR, "s = pl.Series(arr)", CALL, "v = arr[:20]", "v[:] = 1000.0", CALL],
    "polars over an array a dict holds": [
        "data = {'x': np.arange(1000.0)}",
        "s = pl.Series(data['x'])",
        CALL,
        "data['x'][:10] = 1000.0",
        CALL,
    ],
    "polars written in a loop": [ARR, "s = pl.Series(arr)", CALL, "for i in range(10):\n    arr[i] = 1000.0", CALL],
    "polars written by a function": [
        ARR,
        "s = pl.Series(arr)",
        CALL,
        "def fill(a):\n    a[:10] = 1000.0",
        "fill(arr)",
        CALL,
    ],
    "polars from pandas": [
        "p = pd.DataFrame({'a': np.arange(1000.0)})",
        "s = pl.from_pandas(p)",
        CALL,
        "p.iloc[:10, 0] = 1000.0",
        CALL,
    ],
    "polars LazyFrame": [ARR, "s = pl.LazyFrame({'a': arr})", CALL, WRITE, CALL],
    "polars LazyFrame of a borrower": [ARR, "d = pl.DataFrame({'a': arr})", "s = d.lazy()", CALL, WRITE, CALL],
    "pandas Series copy=False": [ARR, "s = pd.Series(arr, copy=False)", CALL, WRITE, CALL],
    "pandas DataFrame copy=False": [ARR, "s = pd.DataFrame({'a': arr}, copy=False)", CALL, WRITE, CALL],
    "pandas written into the array": [
        ARR,
        "s = pd.Series(arr, copy=False)",
        "r = total(arr)",
        "s.iloc[:10] = 1000.0",
        "r = total(arr)",
    ],
    "numpy view of a pandas column": [
        "p = pd.DataFrame({'a': np.arange(1000.0)})",
        "s = p['a'].to_numpy()",
        CALL,
        "p.loc[:9, 'a'] = 1000.0",
        CALL,
    ],
    "arrow array": [ARR, "s = pa.array(arr)", CALL, WRITE, CALL],
    "arrow chunked array": [ARR, "s = pa.chunked_array([arr])", CALL, WRITE, CALL],
    "arrow table": [ARR, "s = pa.table({'a': arr})", CALL, WRITE, CALL],
    "a plain statement over a polars Series": [
        ARR,
        "s = pl.Series(arr)",
        f"r = float(s.sum()) + (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}) or 0)",
        WRITE,
        f"r = float(s.sum()) + (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}) or 0)",
    ],
    "a slow call making the Series, run twice": [
        ARR,
        "s = make(arr)",
        "s = make(arr)",
        CALL,
        WRITE,
        CALL,
    ],
}


def _run_all(magics, cells: list[str]) -> None:
    """Run *cells* in order as one notebook; the same text twice is two cells."""
    ids = [f"cell-{index}" for index in range(len(cells))]
    for cell, cell_id in zip(cells, ids):
        run_cash_cell(magics, cell, cells=cells, cell_ids=ids, cell_id=cell_id)


def _plain(cells: list[str], count_fd: int) -> float:
    ns: dict = {"COUNT": count_fd}
    for cell in cells:
        exec(compile(cell, "<cell>", "exec"), ns)
    return ns["r"]


@pytest.fixture
def count_fd(tmp_path):
    fd = os.open(tmp_path / "calls", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    yield fd
    os.close(fd)


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_a_write_to_the_memory_reaches_the_cached_call(cash_magics, count_fd, name):
    cells = [SETUP, *NOTEBOOKS[name]]
    expected = _plain(cells, count_fd)
    cash_magics.shell.user_ns["COUNT"] = count_fd
    _run_all(cash_magics, cells)
    assert cash_magics.shell.user_ns["r"] == expected


@pytest.mark.parametrize(
    "make",
    ["s = pd.Series(arr)", "s = pl.Series(arr * 1.0)", "s = pl.Series(arr.astype(np.int64))"],
    ids=["pandas-copies", "polars-of-a-new-array", "polars-of-a-cast"],
)
def test_a_frame_with_its_own_copy_stays_cached(cash_magics, tmp_path, count_fd, make):
    """Control: no shared memory, no new key -- the second call is a hit."""
    cells = [SETUP, ARR, make, CALL, WRITE, CALL]
    cash_magics.shell.user_ns["COUNT"] = count_fd
    _run_all(cash_magics, cells)
    assert (tmp_path / "calls").read_bytes() == b"x"


def test_a_frame_over_a_named_array_is_not_stored(cash_magics, statement_processor):
    """A restored copy would no longer read ``arr``'s memory."""
    run_cash_cell(cash_magics, SETUP.replace("COUNT", "2"))
    run_cash_cell(cash_magics, ARR)
    metrics = statement_processor.process_statement(f"s = pl.Series(arr); time.sleep({ABOVE_PERSISTENCE_FLOOR_S})")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert any("Live-alias" in reason for reason in metrics.get("uncacheable_reasons", ()))


def test_a_borrower_of_a_live_array_is_an_uncacheable_alias():
    arr = np.arange(10.0)
    ns = {"arr": arr}
    assert is_uncacheable_alias(pl.Series(arr), ns)
    assert is_uncacheable_alias({"t": pa.table({"a": arr})}, ns)
    assert not is_uncacheable_alias(pl.Series(arr.copy()), ns)
