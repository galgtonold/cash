"""An Arrow argument's digest is remembered only when nothing can change it.

A PyArrow table, batch, array or chunked array never changes what it holds,
and the memory under it does not change either when Arrow allocated it, so
its digest is read once and then looked up while the object lives. Arrow
over memory someone else owns -- ``pa.array(a)`` and ``pa.table({...: a})``
over a numpy array copy nothing -- changes when that memory is written,
and must be read again: served from memory, a call over it answered with
the values from before the write. Polars frames are never remembered: one
built over a numpy array (``pl.Series(a)``, ``pl.from_numpy``) shares its
memory too, and polars cannot say which of its memory is its own.
"""

from __future__ import annotations

import pytest

from cash import Cash, FileBackend

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def _total(x) -> float:
    if isinstance(x, (pa.Table, pa.RecordBatch)):
        x = x.column(0)
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        return float(pc.sum(x).as_py())
    if type(x).__module__.startswith("polars"):
        x = x.to_frame() if hasattr(x, "to_frame") else x
        return float(x.sum().row(0)[0])
    return float(sum(x))


BUILDERS = {
    "table": lambda a: pa.table({"x": a}),
    "record batch": lambda a: pa.record_batch({"x": a}),
    "array": lambda a: a if isinstance(a, pa.Array) else pa.array(a),
    "chunked array": lambda a: pa.chunked_array([a[:500], a[500:]]),
    "table slice": lambda a: pa.table({"x": a}).slice(10, 900),
}


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_arrow_over_a_numpy_array_is_read_again_after_a_write(tmp_path, kind):
    c = _cash(tmp_path)
    calls = []

    @c.cache
    def total(x):
        calls.append(1)
        return _total(x)

    a = np.arange(1000.0)
    value = BUILDERS[kind](a)
    first = total(value)
    assert total(value) == first
    assert len(calls) == 1  # a hit while nothing changed
    a[10:20] = 1e6
    assert total(value) == _total(value) != first
    assert len(calls) == 2


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_arrow_in_arrows_own_memory_keys_by_its_rows(tmp_path, kind):
    """Remembered, and still keyed by the rows: equal data shares a result,
    other data does not, whatever the object."""
    c = _cash(tmp_path)
    calls = []

    @c.cache
    def total(x):
        calls.append(1)
        return _total(x)

    def owned(a):
        # Built from Python values: memory Arrow allocated.
        return BUILDERS[kind](pa.array(a.tolist()))

    first = owned(np.arange(1000.0))
    assert total(first) == _total(first)
    assert total(first) == _total(first)
    assert total(owned(np.arange(1000.0))) == _total(first)
    assert len(calls) == 1
    other = owned(np.arange(1000.0) * 2)
    assert total(other) == _total(other) != _total(first)
    assert len(calls) == 2


def test_a_polars_series_over_a_numpy_array_is_read_again_after_a_write(tmp_path):
    pl = pytest.importorskip("polars")
    c = _cash(tmp_path)

    @c.cache
    def total(x):
        return float(x.sum())

    a = np.arange(1000.0)
    for series in (pl.Series("x", a), pl.from_numpy(a.reshape(-1, 1), schema=["x"])["x"]):
        before = total(series)
        a[:10] += 1e6
        assert total(series) == float(series.sum()) != before
