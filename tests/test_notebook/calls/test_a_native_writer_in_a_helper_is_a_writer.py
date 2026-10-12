"""A helper of yours that saves through polars, pyarrow, numpy or a model
library is a writer, though the file is written from native code.

The runtime observer sees Python's ``open()`` only; ``df.write_parquet(p)``,
``pq.write_table(t, p)``, ``pacsv.write_csv(t, p)`` and ``model.save_model(p)``
write from Rust or C++. The static check knew none of the names, so the
statement calling such a helper was stored and the next Run All skipped the
write.
"""

from cash.analysis.namespace_effects import user_callee_writing_files


class _Lib:
    """Stands for a library module or frame: only the call's spelling counts."""

    def __getattr__(self, name):
        return lambda *a, **k: None


pl = pq = pacsv = np = xr = _Lib()


def polars_parquet(rows, p):
    df = pl.DataFrame({"a": rows})
    df.write_parquet(p)
    return len(rows)


def polars_csv(rows, p):
    pl.DataFrame({"a": rows}).write_csv(p)
    return len(rows)


def pyarrow_table(t, p):
    pq.write_table(t, p)


def pyarrow_csv(t, p):
    pacsv.write_csv(t, p)


def catboost(model, p):
    model.save_model(p)


def numpy_text(a, p):
    np.savetxt(p, a)


def xarray(ds, p):
    ds.to_netcdf(p)


def computes(rows):
    df = pl.DataFrame({"a": rows})
    return df.select("a")


def test_each_native_writer_is_seen():
    for fn in (polars_parquet, polars_csv, pyarrow_table, pyarrow_csv, catboost, numpy_text, xarray):
        assert user_callee_writing_files(fn) == fn.__name__, fn.__name__


def test_a_helper_that_only_computes_is_no_writer():
    assert user_callee_writing_files(computes) is None
