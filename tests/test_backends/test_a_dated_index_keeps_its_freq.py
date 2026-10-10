"""A table with a dated index keeps the index's ``freq`` through the cache.

The RAM tier shares a frozen copy of a table (`cash.backends.frame_sharing`),
and the disk write pickles that copy. Freezing rebuilt a DatetimeArray or
TimedeltaArray without its ``freq``, so a hit, and every disk entry, came
back with ``index.freq`` None: ``df.shift(1, freq=df.index.freq)`` then
shifts the data instead of the dates, and the hit, passed on to another
cached function, keys differently from the fresh table.
"""

from __future__ import annotations

import pickle
import textwrap

import pytest

from tests._scripts import run_python

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.backends.memory_backend import InMemoryBackend  # noqa: E402

TABLES = {
    "daily index": lambda: pd.DataFrame({"sales": np.arange(5.0)}, index=pd.date_range("2024-01-01", periods=5, freq="D")),
    "hourly timedelta index": lambda: pd.DataFrame(
        {"v": np.arange(4.0)}, index=pd.timedelta_range("1D", periods=4, freq="h")
    ),
    "series by month end": lambda: pd.Series(np.arange(3.0), index=pd.date_range("2024-01-31", periods=3, freq="ME")),
    "dates as column labels": lambda: pd.DataFrame(
        np.arange(6.0).reshape(2, 3), columns=pd.date_range("2024-01-01", periods=3, freq="D")
    ),
    "tz-aware daily index": lambda: pd.DataFrame(
        {"v": np.arange(3.0)}, index=pd.date_range("2024-01-01", periods=3, freq="D", tz="Europe/Berlin")
    ),
}


def _axes(table):
    return [table.index] if table.ndim == 1 else [table.index, table.columns]


def _freqs(table):
    return [getattr(axis, "freq", None) for axis in _axes(table)]


@pytest.mark.parametrize("name", TABLES)
def test_a_ram_hit_keeps_the_freq(name):
    table = TABLES[name]()
    backend = InMemoryBackend()
    backend.set("k", table, {"execution_time": 1.0})
    for _ in range(2):
        hit = backend.get("k")[1]
        assert _freqs(hit) == _freqs(table)


@pytest.mark.parametrize("name", TABLES)
def test_the_copy_the_disk_write_pickles_keeps_the_freq(name):
    """The disk write pickles the RAM tier's own copy (`private_copy`)."""
    table = TABLES[name]()
    backend = InMemoryBackend()
    metadata = {"execution_time": 1.0}
    backend.set("k", table, metadata)
    stored = backend.private_copy("k", default=table)
    assert _freqs(pickle.loads(pickle.dumps(stored))) == _freqs(table)


@pytest.mark.parametrize("name", TABLES)
def test_a_table_read_from_disk_keeps_the_freq_when_kept_in_ram(name):
    """A disk read is frozen where it is (`promote`), and every hit of it after."""
    table = pickle.loads(pickle.dumps(TABLES[name]()))
    expected = _freqs(table)
    backend = InMemoryBackend()
    backend.promote("k", table, {"execution_time": 1.0})
    assert _freqs(table) == expected
    assert _freqs(backend.get("k")[1]) == expected


def test_freq_dependent_code_gives_the_same_numbers_on_every_hit(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            import pandas as pd, numpy as np, cash

            @cash.cache
            def load():
                return pd.DataFrame({"sales": np.arange(5.0)},
                                    index=pd.date_range("2024-01-01", periods=5, freq="D"))

            for _ in range(2):
                daily = load()
                print(daily.index.freqstr, daily.shift(1, freq=daily.index.freq)["sales"].tolist())
            """
        ),
        encoding="utf-8",
    )
    expected = "D [0.0, 1.0, 2.0, 3.0, 4.0]"
    for process in ("computes, then a RAM hit", "two disk hits"):
        lines = run_python(script, cwd=tmp_path, cache_dir=tmp_path / "cache").stdout.strip().splitlines()
        assert lines[-2:] == [expected, expected], process
