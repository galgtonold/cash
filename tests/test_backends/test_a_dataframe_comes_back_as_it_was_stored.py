"""A DataFrame restored from disk is the frame the call returned.

Parquet wrote a RangeIndex out as an int64 column, so a decorated function
returning ``pd.DataFrame({"a": np.arange(n)})`` handed back a plain Index on
the next process's hit: code that checked the index type, or the memory it
used, saw a different frame on a hit than on a miss. Frames Parquet cannot
store as they are (non-string labels, an index frequency, a column pyarrow
cannot convert) are pickled instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

import cash
from cash.backends.serialization import ParquetSerializer

FRAMES = {
    "range index": lambda: pd.DataFrame({"a": np.arange(5)}),
    "stepped range index": lambda: pd.DataFrame({"a": np.arange(5)}, index=pd.RangeIndex(10, 20, 2, name="i")),
    "categorical and tz columns": lambda: pd.DataFrame(
        {
            "c": pd.Categorical(["a", "b"], categories=["b", "a"], ordered=True),
            "t": pd.date_range("2024-01-01", periods=2, tz="Europe/Berlin"),
        }
    ),
    "index with a frequency": lambda: pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2024", periods=3, freq="D")),
    "integer column labels": lambda: pd.DataFrame({0: [1, 2], 1: [3, 4]}),
    "duplicate column labels": lambda: pd.DataFrame([[1, 2]], columns=["a", "a"]),
    "mixed object column": lambda: pd.DataFrame({"o": [1, "a"]}),
    "no columns": lambda: pd.DataFrame(),
}


@pytest.mark.parametrize("name", FRAMES)
def test_the_serializer_gives_the_frame_back_unchanged(name):
    df = FRAMES[name]()
    s = ParquetSerializer()
    back = s.deserialize(s.serialize(df))
    pd.testing.assert_frame_equal(df, back, check_index_type=True, check_column_type=True, check_freq=True)
    assert type(back.index) is type(df.index)


def test_a_plain_frame_is_still_stored_as_parquet():
    """Positive control: the common frame keeps the Parquet format."""
    assert ParquetSerializer().serialize(FRAMES["range index"]()).startswith(b"PAR1")


def test_a_disk_hit_in_a_new_process_keeps_the_range_index(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            import numpy as np, pandas as pd
            import cash

            @cash.cache
            def make(n):
                fd = os.open(os.environ["RUNS"], os.O_WRONLY | os.O_APPEND | os.O_CREAT)
                os.write(fd, b"x")
                os.close(fd)
                return pd.DataFrame({"a": np.arange(n)})

            print(type(make(10).index).__name__)
            """
        ),
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / "cache"), RUNS=str(runs), PYTHONDONTWRITEBYTECODE="1")
    # The cash under test, whichever checkout it is in.
    src = os.path.dirname(os.path.dirname(cash.__file__))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
    seen = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
        assert p.returncode == 0, p.stderr
        seen.append(p.stdout.strip().splitlines()[-1])
    assert runs.read_bytes() == b"x", "the second process did not hit the disk entry"
    assert seen == ["RangeIndex", "RangeIndex"], seen
