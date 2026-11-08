"""A write into a table a cached function returned from the cache, while an
array taken from the table is held, warns CACHE-SERVED-VIEW.

The RAM tier serves the table as a shallow copy of its own frozen copy, so
pandas copies the column before writing it, and ``v = df['t'].to_numpy()``
taken before ``df.iloc[0, 0] = 100`` keeps ``0.0`` where a plain run, whose
table no other table shares, shows it ``100.0``. That cannot be matched
without copying every hit, so the write warns, at the line that writes.
Each script runs in a fresh interpreter: no notebook counts the writes
after a statement there.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pd = pytest.importorskip("pandas")

SCRIPT = """
import warnings
warnings.simplefilter("always")
import pandas as pd, cash

@cash.cache
def load():
    return pd.DataFrame({{"t": [0.0, 1.0, 2.0]}})

for run in ("computed", "served"):
    df = load()
{body}
    print(run, value)
"""


def _run(tmp_path, body: str):
    script = tmp_path / "job.py"
    script.write_text(SCRIPT.format(body=textwrap.indent(textwrap.dedent(body), "    ")), encoding="utf-8")
    return run_python(script, cwd=tmp_path)


def test_a_view_held_across_the_write_warns_on_the_served_table(tmp_path):
    done = _run(
        tmp_path,
        """
        v = df["t"].to_numpy()
        df.iloc[0, 0] = 100.0  # the write
        value = v[0]
        """,
    )
    assert "computed 100.0" in done.stdout, "control: the computed table is the caller's own"
    assert "served 0.0" in done.stdout, done.stdout
    assert done.stderr.count("CACHE-SERVED-VIEW") == 1, done.stderr
    lines = (tmp_path / "job.py").read_text(encoding="utf-8").splitlines()
    write = next(n for n, line in enumerate(lines, 1) if "# the write" in line)
    assert f"job.py:{write}:" in done.stderr, done.stderr  # blamed on the line that writes


def test_values_and_a_series_written_alone_warn_too(tmp_path):
    done = _run(
        tmp_path,
        """
        y = df["t"]
        df = df.copy()
        v = y.values
        y.iloc[0] = 100
        value = v[0]
        """,
    )
    assert "served 0.0" in done.stdout, done.stdout
    assert "CACHE-SERVED-VIEW" in done.stderr, done.stderr


def test_a_write_with_no_array_held_does_not_warn(tmp_path):
    done = _run(
        tmp_path,
        """
        df.iloc[0, 0] = 100.0
        value = df["t"].iloc[0]
        """,
    )
    assert "served 100.0" in done.stdout, done.stdout
    assert "CACHE-SERVED-VIEW" not in done.stderr, done.stderr
