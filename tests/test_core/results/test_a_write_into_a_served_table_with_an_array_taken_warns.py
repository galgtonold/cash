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


@pytest.mark.parametrize(
    "write",
    [
        "df.fillna(0.0, inplace=True)",
        "df.replace(1.0, 9.0, inplace=True)",
        "df.where(df > 0, 0.0, inplace=True)",
        "df.mask(df > 1, 0.0, inplace=True)",
        "df.clip(lower=0.5, inplace=True)",
    ],
    ids=["fillna", "replace", "where", "mask", "clip"],
)
def test_a_write_in_place_with_no_array_held_does_not_warn(tmp_path, write):
    done = _run(tmp_path, f"{write}\nvalue = df['t'].iloc[2]\n")
    computed = done.stdout.split()[1]
    assert f"served {computed}" in done.stdout, done.stdout
    assert "CACHE-SERVED-VIEW" not in done.stderr, done.stderr


def test_a_view_taken_and_dropped_inside_a_function_warns(tmp_path):
    done = _run(
        tmp_path,
        """
        def first_after_write(d):
            v = d["t"].to_numpy()
            d.iloc[0, 0] = 100.0
            return v[0]
        value = first_after_write(df)
        """,
    )
    assert "computed 100.0" in done.stdout, done.stdout
    assert "served 0.0" in done.stdout, done.stdout
    assert "CACHE-SERVED-VIEW" in done.stderr, done.stderr


def test_without_exact_reference_counts_it_warns_once_that_it_cannot_check(tmp_path):
    """On a free-threaded Python with the GIL off, reference counts are not
    exact, so a write cannot tell an array held from none: said once."""
    script = tmp_path / "job.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys, warnings
            warnings.simplefilter("always")
            sys._is_gil_enabled = lambda: False
            import pandas as pd, cash

            @cash.cache
            def load():
                return pd.DataFrame({"t": [0.0, 1.0, 2.0]})

            load()
            for _ in range(2):
                df = load()
                df.iloc[0, 0] = 100.0
            """
        ),
        encoding="utf-8",
    )
    done = run_python(script, cwd=tmp_path)
    assert done.stderr.count("CACHE-SERVED-VIEW") == 1, done.stderr
    assert "cannot tell" in done.stderr, done.stderr


@pytest.mark.parametrize(
    ("take", "write"),
    [
        ("df['t'].to_numpy()", "df.iloc[:, 0] = v"),
        ("df['t'].to_numpy()", "df.loc[:, 't'] = v"),
        ("df.to_numpy()", "df.iloc[:, :] = v"),
        ("df.to_numpy()", "df.mask(df > 1, v, inplace=True)"),
    ],
    ids=["iloc", "loc", "whole", "mask"],
)
def test_writing_from_the_array_itself_warns(tmp_path, take, write):
    """The array the write is given is held by pandas while it writes, and
    by the caller: it is still the caller's reader."""
    done = _run(tmp_path, f"v = {take}\n{write}\ndf.iloc[0, 0] = 100.0\nvalue = v.ravel()[0]\n")
    assert "computed 100.0" in done.stdout, done.stdout
    assert "served 0.0" in done.stdout, done.stdout
    assert "CACHE-SERVED-VIEW" in done.stderr, done.stderr
