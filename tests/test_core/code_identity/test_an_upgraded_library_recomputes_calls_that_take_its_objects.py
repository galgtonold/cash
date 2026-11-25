"""An upgraded library recomputes the calls that were handed its objects.

``def summarise(df): return df.describe()`` names no library: the code that
runs is the one the argument brings. The installed versions in the key were
those of the libraries the function's code NAMES, so after ``pip install -U``
such a function was served what the old version computed, with no warning.

The version of the library that defines the class of each argument, of
``self`` and of the objects they hold is in the key too, with what that
library requires, and so is the library of a function handed over as a
value.
"""

from __future__ import annotations

import os
import shutil
import textwrap

import numpy as np
import pandas as pd
import pytest

from cash.decorator.library_versions import held_library_versions
from tests._scripts import CASH_SRC, run_python
from tests._work_counts import hashed_bytes

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

JOB = """
import sys
import functools
import cash
import carrier  # the user's own module: holds a table, names no library


@cash.cache
def by_argument(table):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return table.total()


@cash.cache
def in_a_list(tables):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return sum(t.total() for t in tables)


@cash.cache
def in_a_dict(tables):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return sum(t.total() for t in tables.values())


@cash.cache
def held_by_my_object(holder):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return holder.table.total()


@cash.cache
def a_function_handed_over(rows, total):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return total(rows)


@cash.cache
def a_bound_method_handed_over(total):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return total()


@cash.cache
def a_class_handed_over(kind, rows):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return kind(rows).total()


class Report:
    def __init__(self, table):
        self.table = table

    @cash.cache
    def total(self):
        print("[RUN]", file=sys.stderr)  # @cash:assume-safe
        return self.table.total()


@cash.cache
def plain(x):
    print("[RUN]", file=sys.stderr)  # @cash:assume-safe
    return x * 2


table = carrier.make([1, 2, 3])
kind = type(table)
which = sys.argv[1]
if which == "by_argument":
    print(by_argument(table))
elif which == "in_a_list":
    print(in_a_list([table, table]))
elif which == "in_a_dict":
    print(in_a_dict({"a": table}))
elif which == "held_by_my_object":
    print(held_by_my_object(carrier.Holder(table)))
elif which == "a_function_handed_over":
    print(a_function_handed_over([1, 2, 3], carrier.total_function()))
elif which == "a_partial_handed_over":
    print(a_function_handed_over([1, 2, 3], functools.partial(carrier.total_function())))
elif which == "a_bound_method_handed_over":
    print(a_bound_method_handed_over(table.total))
elif which == "a_class_handed_over":
    print(a_class_handed_over(kind, [1, 2, 3]))
elif which == "self":
    print(Report(table).total())
elif which == "plain":
    print(plain(3))
"""

CARRIER = """
import fakelib


class Holder:
    def __init__(self, table):
        self.table = table


def make(rows):
    return fakelib.Table(rows)


def total_function():
    return fakelib.total_of
"""


def _install(folder, version, factor):
    shutil.rmtree(folder, ignore_errors=True)
    package = folder / "fakelib"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            class Table:
                def __init__(self, rows):
                    self.rows = rows

                def total(self):
                    return sum(self.rows) * {factor}

            def total_of(rows):
                return sum(rows) * {factor}
            """
        ),
        encoding="utf-8",
    )
    info = folder / f"fakelib-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: fakelib\nVersion: {version}\n", encoding="utf-8")
    (info / "top_level.txt").write_text("fakelib\n", encoding="utf-8")
    (info / "RECORD").write_text(
        f"fakelib/__init__.py,,\nfakelib-{version}.dist-info/RECORD,,\nfakelib-{version}.dist-info/METADATA,,\n",
        encoding="utf-8",
    )


def _run(proj, folder, which):
    p = run_python("job.py", which, cwd=proj, env={"PYTHONPATH": os.pathsep.join([CASH_SRC, str(folder)])})
    assert p.returncode == 0, p.stderr
    return p.stdout.strip(), "[RUN]" in p.stderr


@pytest.fixture
def proj(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text('[project]\nname = "proj"\n', encoding="utf-8")
    (proj / "job.py").write_text(JOB, encoding="utf-8")
    (proj / "carrier.py").write_text(CARRIER, encoding="utf-8")
    return proj


@pytest.mark.parametrize(
    ("which", "before", "after"),
    [
        ("by_argument", "6", "60"),
        ("in_a_list", "12", "120"),
        ("in_a_dict", "6", "60"),
        ("held_by_my_object", "6", "60"),
        ("a_function_handed_over", "6", "60"),
        ("a_partial_handed_over", "6", "60"),
        ("a_bound_method_handed_over", "6", "60"),
        ("a_class_handed_over", "6", "60"),
        ("self", "6", "60"),
    ],
)
def test_an_upgrade_recomputes_a_call_that_took_the_library_s_object(proj, tmp_path, which, before, after):
    lib = tmp_path / "site"
    _install(lib, "1.0", 1)
    assert _run(proj, lib, which) == (before, True)
    assert _run(proj, lib, which) == (before, False), "an unchanged library recomputed"
    _install(lib, "2.0", 10)
    assert _run(proj, lib, which) == (after, True), "the upgrade served the old version's result"
    assert _run(proj, lib, which) == (after, False)


def test_a_call_that_takes_no_library_object_keeps_its_entry(proj, tmp_path):
    lib = tmp_path / "site"
    _install(lib, "1.0", 1)
    assert _run(proj, lib, "plain") == ("6", True)
    _install(lib, "2.0", 10)
    assert _run(proj, lib, "plain") == ("6", False), "an upgrade recomputed a call that never met the library"


def _names(found):
    return {name for name, _version in found}


def test_the_library_of_an_argument_s_class_counts_with_what_it_requires():
    found = held_library_versions([pd.DataFrame], None)
    assert {"pandas", "numpy"} <= _names(found)
    assert ("numpy", np.__version__) in [(n, v.split(" files ")[0]) for n, v in found]


def test_a_library_function_counts_and_the_user_s_and_python_s_own_do_not():
    def mine(x):
        return x

    assert "numpy" in _names(held_library_versions([np.mean], None))
    assert "numpy" in _names(held_library_versions([np.ndarray], None))
    assert held_library_versions([mine, dict, len, os.path.join, type(mine)], None) == []
    assert held_library_versions([pd.DataFrame], "pandas") == []


def test_explain_predicts_the_key_of_a_call_that_took_a_frame(disk_cash):
    @disk_cash.cache
    def summarise(df):
        return df.sum().sum()

    frame = pd.DataFrame({"a": [1, 2]})
    summarise(frame)
    assert summarise.explain(frame).would_hit


def test_a_miss_after_an_upgrade_names_the_library(disk_cash, monkeypatch):
    from cash.decorator import library_versions

    @disk_cash.cache
    def summarise(df):
        return int(df.sum().sum())

    frame = pd.DataFrame({"a": [1, 2]})
    summarise(frame)
    real = library_versions._version
    monkeypatch.setattr(library_versions, "_version", lambda dist: "999.0" if dist == "pandas" else real(dist))
    library_versions.forget()
    try:
        summarise(pd.DataFrame({"a": [1, 2]}))
        info = summarise.cache_info()
        assert info["misses"] == 2
        assert any("pandas" in reason for reason in info["changed"]), info["changed"]
    finally:
        monkeypatch.undo()
        library_versions.forget()


def test_the_versions_are_looked_up_once_not_on_every_call(disk_cash, monkeypatch):
    """Work count: after the first call, a call with a frame reads no
    distribution again and decides nothing about the class again."""
    from cash.decorator import library_versions

    @disk_cash.cache
    def summarise(df, k):
        return int(df.sum().sum()) * k

    frame = pd.DataFrame({"a": [1, 2]})
    summarise(frame, 0)
    asked: list = []
    real_versions, real_top = library_versions._versions_of, library_versions._library_top

    def versions_of(tops):
        if tops:  # the function's own code names none, asked on every key
            asked.append(tops)
        return real_versions(tops)

    monkeypatch.setattr(library_versions, "_versions_of", versions_of)
    monkeypatch.setattr(library_versions, "_library_top", lambda *a: asked.append(a) or real_top(*a))
    for k in range(1, 6):
        summarise(frame, k)
    assert asked == []
    with hashed_bytes() as hashed:
        summarise(frame, 6)
    assert hashed.bytes < 8192
