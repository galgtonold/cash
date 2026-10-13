"""A file read by a relative path is one file, named one way.

A read by a relative path is recorded twice on purpose: the absolute path it
resolved to, and the path as written, re-resolved at each check so a chdir is
seen. ``cash inspect`` listed both (``reads: /abs/prices.csv, prices.csv``),
and a miss reason named the changed file by whichever spelling came first:
one function's ``data/customers.csv``, another's ``/tmp/.../data/stores.csv``.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from cash.decorator.cache_metadata import CacheMetadata
from cash.decorator.explain import describe_file_deps, describe_stale_files
from tests._scripts import run_python

pytestmark = pytest.mark.core

_JOB = textwrap.dedent("""
    import cash

    @cash.cache(assume_safe=True)
    def load():
        with open("data/prices.csv") as f:
            return f.read()

    print(len(load()))
""")


def _run_job(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "prices.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "job.py").write_text(_JOB, encoding="utf-8")
    run_python("job.py", cwd=tmp_path)


def test_inspect_lists_the_file_once(tmp_path):
    _run_job(tmp_path)
    listing = run_python("-m", "cash", "inspect", tmp_path / ".cash", "--function", "job.load", cwd=tmp_path).stdout
    reads = [line for line in listing.splitlines() if "reads:" in line]
    assert len(reads) == 1, listing
    assert reads[0].count("prices.csv") == 1, listing
    assert str(tmp_path.resolve() / "data" / "prices.csv") in reads[0], listing


def _recorded(tmp_path) -> dict:
    """What a call reading ``data/prices.csv`` recorded, the relative spelling first."""
    _run_job(tmp_path)
    (entry,) = _entries(tmp_path)
    recorded = entry["auto_file_deps"]
    assert len(recorded) == 2, recorded  # the absolute path and the path as written
    return dict(sorted(recorded.items(), key=lambda item: os.path.isabs(item[0])))


def _entries(tmp_path):
    from cash.__main__ import _store

    return [s.metadata for s in _store(tmp_path / ".cash").entries()]


def test_the_two_records_are_paired_without_the_cwd(tmp_path, monkeypatch):
    from cash.decorator.explain import relative_twins

    recorded = _recorded(tmp_path)
    monkeypatch.chdir(tmp_path.parent)  # `cash inspect` runs from anywhere
    absolute = str(tmp_path.resolve() / "data" / "prices.csv")
    assert relative_twins(recorded) == {next(p for p in recorded if p != absolute): absolute}
    assert list(describe_file_deps(recorded)) == [absolute]


def test_a_miss_names_the_file_by_its_absolute_path(tmp_path, monkeypatch):
    recorded = _recorded(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "prices.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    said = describe_stale_files(CacheMetadata(auto_file_deps=recorded))
    assert said.startswith(str(tmp_path.resolve() / "data" / "prices.csv")), said
    assert "more" not in said, said
