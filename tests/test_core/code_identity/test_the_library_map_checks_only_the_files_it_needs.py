"""Which distribution provides a top-level module is read from each one's file
list (`library_versions._packages_distributions`). ``dist.files`` drops the
files not on disk by a stat of every one of them: 20,000 in an ordinary
environment, half a second in every process, each worker of a parallel call
included. The list is now read as written and a file checked only for the
names a cached function reaches; a name whose files are all gone still names
no distribution.
"""

from __future__ import annotations

import os

import pytest

from cash.decorator import library_versions


def _dist(root, name, record, top_level=None):
    info = root / f"{name}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8")
    (info / "RECORD").write_text("".join(f"{line},,\n" for line in record), encoding="utf-8")
    if top_level is not None:
        (info / "top_level.txt").write_text(top_level, encoding="utf-8")


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    library_versions.forget()
    yield tmp_path
    library_versions.forget()


def test_a_listed_file_counts_only_if_it_is_there(site):
    (site / "realpkg").mkdir()
    (site / "realpkg" / "__init__.py").write_text("", encoding="utf-8")
    (site / "single.py").write_text("", encoding="utf-8")
    _dist(site, "fake_real", ["realpkg/__init__.py", "single.py", "fake_real-1.0.dist-info/RECORD"])
    _dist(site, "fake_ghost", ["ghostpkg/__init__.py", "ghostpkg/core.py"])
    assert library_versions._dists_for("realpkg") == ["fake_real"]
    assert library_versions._dists_for("single") == ["fake_real"]
    assert library_versions._dists_for("ghostpkg") == []


def test_finding_one_name_does_not_stat_every_listed_file(site, monkeypatch):
    (site / "realpkg").mkdir()
    (site / "realpkg" / "__init__.py").write_text("", encoding="utf-8")
    _dist(site, "fake_real", ["realpkg/__init__.py"] + [f"bigpkg/m{i}.py" for i in range(500)])
    statted = []
    real = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: statted.append(os.fspath(p)) or real(p, *a, **k))
    assert library_versions._dists_for("realpkg") == ["fake_real"]
    monkeypatch.undo()
    root = os.path.normcase(str(site))
    mine = [p for p in statted if os.path.normcase(os.path.abspath(p)).startswith(root)]
    # One existence check for the name looked up; none for bigpkg's 500.
    assert not [p for p in mine if "bigpkg" in p], len(mine)
