"""A script in another folder reuses the library map built for the installs.

The saved map is kept per ``sys.path`` (`library_versions._slot`), and a
script's own folder, or ``python -c``'s working folder, is on it: every new
folder built the whole map again, 0.15 s for 150 distributions. Each
``sys.path`` entry names the distributions in its own ``*.dist-info`` and
``*.egg-info`` alone, so a folder holding none names none: the map is now
kept without such folders too (`_shared_fingerprint`), and a process whose
path differs only by them takes that one. A folder that does hold metadata,
or the same folders in another order (which install wins), is another map.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.decorator import library_versions


def _dist(root, name, version, record):
    info = root / f"{name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    (info / "RECORD").write_text("".join(f"{line},,\n" for line in record), encoding="utf-8")


def _package(root, name):
    (root / name).mkdir()
    (root / name / "__init__.py").write_text("", encoding="utf-8")


@pytest.fixture
def site(tmp_path, monkeypatch):
    folder = tmp_path / "site"
    folder.mkdir()
    _package(folder, "realpkg")
    _dist(folder, "fake_real", "1.0", ["realpkg/__init__.py"])
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setattr(library_versions, "_map_path", lambda: str(tmp_path / "saved" / "_library_map.json"))
    library_versions.forget()
    yield folder
    library_versions.forget()


def _scans(monkeypatch):
    calls = []
    real = library_versions._packages_distributions
    monkeypatch.setattr(library_versions, "_packages_distributions", lambda: calls.append(1) or real())
    return calls


def _in_folder(monkeypatch, folder, *others):
    """A new process started from *folder*: it is first on ``sys.path``."""
    keep = [p for p in sys.path if p not in {str(f) for f in others}]
    monkeypatch.setattr(sys, "path", [str(folder), *keep])
    library_versions.forget()


def test_scripts_in_other_folders_build_the_map_once(site, tmp_path, monkeypatch):
    folders = []
    for name in ("one", "two", "three"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "script.py").write_text("print(1)\n", encoding="utf-8")
        folders.append(folder)
    scans = _scans(monkeypatch)
    for folder in folders:
        _in_folder(monkeypatch, folder, *folders)
        assert library_versions._dists_for("realpkg") == ["fake_real"]
    assert len(scans) == 1


def test_what_a_shared_map_finds_is_what_a_fresh_build_finds(site, tmp_path, monkeypatch):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    _in_folder(monkeypatch, one, one, two)
    built = {top: library_versions._dists_for(top) for top in ("realpkg", "ghostpkg")}
    versions = library_versions._version("fake_real")
    _in_folder(monkeypatch, two, one, two)
    assert {top: library_versions._dists_for(top) for top in ("realpkg", "ghostpkg")} == built
    assert library_versions._version("fake_real") == versions == "1.0"


def test_a_folder_holding_metadata_is_never_left_out(site, tmp_path, monkeypatch):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    _package(two, "newpkg")
    _dist(two, "fake_new", "1.0", ["newpkg/__init__.py"])
    _in_folder(monkeypatch, one, one, two)
    assert library_versions._dists_for("newpkg") == []
    _in_folder(monkeypatch, two, one, two)
    assert library_versions._dists_for("newpkg") == ["fake_new"]
    _in_folder(monkeypatch, one, one, two)
    assert library_versions._dists_for("newpkg") == []


def test_the_same_installs_in_another_order_are_another_map(tmp_path, monkeypatch):
    """Which of two installs of a distribution a process imports is the one
    earlier on ``sys.path``: its version is the one folded."""
    monkeypatch.setattr(library_versions, "_map_path", lambda: str(tmp_path / "saved" / "_library_map.json"))
    a, b, work = tmp_path / "a", tmp_path / "b", tmp_path / "work"
    for folder, version in ((a, "1.0"), (b, "2.0")):
        folder.mkdir()
        _package(folder, "twicepkg")
        _dist(folder, "fake_twice", version, ["twicepkg/__init__.py"])
    work.mkdir()
    base = list(sys.path)
    try:
        for order, version in (((a, b), "1.0"), ((b, a), "2.0"), ((a, b), "1.0")):
            sys.path[:] = [str(work), *map(str, order), *base]
            library_versions.forget()
            assert library_versions._dists_for("twicepkg") == ["fake_twice", "fake_twice"]
            assert library_versions._version("fake_twice") == version
            assert library_versions._closure("fake_twice")
            assert library_versions._version("fake_twice") == version
    finally:
        sys.path[:] = base
        library_versions.forget()


def test_a_saved_map_answers_without_importing_importlib_metadata(site, tmp_path, monkeypatch):
    """Checking that a module's files are there is a read of ``RECORD`` and
    a stat: ``importlib.metadata`` cost 15 ms to import on Windows."""
    (site / "realpkg" / "sub").mkdir()
    (site / "realpkg" / "sub" / "x.py").write_text("", encoding="utf-8")
    _package(site, "oddpkg")
    _dist(site, "fake_odd", "1.0", ["oddpkg//__init__.py", "../../bin/tool"])
    expected = {top: library_versions._dists_for(top) for top in ("realpkg", "oddpkg", "ghostpkg")}
    assert expected == {"realpkg": ["fake_real"], "oddpkg": ["fake_odd"], "ghostpkg": []}

    class NotImported:
        def __getattr__(self, name):
            raise AssertionError(f"importlib.metadata.{name} read")

    monkeypatch.setattr(library_versions, "importlib_metadata", NotImported())
    library_versions.forget()
    assert {top: library_versions._dists_for(top) for top in expected} == expected


def test_a_metadata_folder_is_checked_from_one_listing(site, monkeypatch):
    """The fingerprint stats each metadata file the map reads from the
    listing of its folder, not one by one (six for each of 150 folders)."""
    library_versions._dists_for("realpkg")
    statted = []
    real = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: statted.append(os.fspath(p)) or real(p, *a, **k))
    printed = library_versions._folder_print(str(site))
    monkeypatch.undo()
    inside = [p for p in statted if ".dist-info" + os.sep in p]
    assert not inside, inside
    (entry,) = [e for e in printed[1] if e[0].endswith(".dist-info")]
    by_name = {row[0]: row[1:] for row in entry[-1]}
    record = os.stat(site / "fake_real-1.0.dist-info" / "RECORD")
    assert by_name["RECORD"][:3] == [record.st_mtime_ns, record.st_ctime_ns, record.st_size]
    assert by_name["top_level.txt"] == []
