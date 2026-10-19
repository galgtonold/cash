"""The library map is built once per set of installs, not once per process.

Which installed distribution provides each top-level module
(`library_versions._packages_distributions`) is read from every
distribution's metadata and file list: 0.15 s for 150 of them, paid on the
first key of every process, so each worker of a parallel call (p25's four
joblib workers) and each run of a script paid it again. The map is now saved
with what it was read from, the metadata folders on ``sys.path`` and their
stats (`_install_fingerprint`), and an earlier process's map is used while
those are the same. An install, an uninstall or an upgrade changes them, and
the map is built again; so the map, and every key, is what a fresh build
gives.
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import pytest

from cash.decorator import library_versions


def _dist(root, name, version, record, top_level=None):
    info = root / f"{name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    (info / "RECORD").write_text("".join(f"{line},,\n" for line in record), encoding="utf-8")
    if top_level is not None:
        (info / "top_level.txt").write_text(top_level, encoding="utf-8")
    return info


def _package(root, name):
    (root / name).mkdir()
    (root / name / "__init__.py").write_text("", encoding="utf-8")


@pytest.fixture
def site(tmp_path, monkeypatch):
    folder = tmp_path / "site"
    folder.mkdir()
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setattr(library_versions, "_map_path", lambda: str(tmp_path / "saved" / "_library_map.json"))
    library_versions.forget()
    yield folder
    library_versions.forget()


def _new_process():
    """What a new process starts with: no map in memory."""
    library_versions.forget()


def _lookups():
    return {top: library_versions._dists_for(top) for top in ("realpkg", "toppkg", "ghostpkg", "newpkg")}


def _scans(monkeypatch):
    """Count the full reads of every installed distribution."""
    calls = []
    real = library_versions._packages_distributions
    monkeypatch.setattr(library_versions, "_packages_distributions", lambda: calls.append(1) or real())
    return calls


def _fill(site):
    _package(site, "realpkg")
    _package(site, "toppkg")
    _dist(site, "fake_real", "1.0", ["realpkg/__init__.py"])
    _dist(site, "fake_top", "1.0", [], top_level="toppkg\n")
    _dist(site, "fake_ghost", "1.0", ["ghostpkg/__init__.py"])


def test_a_new_process_uses_the_saved_map_and_finds_what_a_fresh_build_finds(site, monkeypatch):
    _fill(site)
    scans = _scans(monkeypatch)
    first = _lookups()
    assert first == {"realpkg": ["fake_real"], "toppkg": ["fake_top"], "ghostpkg": [], "newpkg": []}
    assert len(scans) == 1
    for _ in range(3):
        _new_process()
        assert _lookups() == first
    assert len(scans) == 1  # built once, read three times


@pytest.mark.parametrize("change", ["install", "uninstall", "upgrade"])
def test_a_change_to_the_installs_builds_the_map_again(site, monkeypatch, change):
    _fill(site)
    _lookups()
    _new_process()
    if change == "install":
        _package(site, "newpkg")
        _dist(site, "fake_new", "1.0", ["newpkg/__init__.py"])
        expected = ["fake_new"]
    elif change == "uninstall":
        shutil.rmtree(site / "fake_real-1.0.dist-info")
        shutil.rmtree(site / "realpkg")
        expected = []
    else:
        shutil.rmtree(site / "fake_real-1.0.dist-info")
        _dist(site, "fake_real2", "2.0", ["realpkg/__init__.py"])
        expected = ["fake_real2"]
    scans = _scans(monkeypatch)
    found = _lookups()
    assert len(scans) == 1
    assert found["newpkg" if change == "install" else "realpkg"] == expected
    _new_process()
    assert _lookups() == found


def test_a_saved_map_still_checks_that_the_listed_files_are_there(site, monkeypatch):
    """A package removed without its metadata names no distribution, as in a fresh build."""
    _fill(site)
    _lookups()
    shutil.rmtree(site / "realpkg")
    _new_process()
    scans = _scans(monkeypatch)
    assert library_versions._dists_for("realpkg") == []
    assert not scans


def test_a_changed_search_path_builds_its_own_map(site, tmp_path, monkeypatch):
    _fill(site)
    _lookups()
    other = tmp_path / "other"
    other.mkdir()
    _package(other, "newpkg")
    _dist(other, "fake_new", "1.0", ["newpkg/__init__.py"])
    monkeypatch.syspath_prepend(str(other))
    _new_process()
    assert library_versions._dists_for("newpkg") == ["fake_new"]


def test_an_unreadable_saved_map_is_built_again(site, monkeypatch):
    _fill(site)
    first = _lookups()
    with open(library_versions._map_path(), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    _new_process()
    scans = _scans(monkeypatch)
    assert _lookups() == first
    assert len(scans) == 1


def test_a_finder_that_names_distributions_of_its_own_is_never_skipped(site, monkeypatch):
    """Another finder on ``sys.meta_path`` may name distributions no folder shows."""

    class Finder:
        @staticmethod
        def find_spec(*a, **k):
            return None

        @staticmethod
        def find_distributions(*a, **k):
            return iter(())

    _fill(site)
    _lookups()
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, Finder])
    _new_process()
    scans = _scans(monkeypatch)
    _lookups()
    assert len(scans) == 1


def _reaches_numpy(x):
    return np.mean(x)


def test_the_versions_folded_into_a_key_are_the_same_from_a_saved_map(site):
    fresh = library_versions.library_versions([_reaches_numpy], None)
    assert any(name == "numpy" for name, _ in fresh)
    _new_process()
    assert library_versions.library_versions([_reaches_numpy], None) == fresh
