"""Which installed distributions a library requires, transitively, and their
versions (`library_versions._closure`) are kept with the saved library map:
reading every ``METADATA`` on the way cost 0.2 s in each new process for a
scientific stack. They are kept under the map's fingerprint (every metadata
folder on ``sys.path`` and the stat of the files in it), so an install, an
uninstall or an upgrade of the library or of anything it requires reads
them again, and the versions folded into a key are what a fresh read gives.
"""

from __future__ import annotations

import shutil

import pytest

from cash.decorator import library_versions


def _dist(root, name, version, requires=()):
    info = root / f"{name}-{version}.dist-info"
    info.mkdir()
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    lines += [f"Requires-Dist: {r}" for r in requires]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (info / "RECORD").write_text(f"{name}/__init__.py,,\n" if name == "fakelib" else "", encoding="utf-8")
    return info


@pytest.fixture
def site(tmp_path, monkeypatch):
    folder = tmp_path / "site"
    folder.mkdir()
    (folder / "fakelib").mkdir()
    (folder / "fakelib" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setattr(library_versions, "_map_path", lambda: str(tmp_path / "saved" / "_library_map.json"))
    monkeypatch.setattr(library_versions, "_records_path", lambda: str(tmp_path / "saved" / "_records.json"))
    library_versions.forget()
    yield folder
    library_versions.forget()


def _versions():
    library_versions.forget()  # what a new process starts with
    found = {}
    for dist in library_versions._dists_for("fakelib"):
        for name in library_versions._closure(dist):
            found[name] = library_versions._version(name)
    library_versions._keep_closures()
    return found


def _reads(monkeypatch):
    calls = []
    real = library_versions._requires
    monkeypatch.setattr(library_versions, "_requires", lambda d: calls.append(d) or real(d))
    return calls


def test_a_new_process_reads_no_metadata_for_what_it_kept(site, monkeypatch):
    _dist(site, "fakelib", "1.0", ["fakedep>=1"])
    _dist(site, "fakedep", "1.0")
    first = _versions()
    assert first == {"fakelib": "1.0", "fakedep": "1.0"}
    reads = _reads(monkeypatch)
    assert _versions() == first
    assert reads == []


@pytest.mark.parametrize("change", ["upgrade", "uninstall", "new requirement"])
def test_a_change_to_what_the_library_requires_is_read_again(site, monkeypatch, change):
    _dist(site, "fakelib", "1.0", ["fakedep>=1"])
    _dist(site, "fakedep", "1.0")
    _versions()
    if change == "upgrade":
        shutil.rmtree(site / "fakedep-1.0.dist-info")
        _dist(site, "fakedep", "2.0")
        expected = {"fakelib": "1.0", "fakedep": "2.0"}
    elif change == "uninstall":
        shutil.rmtree(site / "fakedep-1.0.dist-info")
        expected = {"fakelib": "1.0"}
    else:
        shutil.rmtree(site / "fakelib-1.0.dist-info")
        _dist(site, "fakelib", "1.0", ["fakedep>=1", "otherdep"])
        _dist(site, "otherdep", "3.0")
        expected = {"fakelib": "1.0", "fakedep": "1.0", "otherdep": "3.0"}
    assert _versions() == expected
    assert _versions() == expected  # and kept again
