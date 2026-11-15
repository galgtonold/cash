"""A stored Cython analysis holds under another import path only while
every lookup that depends on it answers the same (`cython_digest._lookups_hold`).

It was keyed by the whole ``sys.path``, so a driver script in another
folder replaced the record and every process parsed every module again.
Now the cimports it looked up are looked up again under the new path: an
entry that adds nothing keeps the record, one that shadows a cimported
``.pxd`` with another file does not.
"""

from __future__ import annotations

import sys

from cash import cython_digest


def _record(importer, package_root, includes):
    inputs = cython_digest._Inputs()
    cython_digest._RECORDING.append(inputs)
    try:
        found = cython_digest._resolve_cimport("mylib", 0, str(importer), str(package_root), str(includes))
    finally:
        cython_digest._RECORDING.pop()
    return found, {
        "import_path": cython_digest._import_path(),
        "lookups": [[list(k), answer] for k, answer in inputs.lookups.items()],
    }


def test_a_cimport_another_entry_shadows_is_looked_up_again(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    importer = project / "mod.pyx"
    importer.write_text("cimport mylib\n", encoding="utf-8")
    includes = tmp_path / "includes"
    includes.mkdir()
    first, second, empty = (tmp_path / n for n in ("first", "second", "empty"))
    for folder in (first, second, empty):
        folder.mkdir()
    (first / "mylib.pxd").write_text("cdef int f(int x)\n", encoding="utf-8")
    (second / "mylib.pxd").write_text("cdef int f(int x, int y)\n", encoding="utf-8")

    monkeypatch.setattr(sys, "path", [str(first)])
    found, stored = _record(importer, project, includes)
    assert found == [cython_digest._norm(str(first / "mylib.pxd"))]
    assert cython_digest._lookups_hold(stored)

    monkeypatch.setattr(sys, "path", [str(empty), str(first)])
    assert stored["import_path"] != cython_digest._import_path()
    assert cython_digest._lookups_hold(stored), "an entry that adds nothing dropped the record"

    monkeypatch.setattr(sys, "path", [str(second), str(first)])
    assert not cython_digest._lookups_hold(stored), "a cimport resolving to another .pxd kept the record"

    monkeypatch.setattr(sys, "path", [str(empty)])
    assert not cython_digest._lookups_hold(stored), "a cimport found nowhere now kept the record"


def test_a_record_without_lookups_is_not_used_under_another_import_path(monkeypatch):
    stored = {"import_path": [["/somewhere/else"], "/"]}
    assert not cython_digest._lookups_hold(stored)
