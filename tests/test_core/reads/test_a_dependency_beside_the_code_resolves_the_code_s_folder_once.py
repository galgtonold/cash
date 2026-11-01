"""Checking many dependencies beside the same code resolves that code's folder
once. Each check resolved the module's file again: a joblib worker checking
48,000 cached dependencies spent two of its five seconds in ``realpath``."""

from __future__ import annotations

import os
import sys
import types

from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import dep_path_for_this_process

_NAME = "fxroot_once_pkg.core"


def test_the_code_folder_is_resolved_once_for_many_checks(tmp_path, monkeypatch):
    pkg = tmp_path / "fxroot_once_pkg"
    pkg.mkdir()
    mod = types.ModuleType(_NAME)
    mod.__file__ = str(pkg / "core.py")
    monkeypatch.setitem(sys.modules, _NAME, mod)
    calls = []
    realpath = os.path.realpath

    def counted(path, *args, **kwargs):
        calls.append(path)
        return realpath(path, *args, **kwargs)

    monkeypatch.setattr(file_dep_snapshot.os.path, "realpath", counted)
    recorded = {"code_rel": "data/a.csv", "code_mod": _NAME}
    found = {dep_path_for_this_process("/elsewhere/a.csv", recorded) for _ in range(1000)}
    assert len(found) == 1
    assert os.path.normcase(found.pop()).endswith(os.path.normcase(os.path.join("fxroot_once_pkg", "data", "a.csv")))
    assert calls.count(mod.__file__) == 1


def test_a_module_loaded_from_another_file_resolves_its_own_folder(tmp_path, monkeypatch):
    recorded = {"code_rel": "data/a.csv", "code_mod": _NAME}
    paths = []
    for copy in ("one", "two"):
        pkg = tmp_path / copy / "fxroot_once_pkg"
        pkg.mkdir(parents=True)
        mod = types.ModuleType(_NAME)
        mod.__file__ = str(pkg / "core.py")
        monkeypatch.setitem(sys.modules, _NAME, mod)
        paths.append(dep_path_for_this_process("/elsewhere/a.csv", recorded))
    assert paths[0] != paths[1]
    assert os.path.normcase(os.sep + "two" + os.sep) in os.path.normcase(paths[1])
