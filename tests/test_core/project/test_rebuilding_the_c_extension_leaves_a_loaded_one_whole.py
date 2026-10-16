"""Rebuilding the C extension leaves the file a running process loaded whole.

An editable install copies the compiled ``_cwatch`` module into ``src/cash/``.
It was copied into the existing file in place, so a process that had the old
module mapped (a kernel left running across ``uv pip install -e .``) read
the new bytes under the old mapping and died of SIGSEGV or SIGBUS. The
module is written to a new file and renamed over the old one, which stays
whole for whoever has it open.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _hook():
    spec = importlib.util.spec_from_file_location("hatch_build", REPO_ROOT / "hatch_build.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    return hook


@pytest.mark.skipif(sys.platform == "win32", reason="Windows refuses to replace a file that is open")
def test_installing_a_rebuilt_module_does_not_write_into_the_old_file(tmp_path):
    built = tmp_path / "build" / "_cwatch.so"
    built.parent.mkdir()
    built.write_bytes(b"new module")
    package = tmp_path / "pkg"
    package.mkdir()
    target = package / "_cwatch.so"
    target.write_bytes(b"old module, as loaded")

    with target.open("rb") as loaded:  # what a running process has mapped
        _hook().install_module(built, target)
        assert loaded.read() == b"old module, as loaded"
    assert target.read_bytes() == b"new module"
    assert [p.name for p in package.iterdir()] == ["_cwatch.so"]


def test_installing_where_nothing_was_built_before_creates_the_module(tmp_path):
    built = tmp_path / "_cwatch.so"
    built.write_bytes(b"module")
    built.chmod(0o755)
    target = tmp_path / "pkg" / "_cwatch.so"
    target.parent.mkdir()
    _hook().install_module(built, target)
    assert target.read_bytes() == b"module"
    if sys.platform != "win32":
        assert target.stat().st_mode & 0o777 == 0o755
    assert [p.name for p in target.parent.iterdir()] == ["_cwatch.so"]
