"""The ``.pyc`` an import writes is the interpreter's cache, not a statement's
output.

The first import of a module whose bytecode is not cached yet writes its
``.pyc`` (``importlib`` writes ``m.cpython-313.pyc.<number>`` and renames it
into place), in a ``__pycache__`` next to the source or under
``PYTHONPYCACHEPREFIX``. CI's interpreters keep their standard library
writable and uncompiled, so ``from xml.dom import minidom`` in a statement
was "seen writing" ``expatbuilder.cpython-313.pyc``, and the statement ran
every time. A hit that skips such a write leaves the module to be compiled
again, nothing else.

Under ``PYTHONPYCACHEPREFIX`` the ``.pyc`` path holds the source folder's
whole path again below the prefix, so both folders are short ones of their
own: under pytest's ``tmp_path`` it came to about 280 characters on a
Windows machine without long paths turned on, where ``importlib`` gives up
writing past 260 without a word.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from cash.notebook.write_observer import kept_writes, observe_writes


@pytest.fixture
def short_dirs():
    """Folders with short paths (see the module docstring), removed after."""
    made: list[Path] = []

    def make() -> Path:
        d = Path(tempfile.mkdtemp(prefix="bc"))
        made.append(d)
        return d

    yield make
    for d in made:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def source_dir(short_dirs):
    return short_dirs()


@pytest.fixture
def fresh_module(source_dir, monkeypatch):
    """A module never imported, whose bytecode an import writes."""
    name = f"_bc_{uuid.uuid4().hex[:8]}"
    (source_dir / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(source_dir))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    importlib.invalidate_caches()
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize("prefix", [False, True], ids=["__pycache__", "PYTHONPYCACHEPREFIX"])
def test_an_import_writing_its_bytecode_writes_nothing_kept(fresh_module, source_dir, short_dirs, monkeypatch, prefix):
    bytecode = short_dirs() if prefix else source_dir / "__pycache__"
    if prefix:
        monkeypatch.setattr(sys, "pycache_prefix", str(bytecode))
    with observe_writes() as written:
        importlib.import_module(fresh_module)
    written_pyc = list(bytecode.rglob("*.pyc"))
    assert written_pyc, f"the import wrote no bytecode under {bytecode}: proves nothing"
    assert kept_writes(written, None, written.appended) == []


def test_a_file_written_beside_the_import_is_kept(fresh_module, source_dir):
    """Control: the statement's own file is still its output."""
    out = source_dir / "out.txt"
    with observe_writes() as written:
        importlib.import_module(fresh_module)
        out.write_text("x", encoding="utf-8")
    assert kept_writes(written, None, written.appended) == [str(out)]
