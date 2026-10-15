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
"""

from __future__ import annotations

import importlib
import sys

import pytest

from cash.notebook.write_observer import kept_writes, observe_writes


@pytest.fixture
def fresh_module(tmp_path, monkeypatch):
    """A module never imported, whose bytecode an import writes."""
    name = f"_bytecode_written_{abs(hash(tmp_path))}"
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    importlib.invalidate_caches()
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize("prefix", [False, True], ids=["__pycache__", "PYTHONPYCACHEPREFIX"])
def test_an_import_writing_its_bytecode_writes_nothing_kept(fresh_module, tmp_path, monkeypatch, prefix):
    if prefix:
        monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "bytecode"))
    with observe_writes() as written:
        importlib.import_module(fresh_module)
    written_pyc = list((tmp_path / "bytecode" if prefix else tmp_path / "__pycache__").rglob("*.pyc"))
    assert written_pyc, "the import wrote no bytecode: proves nothing"
    assert kept_writes(written, None, written.appended) == []


def test_a_file_written_beside_the_import_is_kept(fresh_module, tmp_path):
    """Control: the statement's own file is still its output."""
    out = tmp_path / "out.txt"
    with observe_writes() as written:
        importlib.import_module(fresh_module)
        out.write_text("x", encoding="utf-8")
    assert kept_writes(written, None, written.appended) == [str(out)]
