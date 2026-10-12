"""The file part of a statement's lineage moves with a same-size edit that
puts the mtime back (``cp -p``, ``rsync -t``, unzip).

It folded the size and the mtime alone, so a value read through a helper
that also writes a file -- never stored, so no entry checks the file by
content -- kept its lineage, and the cells below served their old results.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook.statement.file_deps import compute_file_hash_component


@pytest.mark.skipif(sys.platform == "win32", reason="on Windows ctime is the creation time: a documented limit")
def test_the_component_changes(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"v": 1}', encoding="utf-8", newline="\n")
    before = compute_file_hash_component({str(cfg)})
    st = os.stat(cfg)
    cfg.write_text('{"v": 7}', encoding="utf-8", newline="\n")
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(cfg).st_mtime_ns == st.st_mtime_ns
    assert compute_file_hash_component({str(cfg)}) != before


def test_an_untouched_file_keeps_it(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"v": 1}', encoding="utf-8", newline="\n")
    assert compute_file_hash_component({str(cfg)}) == compute_file_hash_component({str(cfg)})
