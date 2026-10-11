"""Running a cell below an edit that keeps the file's size and mtime re-reads it.

``cp -p``, ``rsync -t`` and an archive with fixed timestamps rewrite a file and
put its mtime back. The upstream simulation's quick scan compared only the
mtime, with a 10 ms tolerance, and the lineage folds only size and mtime, so
running only a cell below served the value read from the old file. Run All was
right: the runtime checks the content.
"""

from __future__ import annotations

import os
import sys

import pytest

from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="on Windows ctime is the creation time: such an edit is a documented limit"
)


@pytest.fixture(autouse=True)
def _notebook(monkeypatch, cash_instance):
    """A notebook is found, so the upstream check runs, and every statement is
    stored however cheap."""
    monkeypatch.setattr("cash.notebook.upstream.checker.get_notebook_path", lambda: "/fake/notebook.ipynb")
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0


def _rewrite_keeping_stat(path, text):
    st = os.stat(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_a_cell_below_sees_an_edit_that_keeps_size_and_mtime(cash_magics, tmp_path):
    data = tmp_path / "a.txt"
    data.write_text("1", encoding="utf-8", newline="\n")
    os.utime(data, (1e9, 1e9))
    path = str(data).replace("\\", "/")
    cells = [
        f"text = open('{path}').read()",
        "t = int(text) * 10",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == 10

    _rewrite_keeping_stat(data, "2")
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert ns["text"] == "2"
    assert ns["t"] == 20


def test_an_unchanged_file_is_not_re_read(cash_magics, tmp_path):
    """Control: with the file left alone the cell above is not run again."""
    data = tmp_path / "a.txt"
    data.write_text("1", encoding="utf-8", newline="\n")
    path = str(data).replace("\\", "/")
    cells = [
        f"text = open('{path}').read()\nreads = reads + 1 if 'reads' in dir() else 1",
        "t = int(text) * 10",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    reads = ns["reads"]
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert ns["reads"] == reads and ns["t"] == 10


def test_a_link_re_pointed_to_a_same_size_file_is_seen_below(cash_magics, tmp_path):
    """Two releases unpacked together: same size, mtimes within 10 ms."""
    v1, v2 = tmp_path / "v1.txt", tmp_path / "v2.txt"
    v1.write_text("1", encoding="utf-8", newline="\n")
    v2.write_text("2", encoding="utf-8", newline="\n")
    os.utime(v1, ns=(10**18, 10**18))
    os.utime(v2, ns=(10**18 + 5_000_000, 10**18 + 5_000_000))
    cur = tmp_path / "cur.txt"
    os.symlink(v1, cur)
    path = str(cur).replace("\\", "/")
    cells = [f"text = open('{path}').read()", "t = int(text) * 10"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == 10

    cur.unlink()
    os.symlink(v2, cur)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert ns["t"] == 20
