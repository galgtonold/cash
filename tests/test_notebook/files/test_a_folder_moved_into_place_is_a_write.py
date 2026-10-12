"""A helper that publishes a folder atomically -- fills a temporary folder,
then ``tmp.rename(dest)`` -- writes ``dest``.

The files under the temporary folder are gone by the end (taken for
temporary files), and the move's destination was dropped as a folder, a rule
meant for ``O_TMPFILE`` opens. The statement was stored, and the next Run All
left the release folder missing. A folder opened to write (a file with no
name) still leaves nothing.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from cash.notebook.write_observer import kept_writes, observe_writes
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def test_the_destination_is_kept(tmp_path):
    with observe_writes() as written:
        tmp = Path(tempfile.mkdtemp(dir=tmp_path))
        (tmp / "a.txt").write_text("x", encoding="utf-8")
        tmp.rename(tmp_path / "release")
    assert kept_writes(written, None, written.appended) == [os.path.abspath(tmp_path / "release")]


def test_a_file_with_no_name_leaves_nothing(tmp_path):
    with observe_writes() as written, tempfile.TemporaryFile(dir=tmp_path) as fh:
        fh.write(b"x")
    assert kept_writes(written, None, written.appended) == []


def test_the_folder_is_published_on_every_run(cash_magics, tmp_path):
    dest = tmp_path / "release"
    cells = [
        "import gzip, shutil, tempfile, time\nfrom pathlib import Path\n"
        "def publish(parts, dest):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        f"    tmp = Path(tempfile.mkdtemp(dir={str(tmp_path)!r}))\n"
        "    for i, part in enumerate(parts):\n"
        "        f = gzip.open(tmp / f'part{i}.gz', 'wt'); f.write(part); f.close()\n"
        "    tmp.rename(dest)\n"
        "    return len(parts)",
        f"shutil.rmtree({str(dest)!r}, ignore_errors=True)",
        f"n = publish(['a', 'b'], {str(dest)!r})",
    ]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert sorted(p.name for p in dest.iterdir()) == ["part0.gz", "part1.gz"]
