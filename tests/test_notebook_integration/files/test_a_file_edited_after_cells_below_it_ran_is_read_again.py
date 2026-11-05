"""A file edited after the cells below its reader ran is read again.

Before each cell, the check of the cells above holds its verdict on their
files while no watched folder reports a change (`run_memo.stat_keys_as_recorded`):
the cells below a read of many files ran several times with that verdict
held before the edit. The edit must still reach the reader and every cell
that uses what it read.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


def test_an_edit_after_held_checks_is_read_again(nb_runner, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(40):
        (folder / f"d{i:02d}.txt").write_text(f"doc {i:02d}", encoding="utf-8")
    path = str(folder).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import glob, time",
            f"texts = [open(p).read() for p in sorted(glob.glob('{path}/*.txt'))]\ntime.sleep(0.3)",
            "n = len(texts)",
            "first = texts[7]",
            "print(first, n)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert "doc 07 40" in nb_runner.get_output(5)

    (folder / "d07.txt").write_text("DOC 07", encoding="utf-8")  # same size
    nb_runner.run_all()
    assert "DOC 07 40" in nb_runner.get_output(5), nb_runner.get_output(5)
    assert nb_runner.peek("first") == repr("DOC 07")


def test_an_edit_reaches_a_cell_run_on_its_own(nb_runner, tmp_path):
    """Running only the last cell after the edit: the check before it finds
    the reader's file changed and brings what it reads up to date."""
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(40):
        (folder / f"d{i:02d}.txt").write_text(f"doc {i:02d}", encoding="utf-8")
    path = str(folder).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import glob, time",
            f"texts = [open(p).read() for p in sorted(glob.glob('{path}/*.txt'))]\ntime.sleep(0.3)",
            "first = texts[7]",
            "print(first)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cells([4])
    nb_runner.run_cells([4])
    assert "doc 07" in nb_runner.get_output(4)

    (folder / "d07.txt").write_text("DOC 07", encoding="utf-8")
    nb_runner.run_cells([4])
    assert "DOC 07" in nb_runner.get_output(4), nb_runner.get_output(4)
