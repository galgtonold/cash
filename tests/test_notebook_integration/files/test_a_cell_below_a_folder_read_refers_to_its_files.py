"""A cell below a folder read refers to the read's records of its files.

Its entry keeps a reference to the entry of the read, not a copy of a
record per file (``cash.notebook.file_records``), and the files are read by
relative path, whose second spelling is folded into its absolute twin's
record. After a restart the cell is served while the files are as read, and
runs again once one of them changed.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(300)]

FILES = 30


def test_served_after_a_restart_and_run_again_after_an_edit(nb_runner, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(FILES):
        (docs / f"d{i:02d}.txt").write_text("x" * (i + 1), encoding="utf-8")
    nb_runner.create_notebook(
        [
            "%cash_badge print\nimport glob, time",
            "X = [open(p).read() for p in sorted(glob.glob('docs/*.txt'))]",
            "n = (time.sleep(0.3), sum(len(x) for x in X))[1]\nprint('N', n)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    total = FILES * (FILES + 1) // 2
    assert f"N {total}" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert f"N {total}" in out and shows_cached(out), nb_runner.get_raw_output(2) + nb_runner.get_raw_output(3)

    nb_runner.restart()
    nb_runner._init_cash()
    (docs / "d05.txt").write_text("x" * 100, encoding="utf-8")
    nb_runner.run_all()
    assert f"N {total - 6 + 100}" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
