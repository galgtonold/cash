"""A statement that writes a file and binds a name the cell needs, re-run for
that name, is a writer too: what reads the file after it reads it again.

``n = P.write_text(s)`` is planned as the producer of ``n`` once ``s`` is
edited. The check that re-reads files after a re-run writer only looked at
writers it scheduled itself, so ``back = P.read_text()`` kept the old text
while ``n`` counted the new one: ``GOT 3 v1``, a state no plain run has.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _cells(path, text):
    return [
        f"import pathlib\nP = pathlib.Path(r'{path}')\ns = '{text}'",
        "n = P.write_text(s)",
        "back = P.read_text()",
        "print('GOT', n, back)",
    ]


def test_the_reader_after_a_writer_run_for_its_value_sees_the_new_file(nb_runner, tmp_path):
    path = (tmp_path / "note.txt").as_posix()
    nb_runner.create_notebook(_cells(path, "v1"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "GOT 2 v1" in nb_runner.get_output(4)

    nb_runner.set_cell_source(1, _cells(path, "v22")[0])
    nb_runner.run_cell(4)
    assert "GOT 3 v22" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_an_unchanged_rerun_does_not_rewrite_the_file(nb_runner, tmp_path):
    path = tmp_path / "note.txt"
    nb_runner.create_notebook(_cells(path.as_posix(), "v1"))
    nb_runner.start_kernel()
    nb_runner.run_all()
    written = path.stat().st_mtime_ns

    nb_runner.run_cell(4)
    assert "GOT 2 v1" in nb_runner.get_output(4)
    assert path.stat().st_mtime_ns == written
