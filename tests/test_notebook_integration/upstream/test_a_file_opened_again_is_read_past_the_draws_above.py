"""A file the upstream repair opens again is read past what the cells above
drew from it, in a real kernel, as a run from the top reads it.

``f = open(path)``, a cell reading the header off it, and a cell reading the
rows. After a restart, running the rows cell made ``f`` again by re-running
``f = open(path)`` alone: the header came back as the first row. Re-run on
its own after ``next(f)`` drew the header, the rows cell read nothing, where
a run from the top reads every row.
"""

import pytest

pytestmark = [pytest.mark.timeout(240), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_on"
OPEN = "f = open('d.csv')"
ROWS = "rows = f.readlines()\nprint('ROWS', len(rows), rows[0].strip())"
DATA = "h1,h2\n1,2\n3,4\n5,6\n"


def _notebook(nb_runner, skip):
    (nb_runner.work_dir / "d.csv").write_text(DATA, encoding="utf-8")
    nb_runner.create_notebook([SETUP, OPEN, skip, ROWS])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "ROWS 3 1,2" in nb_runner.get_output(4), nb_runner.get_output(4)


@pytest.mark.parametrize("skip", ["header = f.readline()", "header = next(f)"])
def test_after_a_restart_the_rows_cell_skips_the_header(nb_runner, skip):
    _notebook(nb_runner, skip)
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner.run_cell(1)

    nb_runner.run_cell(4)

    assert "ROWS 3 1,2" in nb_runner.get_output(4), nb_runner.get_output(4)


@pytest.mark.parametrize("skip", ["header = next(f)", "next(f)"])
def test_the_rows_cell_re_run_alone_reads_every_row(nb_runner, skip):
    _notebook(nb_runner, skip)

    nb_runner.run_cell(4)
    assert "ROWS 3 1,2" in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_cell(4)
    assert "ROWS 3 1,2" in nb_runner.get_output(4), nb_runner.get_output(4)


def test_a_second_run_all_reads_the_same_rows(nb_runner):
    _notebook(nb_runner, "header = next(f)")

    nb_runner.run_all()

    assert "ROWS 3 1,2" in nb_runner.get_output(4), nb_runner.get_output(4)
