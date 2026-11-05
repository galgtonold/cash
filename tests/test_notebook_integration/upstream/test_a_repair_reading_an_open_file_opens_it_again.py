"""A statement a repair re-runs that reads an open file gets the file opened
again first, in a real kernel, as a run from the top does.

Found by replaying a real session (JuNE student_2): ``f = open(path)`` and
``lines = f.readlines()`` in one cell, then a cell copying the lines and
ending ``lines = lines2``. To hand that cell its ``lines``, cash re-ran
``lines = f.readlines()`` alone, on the handle already read to its end, in a
plain Run All: the cell and every cell below counted 0 lines, without a
warning. A run from the top reads all of them.
"""

import pytest

pytestmark = [pytest.mark.timeout(120), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_on"
OPEN = 'data_path = "log.txt"\n\nf = open(data_path, "r")\nlines = f.readlines()\nlen(lines)'
COPY = "lines2 = []\nfor line in lines:\n    lines2.append(line)\n\nlines = lines2\nprint('COPIED', len(lines))"
COUNT = "print('N', len(lines))"


def test_a_run_all_counts_every_line(nb_runner):
    (nb_runner.work_dir / "log.txt").write_text("".join(f"@User{i}: Action_1 -> Action_2\n" for i in range(50)))
    nb_runner.create_notebook([SETUP, OPEN, COPY, COUNT])
    nb_runner.start_kernel()

    nb_runner.run_all()

    assert "COPIED 50" in nb_runner.get_output(3), nb_runner.get_output(3)
    assert "N 50" in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_all()
    assert "N 50" in nb_runner.get_output(4), nb_runner.get_output(4)
