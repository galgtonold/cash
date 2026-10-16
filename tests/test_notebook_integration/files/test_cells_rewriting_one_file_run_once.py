"""Cells that each read a file and write it back run once each on Run All.

Three cells each read a counter from a file, add 1 and write it back. Plain
Python leaves 3. cash re-ran the cells above each one before it ran: the file
a cell above had read had changed since -- by that cell's own write, after the
read, as in any top-to-bottom run -- so each writer ran again, and a fresh Run
All left 9. A change the notebook itself made after a read is no reason to
read again; one from outside, or from an edited cell above, still is.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files]

STEP = "n = int(open(P).read())\nopen(P, 'w').write(str(n + 1))"
SETUP = "P = 'count.txt'\nopen(P, 'w').write('{start}')"


def _cells(start=0):
    return [
        "import cash\n%cash_on",
        SETUP.format(start=start),
        "# step 1\n" + STEP,
        "# step 2\n" + STEP,
        "# step 3\n" + STEP,
        "print('HOLDS', open(P).read())",
    ]


def _holds(nb_runner):
    return (nb_runner.work_dir / "count.txt").read_text()


def test_run_all_twice_gives_what_plain_python_gives(nb_runner):
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "HOLDS 3" in nb_runner.get_output(6), nb_runner.get_output(6)
    assert _holds(nb_runner) == "3"
    nb_runner.run_all()
    assert "HOLDS 3" in nb_runner.get_output(6), nb_runner.get_output(6)
    assert _holds(nb_runner) == "3"


def test_a_cell_run_alone_after_its_own_write_reads_what_it_wrote(nb_runner):
    """Running step 1 again by hand reads 1, as in a plain kernel."""
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3])
    nb_runner.run_cell(3)
    nb_runner.run_cell(6)
    assert "HOLDS 2" in nb_runner.get_output(6), nb_runner.get_output(6)


def test_a_change_from_outside_is_still_read(nb_runner):
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_all()
    (nb_runner.work_dir / "count.txt").write_text("10")
    nb_runner.run_cell(5)
    nb_runner.run_cell(6)
    assert "HOLDS 11" in nb_runner.get_output(6), nb_runner.get_output(6)


def test_a_value_read_before_the_cells_own_write_keeps_its_value_below(nb_runner):
    """``n`` is what the file held before the cell wrote it back; a cell below
    that uses ``n``, run again, uses that ``n``, not the file's new content."""
    cells = [
        "import cash\n%cash_on\nimport time",
        SETUP.format(start=0),
        STEP,
        "m = (time.sleep(0.3), n * 10)[1]\nprint('M', m)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "M 0" in nb_runner.get_output(4), nb_runner.get_output(4)
    nb_runner.run_cell(4)
    assert "M 0" in nb_runner.get_output(4), nb_runner.get_output(4)
    assert _holds(nb_runner) == "1"


def test_an_edited_cell_above_rebuilds_the_chain_once(nb_runner):
    """Edit the setup to start at 5 and run only the last cell: the steps run
    once each after it, as top to bottom: 8."""
    nb_runner.create_notebook(_cells())
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, SETUP.format(start=5))
    nb_runner.run_cell(6)
    assert "HOLDS 8" in nb_runner.get_output(6), nb_runner.get_output(6)
