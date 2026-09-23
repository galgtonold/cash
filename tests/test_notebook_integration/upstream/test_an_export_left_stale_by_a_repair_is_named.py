"""A repair that rebuilds an export's input says the export is now out of date.

An upstream edit, then a run of a cell further down.
The repair rebuilt ``sweep`` for that cell, but not ``sweep.to_csv(...)`` in
the cell between -- nothing the run needs reads the file, and a plain kernel
leaves a cell the user did not run alone too. The file kept the pre-edit
numbers while the badge said "1 upstream step not re-run (what they built is
already current)". Not re-writing is right; claiming it is current is not.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.files, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import pandas as pd\nK = 2",
    "sweep = pd.DataFrame({'v': [1, 2, 3]}) * K",
    "sweep.to_csv('sweep.csv', index=False)",
    "print('total', int(sweep['v'].sum()))",
]


def test_the_badge_names_the_export_the_repair_left_stale(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "total 12" in nb_runner.get_output(5)

    nb_runner.set_cell_source(2, "import pandas as pd\nK = 3")
    nb_runner.run_cell(5)

    out = nb_runner.get_output(5)
    assert "total 18" in out, nb_runner.get_raw_output(5)
    # the file is left alone, as a plain kernel leaves it ...
    assert (nb_runner.work_dir / "sweep.csv").read_text().split() == ["v", "2", "4", "6"]
    # ... and the badge says so
    assert "sweep.csv" in out and "not rewritten" in out, out
    assert "already current" not in out, out


def test_an_export_whose_inputs_did_not_change_is_not_named(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(5, "print('total', int(sweep['v'].sum()), 'again')")
    nb_runner.run_cell(5)
    out = nb_runner.get_output(5)
    assert "total 12 again" in out, nb_runner.get_raw_output(5)
    assert "not rewritten" not in out, out
