"""A frame rebuilt to the same content keeps what was cached from it.

In a student's notebook ``temptastic = tsimp[...]`` ran again after
``tsimp`` had been rebuilt above it. The table held the same rows, but its
lineage named the new ``tsimp``, so the 150 s loop reading it ran again.
A frame, series or array a statement executed to bind now takes the lineage
of an earlier one with the same content.

Counted with ``os.write`` from inside the cached function.
"""

from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration]

ON = "import cash\n%cash_on"
SETUP = (
    "import os, time\n"
    "import pandas as pd\n"
    "def slow_total(frame):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.2)\n"
    "    return int(frame['a'].sum())"
)
RAW = "raw = pd.DataFrame({'a': range(300_000), 'b': 1})"
PICK = "t = raw[raw.a >= 0][['a']]"
TOTAL = "total = slow_total(t)\nprint('TOTAL', total)"


def _runs(runner) -> int:
    log = Path(runner.work_dir) / "runs.log"
    return log.read_text(encoding="utf-8").count("x") if log.exists() else 0


@pytest.mark.parametrize(
    "edited",
    [
        "raw = pd.DataFrame({'a': list(range(300_000)), 'b': 1})",
        "raw = pd.DataFrame({'a': range(300_000), 'b': 2})",
    ],
)
def test_an_edit_that_rebuilds_the_same_table_keeps_what_is_built_from_it(nb_runner, edited):
    """The first edit rebuilds ``raw`` as it was; the second changes a
    column ``t`` leaves out, so ``t`` comes out the same."""
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 1, "a table rebuilt to the same content ran what was built from it again"

    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 1


def test_an_edit_that_changes_the_table_still_runs_what_is_built_from_it(nb_runner):
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': range(1, 300_001), 'b': 1})")
    nb_runner.run_all()

    assert "TOTAL 45000150000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 2


def test_the_old_value_back_after_a_change_is_served_again(nb_runner):
    """Edited away and back: the table equals the first one again, and the
    total built from it then is served."""
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': range(1, 300_001), 'b': 1})")
    nb_runner.run_all()
    assert _runs(nb_runner) == 2

    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': list(range(300_000)), 'b': 1})")
    nb_runner.run_all()

    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert _runs(nb_runner) == 2


def test_after_a_restart_the_results_are_right(nb_runner):
    """The aliases live for the process: after a restart each value is
    named by its formula again, and whatever runs or is restored must
    still be right."""
    nb_runner.create_notebook([ON, SETUP, RAW, PICK, TOTAL, "print('LEN', len(t))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "raw = pd.DataFrame({'a': list(range(300_000)), 'b': 1})")
    nb_runner.run_all()

    nb_runner.restart()
    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
    assert "LEN 300000" in nb_runner.get_output(6)

    nb_runner.set_cell_source(3, RAW)
    nb_runner.run_all()
    assert "TOTAL 44999850000" in nb_runner.get_output(5)
