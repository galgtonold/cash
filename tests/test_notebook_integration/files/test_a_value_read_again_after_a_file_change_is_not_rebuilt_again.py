"""After a changed file was read again, a cell further down rebuilds nothing more.

The upstream check remembers, per cell, which names a changed file left
stale when it last simulated the notebook. Once a repair had read the new
file, the next cell below still found ``score`` stale and ran its statement
again. A statement served from the cache hid that; one that runs every time
-- it calls a helper appending to a log -- ran twice, and the log got a row
a plain run does not write.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

CELLS = [
    "import cash\n%cash_on",
    "rows = [float(v) for v in open('data.csv').read().split()]",
    "def evaluate(values):\n"
    "    with open('runs.log', 'a') as fh:\n"
    "        fh.write('run\\n')\n"
    "    return round(sum(values) / len(values), 3)",
    "score = evaluate(rows)\nprint('score', score)",
    "print('report', score, len(rows))",
]


@pytest.mark.parametrize("restart", [False, True], ids=["same-kernel", "after-a-restart"])
def test_the_log_gets_one_row_per_cell_run(nb_runner, restart):
    work = Path(nb_runner.work_dir)
    (work / "data.csv").write_text("1\n2\n3\n", encoding="utf-8", newline="\n")
    log = work / "runs.log"
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert log.read_text(encoding="utf-8").split() == ["run"]
    if restart:
        nb_runner.restart()
        nb_runner.run_cell(1)
    (work / "data.csv").write_text("1\n2\n3\n10\n", encoding="utf-8", newline="\n")
    nb_runner.run_cell(4)
    assert "score 4.0" in nb_runner.get_output(4)
    assert log.read_text(encoding="utf-8").split() == ["run", "run"]
    nb_runner.run_cell(5)
    assert "report 4.0 4" in nb_runner.get_output(5)
    assert log.read_text(encoding="utf-8").split() == ["run", "run"], "the report re-ran the evaluation"
