"""A cheap value over a costly input survives a restart without its input.

A statement is written to disk when its own compute time clears the floor
(0.1 s), so ``latest = int(raw['v'].max())`` -- milliseconds -- lived in RAM
only. After a restart the next cell that needed it rebuilt ``raw`` and
everything behind it: in round 23's r23s2, 49 statements and a 1,200-file
folder read again (35 s) for a table cell whose inputs were a few KB. Each
read there was under the floor too, so nothing in the chain was on disk.

At the end of a cell, each value it leaves is now judged by what rebuilding
it would cost -- the entries not on disk it came through -- and written to
disk when restoring beats that, by the same cost-model rule.

Observed in the kernel: reading the folder again binds ``parts``; restoring
what was built from it does not.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

READS = 20
SETUP = (
    "import glob\nimport time\nimport pandas as pd\ndef slow_read(f):\n    time.sleep(0.06)\n    return pd.read_csv(f)"
)
# Every read is under the persistence floor on its own; together they are not.
# A body of several statements, as r23s2's: cached per iteration, in RAM.
LOAD = (
    "files = sorted(glob.glob('exports/*.csv'))\n"
    "parts = []\n"
    "for f in files:\n    d = slow_read(f)\n    d['source'] = f\n    parts.append(d)\n"
    "raw = pd.concat(parts, ignore_index=True)"
)
LATEST = "latest = int(raw['v'].max())"
SHOW = "print('LATEST', latest)"
CELLS = ["import cash\n%cash_on", SETUP, LOAD, LATEST, SHOW]


def _exports(work_dir: Path) -> None:
    folder = work_dir / "exports"
    folder.mkdir()
    for i in range(READS):
        (folder / f"e{i:02d}.csv").write_text(f"k,v\n{i},{i * 3}\n")


def test_a_cheap_value_over_a_costly_input_is_restored_after_a_restart(nb_runner):
    _exports(Path(nb_runner.work_dir))
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert f"LATEST {3 * (READS - 1)}" in nb_runner.get_output(len(CELLS))

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(len(CELLS))

    assert f"LATEST {3 * (READS - 1)}" in nb_runner.get_output(len(CELLS))
    # Measured before the fix: 'True' -- the folder read again to rebuild it.
    assert nb_runner.peek("'parts' in globals()") == "False", "the folder was read again"
