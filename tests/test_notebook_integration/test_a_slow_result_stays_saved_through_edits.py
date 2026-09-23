"""A slow result keeps being saved however often the cells above it change.

The perpetual-miss guard stops writing a statement whose cache key has
changed on five runs in a row without a hit: such a key usually hashes
something unstable, and writing a large value every run then only costs.
But five edits upstream in a row is an ordinary morning of model tuning.
Round 23's r23s1 edited one model at a time; after the fifth the
cross-validation -- the slowest step in the notebook, its value a few
numbers -- was no longer saved, and the next morning's restart ran every CV
again although nothing had changed since the evening.

The guard now spares a statement whose value is cheap to write next to what
computing it cost: a wasted write is then nearly free, and one later hit
repays all of them.

Counted with ``os.write`` from inside the slow function.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(180)]

SETUP = (
    "import os, time\n"
    "def slow_score(seed):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.3)\n"
    "    return seed * 1.5"
)
# The statement's own entry is what the guard governs; the call inside it has
# a cache of its own, which r23s1's comprehension did not get (#225).
SCORE = "# @cash:no-cache-calls\nscore = slow_score(SEED)\nprint('SCORE', score)"


def _runs(runner) -> int:
    log = Path(runner.work_dir) / "runs.log"
    return log.read_text().count("x") if log.exists() else 0


def test_a_slow_small_result_is_restored_after_six_upstream_edits(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, "SEED = 0", SCORE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    for seed in range(1, 6):  # five more keys, none seen before
        nb_runner.set_cell_source(3, f"SEED = {seed}")
        nb_runner.run_cells([3, 4])
    assert "SCORE 7.5" in nb_runner.get_output(4)
    assert _runs(nb_runner) == 6

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_all()

    assert "SCORE 7.5" in nb_runner.get_output(4)
    # Measured before: 7 -- the sixth result was never written.
    assert _runs(nb_runner) == 6, "the last result was not saved; the restart computed it again"
