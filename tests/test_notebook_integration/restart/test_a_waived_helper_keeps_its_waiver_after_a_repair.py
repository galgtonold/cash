"""A helper re-defined by an upstream repair keeps its ``# @cash:assume-safe``.

A repair runs a statement from its unparsed code, which has no comments.
After a restart, a helper defined that way lost the waiver on the line that
appends to its call counter, so the statement calling it counted as one that
writes files and ran every time: the expensive call was made again although
its result was in the cache.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

CELLS = [
    "import cash\n%cash_on",
    "import time\n"
    "def slow(v):\n"
    "    open('calls.log', 'a').write('slow\\n')  # @cash:assume-safe (a call counter)\n"
    "    time.sleep(0.3)\n"
    "    return v * 2",
    "x = slow(21)",
    "print('x =', x)",
]


def test_the_call_is_served_from_the_cache_after_a_restart(nb_runner):
    log = Path(nb_runner.work_dir) / "calls.log"
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert log.read_text(encoding="utf-8").split() == ["slow"]
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.enable_persist()
    # Straight to the call: the repair defines `slow` first.
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    assert "x = 42" in nb_runner.get_output(4)
    assert log.read_text(encoding="utf-8").split() == ["slow"]
