"""A statement re-run as an upstream repair keeps its ``# @cash:`` directives.

Round 25's r25s4 put ``# @cash:no-cache-calls`` on a comprehension. Run
directly, its calls were not cached. Re-run as an Upstream repair (edit the cell
above, run the cell below), they were: the repair executed the statement's code
with no annotation at all, so the directive only held for direct runs. Values
were right; the directive was not.

Counted with ``os.write``: a second repair over values the first one saw runs
every call again when the directive held, and none when the repair cached them.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

WORK = (
    "import os, time\n"
    "def work(i):\n"
    "    fd = os.open('calls.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'w|')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.06)\n"
    "    return i * i"
)
# The tester's layout: the directive mid-cell, under the function it calls.
COMPREHENSION = WORK + "\n\n# @cash:no-cache-calls\nvals = [work(i) for i in range(N)]"


def _calls(runner) -> int:
    log = Path(runner.work_dir) / "calls.log"
    return log.read_text().count("w|") if log.exists() else 0


def test_no_cache_calls_holds_when_the_statement_is_repaired(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", "N = 20", "pass", COMPREHENSION, "print('SUM', sum(vals))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "SUM 2470" in nb_runner.get_output(5), nb_runner.get_output(5)

    nb_runner.set_cell_source(2, "N = 21")
    nb_runner.run_cell(5)                       # cell 4 re-runs as an upstream repair
    assert "SUM 2870" in nb_runner.get_output(5), nb_runner.get_output(5)

    # A second repair over values the first one saw: had it cached the calls,
    # these would all be hits.
    before = _calls(nb_runner)
    nb_runner.set_cell_source(2, "N = 19")
    nb_runner.run_cell(5)
    assert "SUM 2109" in nb_runner.get_output(5), nb_runner.get_output(5)
    assert _calls(nb_runner) - before == 19, "the repair cached the calls the directive refuses"
