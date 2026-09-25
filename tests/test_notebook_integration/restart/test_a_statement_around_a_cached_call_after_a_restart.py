"""After a restart, a statement around a cached call reads the call back from
disk and runs its cheap rest again.

``b = shifted(a) + [1]`` is stored for its own work, which is the ``+ [1]``:
its value is not written a second time beside the call's result. So after a
restart the call must come back from its own disk entry, the rest must run,
and an edit above must still reach it -- never an old ``b``.

Counted, not timed: the callee writes a byte per run with ``os.open`` (not
``builtins.open``, which the file tracker would make a dependency of it).
"""

import pytest

pytest.importorskip("pandas")

from tests._nbharness.badge import shows_cached, shows_executed

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


def _cells(ticks):
    return [
        "import cash\n%cash_on\n%cash_badge print",
        "import os, time\n"
        "import pandas as pd\n"
        "def _tick():\n"
        f"    fd = os.open(r'{ticks}', os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'X')\n"
        "    os.close(fd)\n"
        "def shifted(x):\n"
        "    _tick()\n"
        "    time.sleep(0.3)\n"
        "    return x + 100",
        "a = pd.DataFrame({'x': range(200_000)})",
        "b = shifted(a) + 1",
        "c = shifted(a) + 2",
        "print('B', len(b), int(b.x.iloc[0]), int(b.x.iloc[-1]), 'C', int(c.x.iloc[-1]))",
    ]


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def test_the_call_comes_back_from_disk_and_the_rest_runs(nb_runner, tmp_path):
    """F and G: neither statement kept a value, so both run again after the
    restart, and neither runs the call."""
    ticks = tmp_path / "ticks.log"
    nb_runner.create_notebook(_cells(ticks))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "B 200000 101 200100 C 200101" in nb_runner.get_output(6)
    assert _n(ticks) == 1, "the second statement did not get the call from the cache"

    nb_runner.restart()
    nb_runner.run_all()
    assert "B 200000 101 200100 C 200101" in nb_runner.get_output(6)
    assert _n(ticks) == 1, "after the restart the call ran again instead of coming back from disk"
    assert shows_executed(nb_runner.get_output(4)), "the rest of the statement was restored, not run"

    # No stale value: an edit above reaches both statements, through a new call.
    nb_runner.set_cell_source(3, "a = pd.DataFrame({'x': range(5, 200_005)})")
    nb_runner.run_all()
    assert "B 200000 106 200105 C 200106" in nb_runner.get_output(6)
    assert _n(ticks) == 2


def test_a_statement_with_expensive_work_of_its_own_restores(nb_runner, tmp_path):
    """C: the statement's own work is worth keeping, so its value is stored and
    comes back after a restart without running the statement or the call."""
    ticks = tmp_path / "ticks.log"
    cells = _cells(ticks)
    cells[3] = "b = shifted(a) + (time.sleep(0.3) or 1)"
    nb_runner.create_notebook(cells[:4] + ["print('B', len(b), int(b.x.iloc[-1]))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "B 200000 200100" in nb_runner.get_output(5)

    nb_runner.restart()
    nb_runner.run_all()
    assert "B 200000 200100" in nb_runner.get_output(5)
    assert _n(ticks) == 1
    assert shows_cached(nb_runner.get_output(4)) and not shows_executed(nb_runner.get_output(4)), nb_runner.get_output(
        4
    )
