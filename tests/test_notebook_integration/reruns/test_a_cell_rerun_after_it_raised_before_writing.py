"""A cell re-run while its error is fixed, after a run that raised before it
wrote into a large value, rebuilds nothing above it.

``rows[0] = rows['typo']`` raises before its item store, so the list is as
the cells above left it. The upstream check rebuilt it before every run of
the cell all the same (the cell had started from that object last time),
seconds per run where plain Python only shows the error. The producer's
runs are counted with an ``os.open`` append; the error is the one plain
IPython shows, and the list is the one Restart & Run All leaves.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(240)]

_MAKE = """\
import os
LOG = 'runs.log'
def make(n):
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b'x')
    os.close(fd)
    return [[i] for i in range(n)]"""
_PRODUCE = "rows = make(2_000)"
_PROBE = "print('runs', os.path.getsize(LOG), 'len', len(rows), 'first', rows[0], 'fifth', rows[5])"


def _error(nb_runner, cell: int) -> tuple[str, str]:
    with pytest.raises(CellExecutionError) as excinfo:
        nb_runner.run_cell(cell)
    return excinfo.value.ename, excinfo.value.evalue


def _plain(nb_runner, cell: str) -> tuple[tuple[str, str], str]:
    """What plain IPython raises for *cell*, and the probe after it."""
    nb_runner.create_notebook([_MAKE, _PRODUCE, cell, _PROBE])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_cells([1, 2])
    error = _error(nb_runner, 3)
    nb_runner.run_cell(4)
    probe = nb_runner.get_output(4)
    nb_runner.shutdown()
    return error, probe


def _attempts(nb_runner, attempts: list[str]) -> tuple[list[tuple[str, str]], str]:
    (nb_runner.work_dir / "runs.log").unlink(missing_ok=True)  # the plain run's
    nb_runner.create_notebook([_MAKE, _PRODUCE, attempts[0], _PROBE])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    errors = []
    for source in attempts:
        nb_runner.set_cell_source(3, source)
        errors.append(_error(nb_runner, 3))
    nb_runner.run_cell(4)
    return errors, nb_runner.get_output(4)


def test_no_rebuild_while_the_error_is_fixed(nb_runner):
    attempts = ["rows[0] = rows['typo']", "rows[0] = rows['typo2']", "n = len(rows)\nrows[0] = rows['typo2']"]
    plain_error, plain_probe = _plain(nb_runner, attempts[-1])

    errors, probe = _attempts(nb_runner, attempts)

    assert errors[-1] == plain_error
    assert all(error[0] == "TypeError" for error in errors), errors
    assert probe == plain_probe
    assert "runs 1 " in probe, f"the producer above re-ran: {probe}"


def test_a_write_before_the_error_is_dropped_on_the_next_run(nb_runner):
    """The control: ``rows[5]`` was written before the second item store
    raised, so the next run starts from the list as the cells above leave
    it, as Restart & Run All does."""
    attempts = ["rows[5], rows['k'] = ['w'], 1", "rows[0], rows['k'] = [7], 1"]
    plain_error, plain_probe = _plain(nb_runner, attempts[-1])

    errors, probe = _attempts(nb_runner, attempts)

    assert errors[-1] == plain_error
    assert "fifth [5]" in probe and "first [7]" in probe, probe
    assert probe.split("len")[1] == plain_probe.split("len")[1]
