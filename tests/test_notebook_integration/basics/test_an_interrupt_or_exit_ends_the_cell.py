"""An interrupt or a ``sys.exit()`` ends the cell as in plain IPython.

Both are ``BaseException``s, not ``Exception``s, and cash let them escape its
``run_cell``: an interrupted cell never got its reply, so the frontend waited
on it forever, and a ``SystemExit`` (``sys.exit()``, or argparse failing on
the kernel's own argv) shut the kernel down with every variable in it.
"""

import asyncio
import threading

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]


def _error_names(nb_runner, cell_num):
    return [o.ename for o in nb_runner.nb.cells[cell_num - 1].outputs if o.output_type == "error"]


def test_an_interrupted_cell_ends_with_keyboard_interrupt(nb_runner):
    nb_runner.create_notebook(["x = 1", "import time\ntime.sleep(30)\ny = 2", "print('next', x)"])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)

    def interrupt():  # Kernel > Interrupt
        res = nb_runner.client.km.interrupt_kernel()
        if asyncio.iscoroutine(res):
            asyncio.run_coroutine_threadsafe(res, nb_runner._loop)

    timer = threading.Timer(3.0, interrupt)
    timer.start()
    try:
        with pytest.raises(CellExecutionError):
            nb_runner.run_cell(2)
    finally:
        timer.cancel()
    assert _error_names(nb_runner, 2) == ["KeyboardInterrupt"]

    nb_runner.run_cell(3)
    assert nb_runner.get_output(3).strip() == "next 1"


def test_a_cell_raising_keyboard_interrupt_ends_with_it(nb_runner):
    nb_runner.create_notebook(["x = 1\nraise KeyboardInterrupt", "print('next', x)"])
    nb_runner.start_kernel()

    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(1)
    assert _error_names(nb_runner, 1) == ["KeyboardInterrupt"]

    nb_runner.run_cell(2)
    assert nb_runner.get_output(2).strip() == "next 1"


@pytest.mark.parametrize(
    "cell",
    [
        "import sys\nsys.exit(2)",
        "import argparse\np = argparse.ArgumentParser()\np.add_argument('--n', type=int, required=True)\nargs = p.parse_args([])",
    ],
    ids=["sys-exit", "argparse"],
)
def test_system_exit_keeps_the_kernel(nb_runner, cell):
    nb_runner.create_notebook(["x = 1", cell, "print('alive', x)"])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)

    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(2)
    assert _error_names(nb_runner, 2) == ["SystemExit"]

    nb_runner.run_cell(3)
    assert nb_runner.get_output(3).strip() == "alive 1"
