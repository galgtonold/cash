"""What an interrupted statement printed and displayed is shown, as in plain.

A statement's output is held while it runs and shown when it ends. An
interrupt ends it by leaving, past that step, so a long call with progress
prints (a fit, a loop over a frame's rows) showed nothing but the
KeyboardInterrupt. Plain shows every line printed before it, first.
"""

import asyncio
import threading

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

HELPERS = (
    "import time\n"
    "from IPython.display import display, HTML\n"
    "def rows(n, then=None):\n"
    "    for i in range(n):\n"
    "        print('row', i, flush=True)\n"
    "    display(HTML('<b>shown</b>'))\n"
    "    if then is not None:\n"
    "        raise then\n"
    "    return n\n"
    "def slow_rows():\n"
    "    for i in range(1000):\n"
    "        print('slow', i, flush=True)\n"
    "        time.sleep(0.2)\n"
)


def _outputs(nb_runner, cell_num):
    """The cell's outputs in order: stream lines, 'html' for the helper's display, the error's name.

    Stream text is joined before it is split into lines, as the notebook shows
    it: printed as it runs, a line can reach the notebook in two messages."""
    seen, text = [], []

    def lines():
        seen.extend(line for line in "".join(text).splitlines() if line and not line.startswith("[Cash]"))
        text.clear()

    for o in nb_runner.nb.cells[cell_num - 1].outputs:
        if o.output_type == "stream":
            if o.name == "stdout":
                text.append(o.text)
            continue
        lines()
        if o.output_type == "display_data" and "<b>shown</b>" in o.data.get("text/html", ""):
            seen.append("html")
        elif o.output_type == "error":
            seen.append(o.ename)
    lines()
    return seen


def _run_raising(nb_runner, cell_num):
    with pytest.raises(CellExecutionError):
        nb_runner.run_cell(cell_num)
    return _outputs(nb_runner, cell_num)


@pytest.mark.parametrize("error", ["KeyboardInterrupt", "ValueError"])
def test_a_statement_that_raises_shows_what_it_printed_first(nb_runner, error):
    cell = f"print('before')\nn = rows(3, then={error}())"
    nb_runner.create_notebook([HELPERS, cell])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    expected = ["before", "row 0", "row 1", "row 2", "html", error]

    assert _run_raising(nb_runner, 2) == expected
    # Run again: nothing was stored, the statement computes and shows it again.
    assert _run_raising(nb_runner, 2) == expected


def test_a_served_statement_and_an_interrupted_one_after_it_both_show(nb_runner):
    cell = "m = rows(2)\nn = rows(3, then=KeyboardInterrupt())"
    nb_runner.create_notebook([HELPERS, cell])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)
    expected = ["row 0", "row 1", "html", "row 0", "row 1", "row 2", "html", "KeyboardInterrupt"]

    assert _run_raising(nb_runner, 2) == expected
    assert _run_raising(nb_runner, 2) == expected  # m is served this time


def test_an_interrupted_call_shows_the_lines_it_printed(nb_runner):
    nb_runner.create_notebook([HELPERS, "print('before')\nx = slow_rows()"])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)

    def interrupt():  # Kernel > Interrupt
        res = nb_runner.client.km.interrupt_kernel()
        if asyncio.iscoroutine(res):
            asyncio.run_coroutine_threadsafe(res, nb_runner._loop)

    timer = threading.Timer(3.0, interrupt)
    timer.start()
    try:
        seen = _run_raising(nb_runner, 2)
    finally:
        timer.cancel()
    assert seen[0] == "before" and seen[-1] == "KeyboardInterrupt", seen
    printed = seen[1:-1]
    assert len(printed) >= 3, seen
    assert printed == [f"slow {i}" for i in range(len(printed))], seen
