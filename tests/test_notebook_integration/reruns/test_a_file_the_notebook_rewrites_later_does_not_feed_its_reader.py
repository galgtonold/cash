"""A file a cell reads and then rewrites itself does not change what the
read gave.

``open(P, 'w').write('1')``, a loop reading ``P``, then
``open(P, 'w').write('2')`` in the same cell: the cell below checked the
loop's files, found ``P`` changed, and ran the loop again, so ``u`` became 2
where plain Python has 1. The change is the notebook's own, made after the
read, as in a top-to-bottom run.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

PIN = "import cash\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n%cash_on"

LOOP = "for _ in range(1):\n    t = int(open(P).read())\n"

CELLS = {
    "a loop": ["open(P, 'w').write('1')\n" + LOOP + "u = t\nopen(P, 'w').write('2')", "w = u"],
    "two loops": [
        "open(P, 'w').write('1')\n" + LOOP + "u = t\nopen(P, 'w').write('2')\n" + LOOP + "v = t",
        "w = (u, v)",
    ],
    "two different loops": [
        "open(P, 'w').write('1')\n" + LOOP + "u = t\nopen(P, 'w').write('2')\n"
        "for _ in range(1):\n    t = int(open(P).read()) + 0\nv = t",
        "w = (u, v)",
    ],
    "statements": [
        "open(P, 'w').write('1')\nt = int(open(P).read())\nu = t\nopen(P, 'w').write('2')\n"
        "t = int(open(P).read())\nv = t",
        "w = (u, v)",
    ],
    "cells": ["open(P, 'w').write('1')\n" + LOOP + "u = t", "open(P, 'w').write('2')\n" + LOOP + "v = t", "w = (u, v)"],
    "cells, a plain read below": [
        "open(P, 'w').write('1')\n" + LOOP + "u = t",
        "open(P, 'w').write('2')\nv = int(open(P).read())",
        "w = (u, v)",
    ],
}

WANT = {"a loop": "1"}


@pytest.mark.parametrize("shape", list(CELLS))
def test_every_run_matches_plain(nb_runner, shape):
    path = (nb_runner.work_dir / "f.txt").as_posix()
    nb_runner.create_notebook([PIN, f"import os\nP = {path!r}", *CELLS[shape]])
    nb_runner.start_kernel()
    seen = []
    for runs in (3, 2):
        for _ in range(runs):
            nb_runner.run_all()
            seen.append(nb_runner.peek("w"))
        if runs == 3:
            nb_runner.settle_writes()
            nb_runner.restart()
            nb_runner._init_cash()
    assert seen == [WANT.get(shape, "(1, 2)")] * 5


ABOVE = {
    "a loop": LOOP + "u = t",
    "a statement": "t = int(open(P).read())\nu = t",
}


@pytest.mark.parametrize("reader", list(ABOVE))
def test_a_cell_above_the_reader_rewriting_the_file_is_a_change(nb_runner, reader):
    """Run All, then the writer above the reader again on its own, then
    only the cell below: Run All would have the reader read the new text."""
    path = (nb_runner.work_dir / "f.txt").as_posix()
    writer = "import time\nopen(P, 'w').write(str(time.time_ns()))"
    nb_runner.create_notebook([PIN, f"P = {path!r}", writer, ABOVE[reader], "w = u"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    for _ in range(2):
        nb_runner.run_cell(3)
        nb_runner.run_cell(5)
        assert nb_runner.peek("w == int(open(P).read())") == "True"
