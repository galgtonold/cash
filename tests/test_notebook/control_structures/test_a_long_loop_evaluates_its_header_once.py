"""A loop run as one unit evaluates its header once, as Python does.

``for a, b in tqdm(list(zip(actions, next_actions))):`` over 2M pairs: the
header was evaluated to size the loop, then again by the unit that ran it, so
the list was built twice and a second bar drawn (3.4 s plain against 8.2 s).
A header that is a call to a pure builtin producer or a progress bar is now
sized from what it wraps, and evaluated only by the unit.
"""

from __future__ import annotations

import re

import pytest

from tests._cell_driver import run_cash_cell

# The progress bar is one of the headers, and every case imports it.
pytest.importorskip("tqdm")

COUNTED = (
    "class Counted:\n"
    "    def __init__(self, n):\n        self.n, self.iters = n, 0\n"
    "    def __len__(self):\n        return self.n\n"
    "    def __iter__(self):\n        self.iters += 1\n        return iter(range(self.n))\n"
    "xs, ys = Counted(2000), Counted(2000)\n"
)


@pytest.mark.parametrize(
    "header",
    ["list(zip(xs, ys))", "zip(xs, ys)", "sorted(zip(xs, ys))", "tqdm(list(zip(xs, ys)))"],
)
def test_the_header_is_iterated_once(cash_magics, header):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, COUNTED + "from tqdm import tqdm")
    run_cash_cell(cash_magics, f"total = 0\nfor a, b in {header}:\n    total += a * b")
    assert ns["total"] == sum(i * i for i in range(2000))
    assert (ns["xs"].iters, ns["ys"].iters) == (1, 1)


def test_one_bar_is_drawn(cash_magics, capsys):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "from tqdm import tqdm\nxs = list(range(2000))")
    capsys.readouterr()
    run_cash_cell(cash_magics, "total = 0\nfor a, b in tqdm(list(zip(xs, xs))):\n    total += a * b")
    assert ns["total"] == sum(i * i for i in range(2000))
    drawn = capsys.readouterr().err
    assert len(re.findall(r"(?<![0-9])0/2000", drawn)) == 1, drawn  # each bar starts at 0/2000


def test_a_rerun_after_an_edit_reads_the_new_data(cash_magics):
    ns = cash_magics.shell.user_ns
    loop = "total = 0\nfor a, b in list(zip(xs, ys)):\n    total += a * b"
    run_cash_cell(cash_magics, "xs = list(range(2000))\nys = [1] * 2000")
    run_cash_cell(cash_magics, loop)
    assert ns["total"] == sum(range(2000))
    run_cash_cell(cash_magics, "ys = [2] * 2000")
    run_cash_cell(cash_magics, loop)
    assert ns["total"] == 2 * sum(range(2000))


def test_a_shadowed_builtin_is_evaluated_first(cash_magics):
    """A notebook's own ``list`` gets no benefit of the doubt: the header is
    evaluated before the loop is judged, as before."""
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "calls = []\ndef list(x):\n    calls.append(1)\n    return [*x]\nxs = range(2000)")
    run_cash_cell(cash_magics, "total = 0\nfor a in list(xs):\n    total += a")
    assert ns["total"] == sum(range(2000))
    assert len(ns["calls"]) == 1
