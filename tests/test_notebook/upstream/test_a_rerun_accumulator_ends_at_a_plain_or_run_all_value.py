"""Whatever mix of runs that go through and runs that raise a cell that
builds on a value from above gets, it ends at a value a plain kernel or
Restart & Run All gives.

A re-run of such a cell starts from what the cells above give, after a run
that went through and after one that raised alike, so it ends at the Run
All value. Before, a run after a failed one went on from what that run had
left: ok, part way, ok left 5 rows where plain gives 8 and Run All 3.
"""

from __future__ import annotations

import itertools

import pytest

from tests._cell_driver import run_cash_cell

_KINDS = {
    # A large list and frame are followed by their identity, a short list by
    # its content and a short frame by its lineage.
    "long-list": ("acc = []", "for i in range(n):\n    new = [i * {f}]\n    acc = acc + new", 3000),
    "short-list": ("acc = []", "for i in range(n):\n    new = [i * {f}]\n    acc = acc + new", 3),
    "appended": ("acc = []", "for i in range(n):\n    acc.append(i * {f})", 3000),
    "frame": (
        "import pandas as pd\nacc = pd.DataFrame({{'a': []}})",
        "for i in range(n):\n    new = pd.DataFrame({{'a': [i * {f}]}})\n    acc = pd.concat([acc, new], ignore_index=True)",
        300,
    ),
    "short-frame": (
        "import pandas as pd\nacc = pd.DataFrame({{'a': []}})",
        "for i in range(n):\n    new = pd.DataFrame({{'a': [i * {f}]}})\n    acc = pd.concat([acc, new], ignore_index=True)",
        3,
    ),
}
#: Runs through; raises at the last pass, after writing; raises at the first
#: pass, before writing (a NameError).
_RUNS = {"ok": "1", "P": "(1 if i < n - 1 else None)", "B": "factor"}
_SEQUENCES = [" ".join(s) for s in itertools.product(_RUNS, repeat=3)]


def _cell(body: str, run: str) -> str:
    return body.format(f=_RUNS[run])


def _plain(above: str, body: str, n: int, sequence: str) -> int:
    ns: dict = {"n": n}
    exec(above.format(), ns)
    for run in sequence.split():
        try:
            exec(_cell(body, run), ns)
        except (TypeError, NameError):
            pass
    return len(ns["acc"])


@pytest.mark.parametrize("sequence", _SEQUENCES)
@pytest.mark.parametrize("kind", list(_KINDS))
def test_the_value_is_the_plain_or_the_run_all_one(cash_magics, kind, sequence):
    above, body, n = _KINDS[kind]
    ns = cash_magics.shell.user_ns
    ns["n"] = n
    runs = sequence.split()
    cells = [above.format(), _cell(body, runs[0])]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    for run in runs:
        cells[1] = _cell(body, run)
        try:
            run_cash_cell(cash_magics, cells[1], cells=cells)
        except (TypeError, NameError):
            pass
    got = len(ns["acc"])
    plain = _plain(above, body, n, sequence)
    run_all = _plain(above, body, n, runs[-1])
    assert got in {plain, run_all}, (got, plain, run_all)
    assert got == run_all
