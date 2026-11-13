"""A cell that builds on a value from above, re-run and then edited so it no
longer changes that value, leaves it at the value Restart & Run All gives.

Every re-run of such a cell (``acc = pd.concat([acc, new])`` in a loop)
starts from what the cells above give, so the value is one only the cell's
text makes. Edited not to change it any more (the line commented out, or
rebound without reading it by a line that raises first), the cell kept that
value: ok, ok, commented out left 3 rows where a plain kernel has 6 and Run
All, with the cell as it now stands, 0.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

_LOOP = "for i in range(3):\n    new = NEW\n"
_KINDS = {
    "concat": (
        "import pandas as pd\nacc = pd.DataFrame({'a': []})",
        "pd.DataFrame({'a': [i * FACTOR]})",
        "acc = pd.concat([acc, new], ignore_index=True)",
    ),
    "loc": ("import pandas as pd\nacc = pd.DataFrame({'a': []})", "i * FACTOR", "acc.loc[len(acc)] = [new]"),
    "list-plus": ("acc = []", "[i * FACTOR]", "acc = acc + new"),
    "list-append": ("acc = []", "i * FACTOR", "acc.append(new)"),
}
#: The writing line as it is; commented out; a binding that does not read
#: the value and raises before it binds.
_FORMS = {"on": "    WRITE", "off": "#    WRITE", "bind": "    acc = no_such_helper(new)"}
#: Runs through; raises part way, after a write; raises before any.
_FACTORS = {"ok": "1", "P": "(1 if i < 1 else None)", "B": "factor"}
_SEQUENCES = [
    "on:ok on:ok off:ok",
    "on:ok on:P off:ok",
    "on:P on:P off:B",
    "on:B on:ok on:ok off:P",
    "on:ok off:ok on:ok off:ok",
    "on:ok on:ok bind:ok",
    "on:ok on:P bind:ok",
]
_ERRORS = (TypeError, NameError)


def _cell(kind: str, step: str) -> str:
    _, new, write = _KINDS[kind]
    form, run = step.split(":")
    loop = _LOOP.replace("NEW", new.replace("FACTOR", _FACTORS[run]))
    return loop + _FORMS[form].replace("WRITE", write)


def _plain(cells: list[str]) -> int:
    ns: dict = {}
    for code in cells:
        try:
            exec(code, ns)
        except _ERRORS:
            pass
    return len(ns["acc"])


@pytest.mark.parametrize("sequence", _SEQUENCES)
@pytest.mark.parametrize("with_lineage", [True, False], ids=["tracked-above", "built-before-cash"])
@pytest.mark.parametrize("kind", list(_KINDS))
def test_the_value_is_the_run_all_one(cash_magics, kind, with_lineage, sequence):
    above = _KINDS[kind][0]
    steps = sequence.split()
    cells = [above, _cell(kind, steps[0]), "probe = len(acc)"]
    ns = cash_magics.shell.user_ns
    if with_lineage:
        run_cash_cell(cash_magics, above, cells=cells)
    else:
        exec(above, ns)
    ran = [above]
    for step in steps:
        cells[1] = _cell(kind, step)
        ran.append(cells[1])
        try:
            run_cash_cell(cash_magics, cells[1], cells=cells)
        except _ERRORS:
            pass
    run_cash_cell(cash_magics, cells[2], cells=cells)
    plain, run_all = _plain(ran), _plain(cells[:2])
    assert ns["probe"] in {plain, run_all}, (ns["probe"], plain, run_all)
    assert ns["probe"] == run_all


def test_a_value_the_notebook_rebinds_afterwards_is_kept(cash_magics):
    """The control: the cell above ran again after the accumulating cell, so
    the value is the one it makes, and the edited cell keeps it."""
    above = "acc = [0] * 3000"
    body = "for i in range(3):\n    acc = acc + [i]"
    cells = [above, body, "probe = len(acc)"]
    for code in (above, body, body, above):
        run_cash_cell(cash_magics, code, cells=cells)
    cells[1] = "for i in range(3):\n#    acc = acc + [i]\n    pass"
    held = cash_magics.shell.user_ns["acc"]
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert cash_magics.shell.user_ns["acc"] is held
