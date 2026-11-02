"""The text a cell ran with before an edit is never run as a statement
above it.

The edited cell's loop held objects of ``data``, rebuilt because its producer
was edited too, so every name the loop bound was to be rebuilt with it. No
cell above binds them: the statement that last did was the cell's own old
loop, which the planner re-applied as an unsaved edit before the new text
ran, printing the old lines first.
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell

DATA = "data = [({uid!r}, [('a', 1), ('a', 2), ('a', 3)]), ('u2', [('b', 1), ('c', 2), ('c', 3)])]"
LOOP = (
    "total = len(data)\n"
    "k = 0\n"
    "for uid, actions in data:\n"
    "    for i in range(len(actions) - 2):\n"
    "        a_name = actions[i][0]\n"
    "        if a_name == actions[i+1][0] and a_name == actions[i+2][0]:\n"
    "            k += 1\n"
    "            print({word!r}, uid)\n"
    "            break\n"
    "k / total"
)


def test_only_the_new_text_prints(cash_magics, capsys):
    cells = [DATA.format(uid="u1"), LOOP.format(word="old")]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        capsys.readouterr()
        cells = [DATA.format(uid="u9"), LOOP.format(word="new")]
        run_cash_cell(cash_magics, cells[1], cells=cells)

    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith(("old", "new"))]
    assert printed == ["new u9"]
    assert cash_magics.shell.user_ns["k"] == 1
