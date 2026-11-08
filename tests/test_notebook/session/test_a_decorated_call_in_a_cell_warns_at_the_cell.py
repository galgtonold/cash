"""A warning about a ``@cash.cache`` function called from a notebook cell
names the line in the cell as cash's notebook warnings do (``<cash>:N``),
not the pseudo-filename cash compiled the cell's statement under
(``/<cash-1a2b>``), which the reader sees nowhere else."""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell


def test_the_warning_names_the_line_in_the_cell(cash_magics, cash_instance, mock_shell):
    mock_shell.user_ns["cash_instance"] = cash_instance
    run_cash_cell(
        cash_magics,
        "class Box:\n    pass\n\n@cash_instance.cache\ndef fill(box):\n    vars(box).update(n=1)\n    return 1\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "n = 1\nfill(Box())\n")
    found = [w for w in caught if getattr(w.message, "code", None) == "IMPURE-OBSERVED-EFFECTS"]
    assert [(w.filename, w.lineno) for w in found] == [("<cash>", 2)], [str(w.message) for w in caught]
