"""A ``@cash.cache`` function always stores its result, and storing it runs
the result's own ``__reduce__``, as outside a notebook. In a notebook that
reducer writing a variable (``CFG['k'] += 1``) changes what plain Python
leaves alone, so it must not go unsaid.

``w = make(1)`` returning a ``W`` whose ``__reduce__`` counts in ``CFG``:
the statement's own pickle refused ``W``, whose warning moved the mark on
before the reducers the call ran were read, so the only warning said cash
did not pickle it, and ``CFG`` changing was never named.
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell

SETUP = (
    "import cash\nCFG = {'k': 0}\n"
    "class W:\n    def __reduce__(self):\n        CFG['k'] += 1\n        return (W, ())\n"
    "@cash.cache\ndef make(i):\n    return W()\n"
)


def test_the_warning_names_the_variable_the_reducer_changed(cash_magics):
    cells = [SETUP, "w = make(1)", "k = CFG['k']"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[1], cells=cells)
    said = [str(w.message) for w in seen if "NOTEBOOK-REDUCE-WROTE" in str(w.message)]
    assert len(said) == 1, said
    assert "CFG counted as changed" in said[0]
    assert "did not pickle" not in said[0]
