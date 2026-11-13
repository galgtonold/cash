"""A cell re-run after an attempt that raised before it rebound a name
does not rebuild that name: the attempt left it as it found it, which is
already what the cells above give.

An accumulator (``acc = acc + new`` in a loop) re-run while its error is
fixed: each attempt raises before the loop reaches the rebinding. The
upstream check took every next attempt for a re-run from the object the
last one started from, read ``acc`` as the cell's own stale output and
rebuilt it from the cells above each time. The value was right (the one
Restart & Run All gives); the rebuild was not needed, and for a costly
producer it is what made each attempt slow.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

_KINDS = {
    # Large from the start, so it is followed by its identity.
    "list": ("acc = [0] * 2000", "for i in range(n):\n    new = [i * {f}]\n    acc = acc + new", 3000),
    "frame": (
        "import pandas as pd\nacc = pd.DataFrame({{'a': [0] * 2000}})",
        "for i in range(n):\n    new = pd.DataFrame({{'a': [i * {f}]}})\n    acc = pd.concat([acc, new], ignore_index=True)",
        1000,
    ),
}


@pytest.mark.parametrize("kind", list(_KINDS))
def test_attempts_that_raise_before_the_rebinding_keep_the_object(cash_magics, kind):
    above, body, n = _KINDS[kind]
    ns = cash_magics.shell.user_ns
    ns["n"] = n
    cells = [above.format(), body.format(f="(1 if i < n - 1 else None)")]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    # Raises part way: the next run starts again from the cells above.
    with pytest.raises(TypeError):
        run_cash_cell(cash_magics, cells[1], cells=cells)
    held = []
    for typo in ("factor", "factor2", "factor3"):
        cells[1] = body.format(f=typo)
        with pytest.raises(NameError):
            run_cash_cell(cash_magics, cells[1], cells=cells)
        held.append(ns["acc"])
    assert all(len(value) == 2000 for value in held)
    # Rebuilt once, for the first of them; the others kept it.
    assert held[1] is held[0] and held[2] is held[0]


def test_a_rebinding_to_the_object_a_helper_changed_in_place_is_still_reset(cash_magics):
    """The control: ``acc = grow(acc)`` binds the very object ``grow``
    changed in place, then the cell raises. The run did write to the
    object it started from, so the next run starts again from what the
    cells above make, as Restart & Run All does."""
    ns = cash_magics.shell.user_ns
    cells = [
        "def grow(rows):\n    rows.append([len(rows)])\n    return rows",
        "acc = [[i] for i in range(3000)]",
        "acc = grow(acc)\nchecked = 1 / flag",
    ]
    ns["flag"] = 1
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns["flag"] = 0
    with pytest.raises(ZeroDivisionError):
        run_cash_cell(cash_magics, cells[2], cells=cells)
    assert len(ns["acc"]) == 3001
    ns["flag"] = 1
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert len(ns["acc"]) == 3001
