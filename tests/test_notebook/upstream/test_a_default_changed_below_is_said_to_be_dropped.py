"""A function's default, changed in place by a cell below, is said to be dropped.

``def draw(k, rows=BIG)`` reads ``BIG`` through its default, not by name.
Re-run on its own after a cell below ran ``BIG.append(...)``, the calling
cell gets ``draw`` and ``BIG`` as the cells above leave them, as a run from
the top does, and computes without the change. That is the result a run
from the top gives; but where the cell names ``BIG`` itself NOTEBOOK-CHANGED-
BELOW says so, and through the default it said nothing.
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.core

SETUP = (
    "import time\nBIG = [{'id': i} for i in range({N})]\n"
    "def draw(k, rows=BIG):\n    time.sleep(0.05)\n    return len(rows)\n"
)

CHANGES = {
    "append": "BIG.append({'id': -1})",
    "extend": "BIG.extend([1, 2])",
    "item-set": "BIG[0] = 'x'",
}


def _rerun_after(cash_magics, cells, reader, change):
    run_cash_cell(cash_magics, cells[0], cells=cells)
    for _ in range(3):
        run_cash_cell(cash_magics, reader, cells=cells)
    run_cash_cell(cash_magics, change, cells=cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, reader, cells=cells)
    return [str(w.message) for w in caught if "NOTEBOOK-CHANGED-BELOW" in str(w.message)]


@pytest.mark.parametrize("size", [3, 20_000], ids=["small", "big"])
@pytest.mark.parametrize("change", sorted(CHANGES))
def test_a_default_changed_below_is_said_to_be_dropped(cash_magics, size, change):
    setup = SETUP.replace("{N}", str(size))
    reader = "y = draw(1)\nprint(y)"
    said = _rerun_after(cash_magics, [setup, reader, CHANGES[change]], reader, CHANGES[change])
    assert cash_magics.shell.user_ns["y"] == size, "not the value a run from the top gives"
    assert said, "the change below was dropped without a word"
    assert "reads 'draw', which holds 'BIG', which cell 3 below changed in place" in said[0], said[0]


def test_an_object_a_default_holds_changed_below_is_said_to_be_dropped(cash_magics):
    setup = "import time\nclass Box:\n    pass\nB = Box()\nB.n = 1\ndef read(k, box=B):\n    time.sleep(0.05)\n    return box.n"
    reader = "y = read(1)"
    said = _rerun_after(cash_magics, [setup, reader, "B.n = 2"], reader, "B.n = 2")
    assert cash_magics.shell.user_ns["y"] == 1, "not the value a run from the top gives"
    assert said, "the change below was dropped without a word"


def test_a_result_sharing_elements_with_a_default_is_not_said_to_be_held(cash_magics):
    """``z = draw(2)`` below is a new list of ``BIG``'s own items, so it
    shares objects with ``BIG``; ``z.append(9)`` changes only ``z``, which
    ``draw`` does not hold, and nothing the calling cell reads is dropped."""
    setup = SETUP.replace("{N}", "3").replace("return len(rows)", "return list(rows[:2])")
    reader = "y = draw(1)"
    change = "z = draw(2)\nz.append(9)"
    said = _rerun_after(cash_magics, [setup, reader, change], reader, change)
    assert cash_magics.shell.user_ns["y"] == [{"id": 0}, {"id": 1}]
    assert not said, said
