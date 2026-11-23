"""A pandas array holding an object's ``id()`` runs again, as a frame or a
series of it does.

``arr = pd.array(keys, dtype='string')`` was read by the identity check as
a library object it does not look inside, so the ``str(id(x))`` in it was
stored and served after a rerun with the old process's address, where plain
Python spells the new object's.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement.identity_holders import identity_held_reason
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")

MAKE = "class Q:\n    pass\nclass P:\n    def __init__(self):\n        self.child = Q()\np = P()"


@pytest.mark.parametrize(
    "make",
    [
        "pd.array(['a', str(id(held))], dtype='string[python]')",
        "pd.array(['a', str(id(held))], dtype='string[pyarrow]')",
        "pd.array([1, id(held)], dtype='Int64')",
        "pd.array([1, id(held)], dtype='int64[pyarrow]')",
    ],
)
def test_a_pandas_array_of_ids_is_found(make):
    if "pyarrow" in make:
        pytest.importorskip("pyarrow")
    held = object()
    arr = eval(make, {"pd": pd, "held": held})
    assert identity_held_reason("arr", arr, {"arr": arr, "held": held}) is not None
    # Positive control: the same array without the id() is stored.
    plain = eval(make.replace("id(held)", "12"), {"pd": pd})
    assert identity_held_reason("plain", plain, {"plain": plain, "held": held}) is None


def test_a_text_array_spells_the_new_object(cash_magics, cash_instance):
    pytest.importorskip("pyarrow")
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    hold = (
        "import pandas as pd\n"
        "arr = pd.array([f'S{i % 97:03d}-{i:05d}' if i != 4321 else str(id(p.child)) for i in range(5000)], "
        "dtype='string[pyarrow]')"
    )
    cells = [MAKE, hold, "r = arr[4321] == str(id(p.child))  # @cash:no-cache"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] is True
