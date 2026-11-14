"""A loop growing a list that holds itself through a pandas object column,
a dict in one or an object NumPy array never hangs, and prints what plain
Python prints after re-runs, a restart and a change through the column.

Sizing such a list for the store walked the column's cells afresh, which
reached the list and its frames again without end; and the loop was
restored into the list, whose stored frames then held a copy of it.
"""

import pytest

pytestmark = [pytest.mark.timeout(300), pytest.mark.loops]

_MAKE = (
    "import time\nimport numpy as np\nimport pandas as pd\n"
    "def objects(*values):\n    cells = np.empty(len(values), dtype=object)\n"
    "    for j, value in enumerate(values):\n        cells[j] = value\n    return cells\n"
    "L = [pd.DataFrame({'a': [1, 2]})]"
)

#: How each shape keeps the list, and how to get the list back from it.
_SHAPES = {
    "column": ("pd.DataFrame({'o': [L, i]})", "x['o'][0]"),
    "dict_in_column": ("pd.DataFrame({'o': [{'k': L}, i]})", "x['o'][0]['k']"),
    "object_array": ("objects(L, i)", "x[0]"),
}


def _cells(shape: str) -> list[str]:
    make, back = _SHAPES[shape]
    loop = f"for i in range(4):\n    time.sleep(0.05)\n    L.append({make})"
    last = back.replace("x", "L[-1]")
    change = f"{last}.append('through the column')"
    read = f"print(len(L), L[-1], all(({back}) is L for x in L[1:-1]), [type(x).__name__ for x in L])"
    return [_MAKE, loop, change, read]


def _plain(cells: list[str]) -> str:
    import contextlib
    import io

    out = io.StringIO()
    namespace: dict = {}
    with contextlib.redirect_stdout(out):
        for cell in cells:
            exec(cell, namespace)  # the cells' own code, run as plain Python
    return out.getvalue().strip()


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_list_matches_plain(nb_runner, shape):
    cells = _cells(shape)
    expected = _plain(cells)
    assert "True" in expected
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for again in ("first", "again", "restart"):
        if again == "restart":
            nb_runner.restart()
        nb_runner.run_all()
        assert nb_runner.get_output(4).strip() == expected, again
