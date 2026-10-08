"""A function that keeps state between calls carries it from cell to cell.

A factory closure (``log = make_log()``), a ``nonlocal`` counter and a mutable
default argument, each called once in three cells: plain Jupyter's Run All
gives ``1, 2, 3``. Cash re-ran the line that made the function before every
cell calling it and gave ``1, 1, 1``, with no warning. An isolated re-run of
a calling cell still starts from fresh state.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

DEFS = (
    "def make_log():\n    seen = []\n    def log(msg):\n        seen.append(msg)\n        return len(seen)\n    return log\n"
    "def make_counter():\n    n = 0\n    def nxt():\n        nonlocal n\n        n += 1\n        return n\n    return nxt\n"
    "def take(item, acc=[]):\n    acc.append(item)\n    return list(acc)"
)
CELLS = [
    DEFS,
    "log = make_log()\nnext_id = make_counter()",
    "a = log('loaded')\nrun1 = next_id()\nt1 = take('x')",
    "b = log('cleaned')\nrun2 = next_id()\nt2 = take('y')",
    "c = log('fitted')\nrun3 = next_id()\nt3 = take('z')",
    "print((a, b, c), (run1, run2, run3), t3)",
]


def test_run_all_carries_the_state_from_cell_to_cell(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(6).strip() == "(1, 2, 3) (1, 2, 3) ['x', 'y', 'z']"


def test_an_isolated_re_run_of_a_calling_cell_starts_fresh(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cells([3])
    assert nb_runner.peek("(a, run1, t1)") == "(1, 1, ['x'])"
