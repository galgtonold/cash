"""A statement that changes a name only through memory it shares with
another does not make that name: the statements that did must still run.

``y = data['w'].to_numpy()`` reads ``data``'s block, so ``m.fit(y)``, which
may write ``y``, is recorded as changing ``data`` too. The replay plan took
that as ``m.fit`` producing ``data``, stopped looking above it, and after an
edit to the filter in the cleaning cell re-ran the merge on the cleaning
cell's old output: the summary counted every row, not the filtered ones.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

CELLS = [
    "import numpy as np, pandas as pd\n"
    "class Model:\n"
    "    def fit(self, a):\n"
    "        self.total = float(a.sum())\n"
    "        return self\n",
    "FILTER = None\n",
    "cust = pd.DataFrame({'r': list('abab'), 'v': [1, 2, 3, 4]})\n"
    "cust['v2'] = cust['v'] * 2\n"
    "if FILTER:\n"
    "    cust = cust[cust['r'] == FILTER]\n",
    "data = cust.copy()\ndata['w'] = data['v2'] + 1\n",
    "y = data['w'].to_numpy()\nm = Model()\nm.fit(y)\n",
    "print('rows', len(data), 'total', m.total)\n",
]


def test_an_edit_above_reaches_a_name_a_later_call_shares(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "rows 4 total 24.0" in nb_runner.get_output(6)

    nb_runner.set_cell_source(2, "FILTER = 'a'\n")
    nb_runner.run_cell(6)

    # Rows 'a' only: v = 1, 3, so w = 3 + 7.
    assert "rows 2 total 10.0" in nb_runner.get_output(6)
