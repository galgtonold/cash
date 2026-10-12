"""A loop over a frame's mixed rows runs once, as in a plain kernel.

``for _, a in adj.iterrows(): pnl.loc[a['month'], a['line']] += a['amount']``
leaves ``a``, a pandas Series of object dtype. With pyplot imported, the
check of what the loop made share an object took that Series for an array of
axes and raised AttributeError ("'Series' object has no attribute 'flat'")
after the loop had run. The cell failed, a cell below repaired it by running
the loop again, and the adjustments were added twice.
"""

import pytest

pytestmark = [pytest.mark.integration]

pytest.importorskip("matplotlib")

CELLS = [
    "import cash\n%cash_on",
    "import pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    "adj = pd.DataFrame({'month': ['m1', 'm2'], 'line': ['opex', 'revenue'], 'amount': [-50.0, 12.5]})\n"
    "pnl = pd.DataFrame({'opex': [100.0, 100.0], 'revenue': [10.0, 10.0]}, index=['m1', 'm2'])",
    "for _, a in adj.iterrows():\n    pnl.loc[a['month'], a['line']] += a['amount']",
    "print('total', float(pnl.sum().sum()))",
]


def test_the_loop_adds_each_adjustment_once(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    loop_out = nb_runner.get_output(4)
    assert "Error" not in loop_out, loop_out
    assert "total 182.5" in nb_runner.get_output(5)
    nb_runner.run_cell(5)
    assert "total 182.5" in nb_runner.get_output(5)
