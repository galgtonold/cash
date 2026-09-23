"""A cell cash runs sees what IPython puts into builtins while it runs a cell.

With cash on, ``pd.get_option("display.max_columns")`` was 0
instead of 20, so tables printed differently from the uncached kernel. pandas
decides at import whether it is in a terminal by calling ``get_ipython()``; IPython
puts that into ``builtins`` only while it runs a cell (its builtin trap), and cash
runs a cell's statements itself, outside it. Any library detecting a notebook
that way was fooled the same way.
"""

import pytest

pytestmark = [pytest.mark.integration]


def test_get_ipython_is_a_builtin_inside_a_cash_cell(nb_runner):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import builtins\nprint('TRAP', hasattr(builtins, 'get_ipython'), hasattr(builtins, 'display'))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "TRAP True True" in nb_runner.get_output(2), nb_runner.get_output(2)
