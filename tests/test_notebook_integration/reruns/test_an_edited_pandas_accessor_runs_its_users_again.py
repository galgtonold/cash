"""A pandas accessor defined in a cell, edited and run again, runs the
statements that use it again, in a kernel.

``res = df.fx.slow()`` names neither the accessor class nor its method, so
after ``* 2`` became ``* 3`` in the accessor's cell the old result was
served, and ``kept = df.fx.slow()`` after showing it was handed the old
shown value. A fresh frame (``df.copy()``) each time: pandas before 3 keeps
the accessor object it made on a frame, so plain Python too hands the same
``df`` the old accessor.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration]

ON = "import cash\n%cash_on"
DATA = "import time\nimport numpy as np, pandas as pd\ndf = pd.DataFrame({'a': np.arange(5.0)})"
ACCESSOR = (
    "@pd.api.extensions.register_dataframe_accessor('fx')\n"
    "class FX:\n"
    "    def __init__(self, d):\n"
    "        self._d = d\n"
    "    def slow(self):\n"
    "        time.sleep(0.6)\n"
    "        return self._d * {k}"
)


@pytest.mark.parametrize(
    "use", ["res = df.copy().fx.slow()", "df.copy().fx.slow()\n"], ids=["assigned", "shown-then-kept"]
)
def test_the_edited_accessor_gives_the_new_result(nb_runner, use):
    nb_runner.create_notebook([ON, DATA, ACCESSOR.format(k=2), use.strip()])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, ACCESSOR.format(k=3))
    nb_runner.run_cell(3)
    nb_runner.set_cell_source(4, "res = df.copy().fx.slow()")
    nb_runner.run_cell(4)

    assert nb_runner.peek("res.a.tolist()") == "[0.0, 3.0, 6.0, 9.0, 12.0]"
