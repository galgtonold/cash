"""Code in a cell reads the execution count plain IPython gives it.

IPython 9 raises ``execution_count`` as a cell starts, so inside ``In[N]`` it
reads ``N + 1`` (IPython 8 raises it after the cell: ``N``). cash runs the
statements itself before IPython's stand-in, so under IPython 9 the count read
inside a cell was one behind plain's, and ``Out.get(execution_count - 1)``
found the cell above instead of nothing.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

READ = (
    "import IPython\n"
    "lead = 1 if IPython.version_info >= (9,) else 0\n"
    "n = get_ipython().execution_count\n"
    "print('count', n == len(In) - 1 + lead)"
)


def test_the_count_inside_a_cell_is_plain_s(nb_runner):
    nb_runner.create_notebook(["%cash_badge off", "1 + 1", READ])
    nb_runner.start_kernel()

    for _ in range(2):  # computed, then a rerun
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count True" in out, out
