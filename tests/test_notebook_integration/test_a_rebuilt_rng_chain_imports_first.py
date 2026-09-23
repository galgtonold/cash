"""After a restart, a re-run draw's seed runs after the import it needs.

A cheap draw may be re-run rather than restored. Its seed is then pulled into
the plan ahead of it, and the seed reads ``np``, which the plan had already
scheduled -- after the seed. The cell failed with ``UpstreamStateError: ...
name 'np' is not defined``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.restore]

pytest.importorskip("numpy")

CELLS = [
    "import cash\n%cash_on\n",
    "import numpy as np\nnp.random.seed(0)\n",
    "x = float(np.random.rand()) + 1\n",
    "print('REPORT', x)\n",
]


def test_a_seed_pulled_into_the_plan_runs_after_its_import(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)
    assert "REPORT 1.5488135" in nb_runner.get_output(4)
