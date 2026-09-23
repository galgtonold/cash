"""After a restart, a seeded draw is restored under the key the runtime
stored it with.

The runtime keys a draw on the RNG state it consumes (the seed's lineage).
When the upstream check rebuilds that key from the simulation trace it must
fold in the same hidden read, or the lookup misses and the draw is re-run.
Re-running it pulled ``np.random.seed(0)`` into the plan ahead of the
already-scheduled ``import numpy as np``, and the cell failed with a
NameError.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.restore]

pytest.importorskip("numpy")

CELLS = [
    "import cash\n%cash_on\n",
    "import time\nimport numpy as np\nnp.random.seed(0)\n",
    # Slow enough that restoring it beats re-running it.
    "x = float(np.random.rand()) + (time.sleep(0.3) or 1)\n",
    "print('REPORT', x)\n",
]


def test_a_seeded_draw_is_restored_after_a_restart(upstream_trace, nb_runner):
    def restart_and_report(r):
        r.restart()
        r.run_cell(1)
        r.run_cell(4)

    t = upstream_trace(CELLS, restart_and_report)
    assert "REPORT 1.5488135" in nb_runner.get_output(4)
    scheduled = [r["stmt"] for r in t.rerun if r.get("event") == "schedule_reexec"]
    assert not any("np.random.rand" in s for s in scheduled), scheduled
