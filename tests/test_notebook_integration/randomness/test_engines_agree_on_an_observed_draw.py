"""A statement that draws from a seeded global RNG without saying so.

``m.train()`` spells no ``np.random``, so the runtime learns that it draws by
watching it run. That observation joins the statement's cache key, so a
re-seed re-keys it, but not its output lineage. The simulation must build the
lineage the same way, or a plain top-to-bottom run sees ``m`` "changed" and
re-runs its producers when the next cell reads it.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

pytest.importorskip("numpy")

CELLS = [
    "import cash\n%cash_on\n",
    "import numpy as np\nnp.random.seed(0)\n",
    "class M:\n    def train(self):\n        self.w = float(np.random.rand())\n        return self\nm = M()\n",
    "m.train()\n",
    "print('REPORT', m.w)\n",
]


def test_engines_agree_on_a_draw_only_the_runtime_saw(upstream_trace, nb_runner):
    t = upstream_trace(CELLS, lambda r: r.run_cell(5))
    records = t.run_all + t.rerun
    assert not [(r["cell_idx"], r["vars"]) for r in records if r.get("event") == "lineage_disagreement" and r["vars"]]
    assert not [r["stmt"] for r in records if r.get("event") == "schedule_reexec"]
    assert "REPORT 0.5488135" in nb_runner.get_output(5)
