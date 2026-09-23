"""A loop that only reuses a variable's name is not re-run with its producer.

Round 29, r29s4 (BLOCKING, 1/1 replaying their notebook, reproduced on HEAD
with a trace): after an upstream fix that dropped some rows, the export cell's
repair re-ran ``for r in sorted(obs.run.unique()): ...`` -- it builds the run
summary the export writes. A later producer of ``r`` then had to run too
(``_complete_later_producers``, there for ``results = {}`` followed by the loop
that fills it): the QC-histogram loop, which also has a ``for r``, and from its
``ax`` the UMAP loop ``for ax, col in zip(axes, ...)``. That loop drew the new
labels over the old embedding -- nothing needed the embedding, so nothing
rebuilt it -- and matplotlib raised on the length mismatch, surfaced as an
UpstreamStateError naming a cell with nothing wrong in it. With the row count
unchanged it would have drawn silently.

A later statement that binds a name WITHOUT reading it does not continue what
the earlier producer built; it starts over, and owes it nothing.
"""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\n"
    "import time\nimport numpy as np\nimport pandas as pd\n"
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    "N = 300\nobs = pd.DataFrame({'g': np.arange(N) % 3, 'run': np.arange(N) % 2})",
    "summ = []\nfor r in sorted(obs.run.unique()):\n    summ.append((int(r), int((obs.run == r).sum())))",
    "def embed(z):\n    time.sleep(0.3)\n    return z * 2.0\n"
    "emb = embed(np.c_[np.arange(len(obs)), np.arange(len(obs)) % 7].astype(float))",
    "fig, axes = plt.subplots(1, 2, figsize=(4, 2))\n"
    "for ax, r in zip(axes, ['g', 'run']):\n"
    "    codes = pd.Categorical(obs[r]).codes\n"
    "    ax.scatter(emb[:, 0], emb[:, 1], c=codes, s=1)\n"
    "plt.close(fig)",
    "print('summary', summ)",
]


def test_the_chart_loop_is_not_re_run_for_a_shared_loop_name(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "summary [(0, 150), (1, 150)]" in nb_runner.get_output(6), nb_runner.get_raw_output(6)

    nb_runner.set_cell_source(2, "N = 280\nobs = pd.DataFrame({'g': np.arange(N) % 3, 'run': np.arange(N) % 2})")
    nb_runner.run_cell(6)
    raw = nb_runner.get_raw_output(6)
    assert "UpstreamStateError" not in raw and "inconsistent" not in raw, raw
    assert "summary [(0, 140), (1, 140)]" in nb_runner.get_output(6), raw
