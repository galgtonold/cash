"""A helper that saves a chart runs every time, as the inline savefig does.

Round 22 (r22s3): ``save(fig, "suspect.png")`` -- a notebook helper doing
``fig.tight_layout(); fig.savefig(...); plt.close(fig)`` -- was served from
the statement cache on a warm Restart & Run All, so the deck went out
without the chart, 3/3. Whether it happened depended on whether the save
took long enough to be persisted; persistence is forced here so the
failure does not depend on the machine.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

SETUP = """import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path("out")
OUT.mkdir(exist_ok=True)
df = pd.DataFrame(np.arange(72.0).reshape(12, 6), columns=list("ABCDEF"))

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=40)
    plt.close(fig)

def log(msg):
    open(OUT / "log.txt", "a").write(msg + "\\n")
"""
CHART = """fig, axes = plt.subplots(2, 3, figsize=(8, 5))
for ax, col in zip(axes.flat, df.columns):
    df[col].plot(ax=ax)
save(fig, "grid.png")
"""


def test_warm_restart_and_run_all_writes_the_chart(nb_runner):
    work = Path(nb_runner.work_dir)
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, CHART])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert (work / "out" / "grid.png").exists()
    for cycle in range(2):
        (work / "out" / "grid.png").unlink()
        nb_runner.restart()
        nb_runner.enable_persist()
        nb_runner.run_all()
        assert (work / "out" / "grid.png").exists(), f"warm run {cycle + 1} wrote no chart"


def test_a_helper_that_only_appends_still_caches(nb_runner):
    """Control: an append is a skippable side effect, like a print -- a
    function that logs keeps caching."""
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            SETUP,
            "import time\ndef slow(v):\n    log('slow')\n    time.sleep(0.3)\n    return v * 2\nx = slow(21)",
            "print('x =', x)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    log = Path(nb_runner.work_dir) / "out" / "log.txt"
    assert log.read_text().split() == ["slow"]
    nb_runner.restart()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert "x = 42" in nb_runner.get_output(4)
    assert log.read_text().split() == ["slow"], "slow() re-ran instead of coming from the cache"
