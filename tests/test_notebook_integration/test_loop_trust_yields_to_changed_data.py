"""A loop over data that changed is not trusted -- and neither is what it picked.

Round 22 (r22s4): a backtest built its table with ``rows.append(...)`` over a
grid and chose ``BEST_K`` from it. A new file landed in the globbed folder;
the forecast cell re-read the data but was served the ``BEST_K`` tuned on the
old data, and the metrics cell printed the old backtest into a fresh file.
A file change is not a code edit, so loop trust -- which lets a loop-built
value whose lineage disagrees with the simulation stand -- still applied to
the loop and everything derived from it. Written as a comprehension, the
same cell was always right.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

LOAD = '''import glob, os
import pandas as pd
files = sorted(glob.glob(os.path.join("rdata", "part_*.csv")))
raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
def score(frame, k):
    open("calls.log", "a").write("score\\n")
    pred = frame.y.rolling(k, min_periods=1).mean().shift(1)
    return float((frame.y - pred).abs().mean())
'''
BACKTEST = '''rows = []
for k in (2, 5, 20):
    rows.append(dict(k=k, err=score(raw, k)))
bt = pd.DataFrame(rows)
BEST_K = int(bt.loc[bt.err.idxmin(), "k"])
'''
FINAL = '''forecast = float(raw.y.rolling(BEST_K).mean().iloc[-1])
print("RESULT BEST_K", BEST_K, "forecast", round(forecast, 4), "rows", len(raw))
'''


def _part(d: Path, n: int) -> None:
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    t = np.arange(300)
    if n == 1:      # flat and noisy: heavy smoothing wins
        pd.DataFrame({"t": t, "y": 10 + rng.normal(0, 3, 300)}).to_csv(d / "part_1.csv", index=False)
    else:           # a steep ramp: light smoothing wins
        pd.DataFrame({"t": 300 + t, "y": 10 + 3.0 * t + rng.normal(0, 1, 300)}).to_csv(
            d / "part_2.csv", index=False)


def _scores(work: Path) -> int:
    log = work / "calls.log"
    return log.read_text().split().count("score") if log.exists() else 0


@pytest.mark.parametrize("new_file", [True, False], ids=["new_file", "unchanged"])
def test_a_new_file_reruns_the_loop_that_tuned_the_parameter(nb_runner, new_file):
    work = Path(nb_runner.work_dir)
    _part(work / "rdata", 1)
    nb_runner.create_notebook(["import cash\n%cash_on", LOAD, BACKTEST, FINAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "RESULT BEST_K 20 " in nb_runner.get_output(4)
    before = _scores(work)
    if new_file:
        _part(work / "rdata", 2)
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    if new_file:
        assert "RESULT BEST_K 2 forecast 906.2375 rows 600" in out, out
        assert _scores(work) - before == 3
    else:
        assert "RESULT BEST_K 20 forecast 10.1284 rows 300" in out, out
        assert _scores(work) == before, "an unchanged loop lost its trust"
