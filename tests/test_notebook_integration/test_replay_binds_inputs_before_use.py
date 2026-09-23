"""After a restart, a replayed statement finds every name it reads already bound.

Round 22: two testers were refused with ``UpstreamStateError: name '...' is
not defined`` on the first jump after a restart.

* r22s3: ``import glob`` in cell 2 and again in cell 3. The replay resolved
  ``glob`` to its last producer, cell 3's import, which runs after cell 2's
  ``files = glob.glob(...)``. Every jump downstream was refused until the
  imports were merged into one cell.
* r22s4: a feature cell of cheap statements (``keys = ...drop_duplicates()``
  then ``panel = keys.merge(...)``): nothing was persisted, and the replay ran
  the merge without the line above it that binds ``keys``.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _stdout(runner, cell):
    outs = runner.nb.cells[cell - 1].get("outputs", [])
    return "".join(
        o["text"] if isinstance(o["text"], str) else "".join(o["text"])
        for o in outs
        if o.get("output_type") == "stream" and o.get("name") == "stdout"
    )


def test_an_import_repeated_in_a_later_cell(nb_runner):
    work = Path(nb_runner.work_dir)
    (work / "files").mkdir()
    for name in ("a.dat", "b.dat", "c.txt"):
        (work / "files" / name).write_text("x")
    cells = [
        "import cash\n%cash_on",
        "from pathlib import Path\nDATA = Path('files')",
        "files = sorted(str(p) for p in DATA.glob('*.dat'))",
        "extra = sorted(str(p) for p in DATA.glob('*.txt'))",
        "print('N', len(files), len(extra))",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "N 2 1" in _stdout(nb_runner, 5)
    nb_runner.restart()
    nb_runner.set_cell_source(3, "import glob\nfiles = sorted(glob.glob(str(DATA / '*.dat')))")
    nb_runner.set_cell_source(4, "import glob\nextra = sorted(glob.glob(str(DATA / '*.txt')))")
    nb_runner.run_cell(1)
    nb_runner.run_cell(5)
    assert "N 2 1" in _stdout(nb_runner, 5), nb_runner.get_raw_output(5)[-800:]


#: r22s4's feature cell, as the tester wrote it. The size matters: smaller,
#: the frames are not persisted and the whole cell simply re-runs.
_DATA = """import numpy as np
import pandas as pd
rng = np.random.default_rng(0)
dates = pd.date_range("2023-09-01", "2026-08-30", freq="D")
regions, families = [f"R{i}" for i in range(8)], [f"F{j}" for j in range(40)]
idx = pd.MultiIndex.from_product([regions, families, dates], names=["region", "family", "date"])
daily = pd.DataFrame({"demand": rng.poisson(50, len(idx)).astype(float)}, index=idx).reset_index()
print(daily.shape)
"""
_FEATURES = """HORIZON = 28
last_day = daily.date.max()
all_dates = pd.date_range(daily.date.min(), last_day + pd.Timedelta(days=HORIZON), freq="D")
keys = daily[["region", "family"]].drop_duplicates()
panel = keys.merge(pd.DataFrame({"date": all_dates}), how="cross")
panel = (panel.merge(daily, on=["region", "family", "date"], how="left")
              .sort_values(["region", "family", "date"]).reset_index(drop=True))
g = panel.groupby(["region", "family"]).demand
panel["lag_28"] = g.shift(28)
panel["rmean_7"] = g.transform(lambda s: s.shift(HORIZON).rolling(7, min_periods=4).mean())
panel["dow"] = panel.date.dt.dayofweek
"""
_SUMMARY = """summary = panel.groupby("region")[["lag_28", "rmean_7"]].mean().round(6)
print("RESULT", summary.lag_28.sum().round(4), summary.rmean_7.sum().round(4), len(panel))
"""


def test_a_feature_cell_after_a_restart(nb_runner):
    """Measured: the replay scheduled only the ``panel`` statements. After
    the restart the recorded lineages of ``keys``, ``all_dates`` and ``g``
    still agree with the simulation, so nothing asked for them -- but none
    of them is bound."""
    nb_runner.create_notebook(["import cash\n%cash_on", _DATA, _FEATURES, _SUMMARY])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = _stdout(nb_runner, 4)
    assert "RESULT" in first
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)
    assert _stdout(nb_runner, 4) == first, nb_runner.get_raw_output(4)[-800:]
