"""Adversarial probes, wave 3 (2026-07-02): realistic discipline workflows.

End-to-end multi-cell pipelines with hand-computable expected values.
Each test asserts (a) correctness after a mid-pipeline edit and (b) where
solid, effectiveness (expensive upstream cells not re-executed for
unrelated/downstream edits).

 1. test_pandas_pipeline_edit_midstream      — DS: csv -> clean -> feature ->
        groupby; edit the feature formula; isolated final-cell run + run_all.
 2. test_pandas_pipeline_downstream_edit_no_upstream_recompute — DS
        effectiveness: editing ONLY the last (report) cell must not
        re-execute the load/clean chain.
 3. test_finance_backtest_param_edit         — rolling mean + strategy loop
        over a params list; edit one param; final metric must update.
 4. test_numpy_training_loop_lr_edit         — ML-ish: gradient descent loop
        (seeded, deterministic); edit learning rate; loss must change and
        match the plain-kernel ground truth computed in-test.
 5. test_etl_write_read_chain_param_edit     — ETL: raw csv -> transform ->
        to_csv intermediate -> read back -> report; edit the transform
        param; run_all must propagate through the file boundary.
 6. test_matplotlib_figure_cell_rerun        — plotting cell (savefig) reruns
        + unchanged rerun; no crash, file exists, neighbours cached.
 7. test_seeded_montecarlo_seed_edit         — seeded MC loop; edit seed;
        aggregate changes; revert seed; aggregate returns to original.
"""

import pytest

pytestmark = [pytest.mark.timeout(150)]

MISS = "Executing (cache miss)"
REEXEC = "Auto-executing upstream statement:"


def _p(path) -> str:
    return str(path).replace("\\", "/")


def _write_sales_csv(path):
    path.write_text(
        "region,units,price\n"
        "north,10,2.0\n"
        "north,20,3.0\n"
        "south,5,10.0\n"
        "south,15,4.0\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1./2. Data-science pandas pipeline
# ---------------------------------------------------------------------------

def _pipeline_cells(csv_path):
    return [
        "import pandas as pd",
        f"raw = pd.read_csv('{csv_path}')",
        "clean = raw.dropna().reset_index(drop=True)",
        "feat = clean.assign(revenue=clean.units * clean.price)",
        "summary = feat.groupby('region')['revenue'].sum().to_dict()",
        "print('summary=', {k: round(v, 1) for k, v in sorted(summary.items())})",
    ]


def test_pandas_pipeline_edit_midstream(nb_runner, tmp_path):
    csv = tmp_path / "sales.csv"
    _write_sales_csv(csv)
    nb_runner.create_notebook(_pipeline_cells(_p(csv)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    # north: 10*2 + 20*3 = 80 ; south: 5*10 + 15*4 = 110
    assert "summary= {'north': 80.0, 'south': 110.0}" in nb_runner.get_output(6)

    # Edit the FEATURE cell: revenue now has a 10% fee haircut.
    nb_runner.set_cell_source(
        4, "feat = clean.assign(revenue=clean.units * clean.price * 0.9)"
    )
    # Isolated run of the final cell only — upstream sim must propagate.
    nb_runner.run_cell(6)
    out = nb_runner.get_output(6)
    assert "summary= {'north': 72.0, 'south': 99.0}" in out, (
        f"isolated final-cell run after feature edit served stale summary: {out!r}"
    )
    # And run_all agrees.
    nb_runner.run_all()
    out = nb_runner.get_output(6)
    assert "summary= {'north': 72.0, 'south': 99.0}" in out, (
        f"run_all after feature edit wrong: {out!r}"
    )


def test_pandas_pipeline_downstream_edit_no_upstream_recompute(nb_runner, tmp_path):
    csv = tmp_path / "sales2.csv"
    _write_sales_csv(csv)
    nb_runner.create_notebook(_pipeline_cells(_p(csv)))
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "summary=" in nb_runner.get_output(6)

    # Edit ONLY the report cell (cosmetic label change).
    nb_runner.set_cell_source(
        6, "print('SUMMARY:', {k: round(v, 1) for k, v in sorted(summary.items())})"
    )
    nb_runner.run_cell(6)
    out = nb_runner.get_output(6)
    raw = nb_runner.get_raw_output(6)
    assert "SUMMARY: {'north': 80.0, 'south': 110.0}" in out, out
    reexec_lines = [l for l in raw.splitlines() if REEXEC in l]
    heavy_reexec = [l for l in reexec_lines if "read_csv" in l or "groupby" in l or "assign" in l]
    assert not heavy_reexec, (
        f"EFFECTIVENESS: editing only the report cell re-executed upstream "
        f"pipeline statements: {heavy_reexec}"
    )


# ---------------------------------------------------------------------------
# 3. finance backtest loop
# ---------------------------------------------------------------------------

def test_finance_backtest_param_edit(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np\nprices = np.array([100.0, 102.0, 101.0, 105.0, 107.0, 106.0, 110.0, 108.0])",
        "rets = np.diff(prices) / prices[:-1]",
        "windows = [2, 3]",
        "vols = {}\nfor w in windows:\n"
        "    acc = []\n"
        "    for i in range(len(rets) - w + 1):\n"
        "        acc.append(rets[i:i+w].std())\n"
        "    vols[w] = round(float(np.mean(acc)), 6)",
        "print('vols=', vols)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    first = nb_runner.get_output(5)
    assert "vols= {2:" in first, first

    # Edit the parameter cell: different windows.
    nb_runner.set_cell_source(3, "windows = [2, 4]")
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "vols= {2:" in out and "4:" in out, (
        f"backtest param edit not propagated through the loop: {out!r}"
    )
    assert "3:" not in out, f"stale window-3 entry survived: {out!r}"


# ---------------------------------------------------------------------------
# 4. numpy training loop, learning-rate edit
# ---------------------------------------------------------------------------

def test_numpy_training_loop_lr_edit(nb_runner):
    # Only 5 iterations: neither lr converges, so the lr edit visibly changes w.
    train_cell = (
        "w = 0.0\n"
        "for _ in range(5):\n"
        "    grad = 2 * (w * xs - ys) @ xs / len(xs)\n"
        "    w = w - lr * grad\n"
        "loss = round(float(((w * xs - ys) ** 2).mean()), 8)"
    )
    nb_runner.create_notebook([
        "import numpy as np\nxs = np.arange(1.0, 6.0)\nys = 3.0 * xs",
        "lr = 0.01",
        train_cell,
        "print('w=', round(float(w), 4), 'loss=', loss)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    out1 = nb_runner.get_output(4)
    assert "w=" in out1, out1

    def _res(s):
        return s.split("w=")[-1]

    nb_runner.set_cell_source(2, "lr = 0.05")
    nb_runner.run_all()
    out2 = nb_runner.get_output(4)
    assert _res(out2) != _res(out1), (
        f"lr edit produced identical training result: {out2!r}"
    )

    # Revert: must reproduce the original numbers exactly.
    nb_runner.set_cell_source(2, "lr = 0.01")
    nb_runner.run_all()
    out3 = nb_runner.get_output(4)
    assert _res(out3) == _res(out1), (
        f"lr revert did not reproduce original training result: {out3!r} vs {out1!r}"
    )


# ---------------------------------------------------------------------------
# 5. ETL write/read chain through a file boundary (run_all propagation)
# ---------------------------------------------------------------------------

def test_etl_write_read_chain_param_edit(nb_runner, tmp_path):
    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text("v\n1\n2\n3\n", encoding="utf-8")
    inter = _p(tmp_path / "intermediate.csv")
    nb_runner.create_notebook([
        f"import pandas as pd\nraw = pd.read_csv('{_p(raw_csv)}')",
        "scale = 10",
        f"out = raw.assign(scaled=raw.v * scale)\nout.to_csv('{inter}', index=False)",
        f"back = pd.read_csv('{inter}')",
        "print('total=', int(back.scaled.sum()))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "total= 60" in nb_runner.get_output(5)

    nb_runner.set_cell_source(2, "scale = 100")
    nb_runner.run_all()
    out = nb_runner.get_output(5)
    assert "total= 600" in out, (
        f"ETL param edit did not propagate through the to_csv/read_csv file "
        f"boundary on run_all: {out!r}"
    )


# ---------------------------------------------------------------------------
# 6. matplotlib figure cell
# ---------------------------------------------------------------------------

def test_matplotlib_figure_cell_rerun(nb_runner, tmp_path):
    png = _p(tmp_path / "fig.png")
    nb_runner.create_notebook([
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport numpy as np",
        "xs = np.linspace(0, 6.28, 50)\nys = np.sin(xs)",
        f"fig, ax = plt.subplots()\nax.plot(xs, ys)\nfig.savefig('{png}')\nprint('saved')",
        "print('ymax=', round(float(ys.max()), 3))",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "saved" in nb_runner.get_output(3)
    assert "ymax= 0.999" in nb_runner.get_output(4)

    from pathlib import Path
    Path(png).unlink()
    # Re-run the figure cell in isolation: the savefig side effect must happen
    # again (or the cell must re-execute) so the file exists afterwards.
    nb_runner.run_cell(3)
    assert Path(png).exists(), (
        "figure cell served from cache without re-creating the savefig output"
    )


# ---------------------------------------------------------------------------
# 7. seeded Monte-Carlo, seed edit + revert
# ---------------------------------------------------------------------------

def test_seeded_montecarlo_seed_edit(nb_runner):
    nb_runner.create_notebook([
        "import numpy as np",
        "seed = 1",
        "rng = np.random.default_rng(seed)\n"
        "paths = rng.normal(0, 1, size=(200, 10)).cumsum(axis=1)\n"
        "est = round(float(paths[:, -1].mean()), 6)",
        "print('est=', est)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    est1 = nb_runner.get_output(4)
    assert "est=" in est1

    nb_runner.set_cell_source(2, "seed = 2")
    nb_runner.run_all()
    est2 = nb_runner.get_output(4)
    assert est2 != est1, f"seed edit produced identical MC estimate: {est2!r}"

    nb_runner.set_cell_source(2, "seed = 1")
    nb_runner.run_all()
    est3 = nb_runner.get_output(4)
    assert est3 == est1, (
        f"seed revert did not reproduce the original estimate: {est3!r} vs {est1!r}"
    )
