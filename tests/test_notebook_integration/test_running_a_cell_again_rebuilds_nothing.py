"""Running a cell again with nothing changed rebuilds nothing upstream.

Round 25 (r25s1): after an upstream edit and one run of the report cell, every
further run of it re-ran ``results = {}``, ``comparison`` and both figure
builds. Two causes: a figure the cell saved (``fig.savefig``) read as changed
downstream of its producer, and a loop the repair re-ran compared its recorded
inputs with a lineage the repair had since corrected.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

FIGURE_CELLS = (
    "import cash\nprint('CASHFILE', cash.__file__)\n%load_ext cash\n%cash_badge print\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0, loop_split_max_iter_seconds=1.0, loop_split_min_remaining_seconds=0.0)\n%cash_on",
    "base = sum(i*i for i in range(2_000_000))",
    "import sys\ndef ev(name, b):\n    print('RUN ev', name, file=sys.stderr)\n    return {'score': b % 97 + len(name)}\nfams = ['aa', 'bbb', 'c']",
    "results = {}\nfor name in fams:\n    results[name] = ev(name, base)\n\ncomparison = sorted((k, v['score']) for k, v in results.items())",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\ndef panel(r):\n    fig, ax = plt.subplots()\n    ax.plot([v['score'] for v in r.values()])\n    return fig",
    "fig_models = panel(results)",
    "fig_models.savefig('model.png')\nprint('REPORT', comparison)",
)

LOOP_CELLS = (
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import sys, time\nimport numpy as np\nimport pandas as pd\npd.DataFrame({'a': np.arange(20000) % 97, 'b': np.arange(20000) % 13, 't': np.arange(20000) % 2}).to_csv('raw.csv', index=False)\nraw = pd.read_csv('raw.csv')",
    "def clean(df):\n    df = df.copy()\n    df.loc[df['a'] > 90, 'a'] = np.nan\n    df['a'] = df['a'].fillna(df['a'].median())\n    return df\n\n\ndef build_features(df):\n    X = pd.get_dummies(df[['b']].astype(str), dtype=float)\n    for col in ['a']:\n        X[col] = df[col].astype(float)\n    return X\n\n\nclean_df = clean(raw)\nmodel_df = clean_df.sample(n=6000, random_state=0)\nX = build_features(model_df)\ny = model_df['t'].to_numpy()",
    "def evaluate(name, k, X, y):\n    time.sleep(0.6)\n    return {'scores': {'model': name, 's': float(X['a'].sum() * k + y.sum())}}\nfamilies = {'m1': 1, 'm2': 2}",
    "results = {}\nfor name, est in families.items():\n    results[name] = evaluate(name, est, X, y)\n\ncomparison = pd.DataFrame([r['scores'] for r in results.values()]).set_index('model').round(4)\ncomparison",
    "print('REPORT', comparison.to_dict())",
)


def _upstream_rows(out):
    return [line.strip() for line in out.splitlines() if line.strip().startswith("^")]


def test_a_saved_figure_is_not_rebuilt_when_the_cell_runs_again(nb_runner):
    nb_runner.create_notebook(list(FIGURE_CELLS))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cells([7])
    out = nb_runner.get_output(7)
    assert "REPORT" in out, out
    assert _upstream_rows(out) == [], out


def test_an_edit_to_the_plotted_data_still_redraws_the_saved_figure(nb_runner):
    nb_runner.create_notebook(list(FIGURE_CELLS))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(2, "base = sum(i*i for i in range(2_000_001))")
    nb_runner.run_cells([7])
    out = nb_runner.get_output(7)
    assert "fig_models = panel(results)" in out, out
    assert nb_runner.peek("[l.get_ydata().tolist() for l in globals()['fig_models'].axes[0].lines]") == "[[25, 26, 24]]"


def test_a_loop_a_repair_re_ran_is_not_rebuilt_on_the_next_run(nb_runner):
    nb_runner.create_notebook(list(LOOP_CELLS))
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, LOOP_CELLS[2].replace("n=6000", "n=6100"))
    nb_runner.run_cells([6])
    nb_runner.run_cells([6])
    out = nb_runner.get_output(6)
    assert "278551.0" in out, out
    assert _upstream_rows(out) == [], out
