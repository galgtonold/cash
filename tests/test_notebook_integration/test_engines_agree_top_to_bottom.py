"""Round 21: in a plain top-to-bottom run the two lineage engines must agree.

The runtime records each variable's lineage as the statement runs; the upstream
simulator recomputes it from the code above a cell. Wherever the two disagree,
the next cell that reads the variable sees "changed" with nothing changed, and
re-runs other cells' statements to "repair" it. In round 21 that re-ran a
``savefig`` without its plotting calls (a blank chart) and recomputed an
``auc_before`` against the refitted model (a wrong report), and made both
notebook testers' projects slower with cash than without it.

The notebook below is an ordinary project rather than an edge case: a CSV with
a text column, a local helper module, pandas, a matplotlib chart saved to disk,
a scikit-learn fit, seeded randomness, an untaken ``if``, a loop, a report
written as JSON. Nothing is edited between runs, so every disagreement is a
defect -- the ``lineage_disagreement`` trace lists all of them, whether or not
the cell being run reads them.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

HELPERS = """\
THRESHOLD = 3


def clean(df):
    df = df.dropna()
    df["region"] = df["region"].str.upper()
    return df
"""

CELLS = [
    "import cash\n%cash_on\n",
    # imports: the import system and matplotlib read files of their own
    "import json\n"
    "import os\n"
    "os.environ['PROJECT_MODE'] = 'report'\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "import helpers\n"
    "from helpers import clean, THRESHOLD\n"
    "OUT = Path('out')\n"
    "OUT.mkdir(exist_ok=True)\n"
    "FLAG = False\n",
    # data with a text column, through a local helper
    "raw = pd.read_csv('data/customers.csv')\n"
    "df = clean(raw)\n"
    "df['spend2'] = df['spend'] * 2\n"
    "if FLAG:\n"
    "    df = df.head(3)\n",
    # seeded randomness, a loop, a function defined in the notebook
    "rng = np.random.default_rng(0)\n"
    "noise = rng.normal(size=len(df))\n"
    "def score(frame):\n"
    "    return float((frame['spend'] > THRESHOLD).mean())\n"
    # the result is named after an attribute the helper uses (`m.forecast(h)`
    # assigned to `forecast` in a round-21 notebook)
    "def top_spend(frame):\n"
    "    return frame.spend.nlargest(3)\n"
    "spend = top_spend(df)\n"
    "totals = {}\n"
    "for region, part in df.groupby('region'):\n"
    "    totals[region] = float(part['spend'].sum())\n",
    # a chart saved to disk (Pillow reopens the PNG it writes)
    "tot = df.groupby('region')['spend'].sum()\n"
    "fig, ax = plt.subplots(figsize=(4, 3))\n"
    "tot.plot(ax=ax, kind='bar')\n"
    "ax.set_xlabel('region'); ax.set_ylabel('spend')\n"
    "fig.tight_layout()\n"
    "fig.savefig(OUT / 'chart.png', dpi=50)\n"
    "plt.close(fig)\n",
    # a model fit, evaluated before and after a refit in the same cell
    "X = df[['spend', 'spend2']].to_numpy()\n"
    "y = (df['churn'] == 'yes').to_numpy()\n"
    "m = LogisticRegression(max_iter=200).fit(X[:6], y[:6])\n"
    "before = float(m.score(X, y))\n"
    "m.fit(X, y)\n"
    "after = float(m.score(X, y))\n",
    # the report: reads nearly everything above
    "report = {'before': before, 'after': after, 'score': score(df),\n"
    "          'totals': totals, 'noise': round(float(noise.sum()), 6),\n"
    "          'mode': os.environ['PROJECT_MODE'], 'spend': float(spend.sum())}\n"
    "(OUT / 'report.json').write_text(json.dumps(report))\n"
    "print('REPORT', json.dumps(report, sort_keys=True))\n",
]

CSV = """id,region,spend,churn
1,north,1.5,no
2,south,4.0,yes
3,north,5.5,no
4,east,2.0,yes
5,south,,no
6,east,7.5,yes
7,north,3.5,no
8,south,6.0,yes
9,east,0.5,no
10,north,8.0,yes
"""


def _disagreements(records):
    return [(r["cell_idx"], r["vars"]) for r in records if r.get("event") == "lineage_disagreement" and r["vars"]]


def _scheduled(records):
    return [r["stmt"] for r in records if r.get("event") == "schedule_reexec"]


@pytest.fixture
def project(nb_runner):
    work = nb_runner.work_dir
    (work / "data").mkdir(exist_ok=True)
    (work / "data" / "customers.csv").write_text(CSV, encoding="utf-8")
    (work / "helpers.py").write_text(HELPERS, encoding="utf-8")
    return work


def _report(nb_runner):
    out = nb_runner.get_output(len(CELLS))
    return next((line for line in out.splitlines() if line.startswith("REPORT")), out)


def _oracle(work, cells):
    """The report the same cells print without cash, in a fresh process."""
    import subprocess
    import sys

    script = "\n".join(cells[1:])
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=work,
        capture_output=True,
        text=True,
        check=True,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    ).stdout
    return next(line for line in out.splitlines() if line.startswith("REPORT"))


def test_a_top_to_bottom_run_has_no_disagreement(project, upstream_trace, nb_runner):
    """Cold run, then Restart & Run All, then the report cell alone."""

    def after(r):
        r.restart()
        r.run_all()
        r.run_cell(len(CELLS))

    first = {}

    def capture_first(r):
        first["report"] = _report(r)
        after(r)

    t = upstream_trace(CELLS, capture_first)
    assert "REPORT" in first["report"], first["report"]

    found = {
        "cold: disagree": _disagreements(t.run_all),
        "cold: re-ran": _scheduled(t.run_all),
        "restart + run all + report: disagree": _disagreements(t.rerun),
        "restart + run all + report: re-ran": _scheduled(t.rerun),
    }
    assert not any(found.values()), "\n".join(f"{k}: {v}" for k, v in found.items())
    assert _report(nb_runner) == first["report"] == _oracle(project, CELLS)


# --- control arms: agreement must not come from ignoring real changes ----------


def test_edited_data_is_still_seen_by_the_report_cell(project, upstream_trace, nb_runner):
    def edit_and_report(r):
        csv = project / "data" / "customers.csv"
        csv.write_text(CSV.replace("2,south,4.0,yes", "2,south,9.0,yes"), encoding="utf-8")
        r.run_cell(len(CELLS))

    t = upstream_trace(CELLS, edit_and_report)
    assert _scheduled(t.rerun), "nothing was re-planned after the data changed"
    assert _report(nb_runner) == _oracle(project, CELLS)


def test_a_file_a_loop_reads_is_still_seen_downstream(nb_runner, upstream_trace):
    """A loop reading files per iteration: its inputs keep their lineage when a
    file changes, so the recorded outcome must also check the files."""
    work = nb_runner.work_dir
    (work / "notes").mkdir(exist_ok=True)
    for name, text in (("a", "alpha"), ("b", "beta")):
        (work / "notes" / f"{name}.txt").write_text(text, encoding="utf-8")
    cells = [
        "import cash\n%cash_on\n",
        "names = ['a', 'b']\n",
        "parts = {}\n"
        "for name in names:\n"
        "    with open(f'notes/{name}.txt', encoding='utf-8') as fh:\n"
        "        parts[name] = fh.read()\n",
        "print('REPORT', sorted(parts.items()))\n",
    ]

    def edit_and_report(r):
        (work / "notes" / "a.txt").write_text("ALPHA", encoding="utf-8")
        r.run_cell(len(cells))

    upstream_trace(cells, edit_and_report)
    assert _report_of(nb_runner, len(cells)) == _oracle(work, cells)


def _report_of(nb_runner, cell):
    out = nb_runner.get_output(cell)
    return next((line for line in out.splitlines() if line.startswith("REPORT")), out)


def test_taking_the_if_is_still_seen_by_the_report_cell(project, upstream_trace, nb_runner):
    """`FLAG = True` makes the `if` shorten `df`: the recorded outcome of the
    untaken branch must not be reused."""
    flagged = [c.replace("FLAG = False", "FLAG = True") for c in CELLS]

    def flip_and_report(r):
        r.set_cell_source(2, flagged[1])
        r.run_cell(len(CELLS))

    t = upstream_trace(CELLS, flip_and_report)
    assert _scheduled(t.rerun), "nothing was re-planned after FLAG changed"
    assert _report(nb_runner) == _oracle(project, flagged)
