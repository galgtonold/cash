"""Project-shaped notebooks and the edits people made to them.

Shapes, not copies: ``sales`` follows a real forecasting notebook (weekly CSVs
in a folder, a two-panel chart drawn through ``ax=`` and saved, a backtest
loop, an export), ``churn`` follows a real model notebook (text-column CSVs, a
cleaning chain, an optional region filter, an evaluate-refit-report cell, an
importance chart). The edits are the ones people made: a corrected
re-delivery, a new weekly file, a parameter, a leak fix, a chart tweak.

Every notebook also has an unedited ``control`` scenario: if that one fails,
the harness is wrong, not cash.
"""

from __future__ import annotations

import numpy as np
from replay_harness import Edit, scenarios_from

SETUP = "import cash\n%cash_on\n"


# ---------------------------------------------------------------------------
# sales
# ---------------------------------------------------------------------------


def _weekly_csv(week: str, seed: int, bump: int = 0) -> str:
    rs = np.random.RandomState(seed)
    rows = ["week,sku,units"]
    for sku in ("A1", "A2", "B1", "B2", "C1"):
        rows.append(f"{week},{sku},{int(rs.randint(5, 40)) + bump}")
    return "\n".join(rows) + "\n"


_WEEKS = ["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27", "2025-02-03", "2025-02-10"]

SALES_FILES = tuple(
    [(f"data/weekly/sales_{w}.csv", _weekly_csv(w, i)) for i, w in enumerate(_WEEKS)]
    + [
        (
            "data/products.csv",
            "sku,category,price\nA1,alpha,2.5\nA2,alpha,3.0\nB1,beta,1.2\nB2,beta,4.1\nC1,gamma,9.9\n",
        )
    ]
)

SALES = (
    SETUP,
    # 2 -- imports and parameters
    "import glob\n"
    "import time\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "DATA = Path('data')\n"
    "OUT = Path('out')\n"
    "OUT.mkdir(exist_ok=True)\n"
    "H = 4\n",
    # 3 -- load every weekly file
    "files = sorted(glob.glob(str(DATA / 'weekly' / 'sales_*.csv')))\n"
    "raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)\n"
    "products = pd.read_csv(DATA / 'products.csv')\n"
    "print('rows', len(raw), 'files', len(files))\n",
    # 4 -- clean
    "weekly = raw.merge(products, on='sku')\n"
    "weekly['week'] = pd.to_datetime(weekly['week'])\n"
    "weekly['revenue'] = weekly['units'] * weekly['price']\n"
    "weekly = weekly[weekly['units'] >= 0]\n"
    "weekly.to_csv(OUT / 'weekly_clean.csv', index=False)\n"
    "print('clean', len(weekly), round(weekly['revenue'].sum(), 2))\n",
    # 5 -- the overview chart, drawn through ax= and saved
    "tot = weekly.groupby(['week', 'category'])['units'].sum().unstack()\n"
    "fig, axes = plt.subplots(1, 2, figsize=(8, 3))\n"
    "tot.plot(ax=axes[0], title='Units by category')\n"
    "weekly.groupby('week')['revenue'].sum().plot(ax=axes[1], title='Revenue')\n"
    "axes[1].set_xlabel('week'); axes[1].set_ylabel('revenue')\n"
    "print('panels', len(axes))\n"
    "fig.tight_layout()\n"
    "fig.savefig(OUT / 'overview.png', dpi=40)\n"
    "plt.close(fig)\n",
    # 6 -- series and the model functions
    "series = weekly.pivot_table(index='sku', columns='week', values='units', aggfunc='sum').fillna(0)\n"
    "def smooth_forecast(y, h, alpha=0.5):\n"
    "    level = y[0]\n"
    "    for v in y[1:]:\n"
    "        level = alpha * v + (1 - alpha) * level\n"
    "    return np.repeat(level, h)\n"
    "def run_backtest(series, n):\n"
    "    time.sleep(0.6)             # the expensive step of the notebook\n"
    "    open('calls.log', 'a').write('backtest\\n')\n"
    "    rows = []\n"
    "    for sku, y in series.iterrows():\n"
    "        vals = y.values\n"
    "        for k in range(len(vals) - n, len(vals)):\n"
    "            rows.append((sku, k, abs(smooth_forecast(vals[:k], 1)[0] - vals[k])))\n"
    "    return pd.DataFrame(rows, columns=['sku', 'k', 'err'])\n"
    "def run_forecast(series, h):\n"
    "    time.sleep(0.4)\n"
    "    open('calls.log', 'a').write('forecast\\n')\n"
    "    rows = [(sku, i, v) for sku, y in series.iterrows()\n"
    "            for i, v in enumerate(smooth_forecast(y.values, h))]\n"
    "    return pd.DataFrame(rows, columns=['sku', 'step', 'forecast'])\n",
    # 7 -- backtest
    "bt = run_backtest(series, 3)\n"
    "bt_metrics = bt.groupby('sku')['err'].mean().rename('mae')\n"
    "print('backtest', len(bt), round(bt['err'].mean(), 4))\n",
    # 8 -- forecast
    "forecast = run_forecast(series, H)\nprint('forecast', forecast.shape, round(forecast['forecast'].sum(), 3))\n",
    # 9 -- export
    "table = forecast.groupby('sku')['forecast'].sum().to_frame('fc').join(bt_metrics).round(4)\n"
    "table.to_csv(OUT / 'table.csv')\n"
    "print(table.to_string())\n",
    # 10 -- a summary that reads only the cleaned file cell 4 saved (the
    # doubled backtest: once that writer was scheduled, every later statement
    # carrying a file dependency was re-run with it)
    "clean_file = pd.read_csv(OUT / 'weekly_clean.csv')\n"
    "print(clean_file.groupby('category')['units'].sum().to_string())\n",
)

_CORRECTED = _weekly_csv(_WEEKS[2], 2, bump=7)
_NEW_WEEK = "2025-02-17"

SALES_EDITS = {
    "control": Edit(),
    "corrected_file": Edit(files=((f"data/weekly/sales_{_WEEKS[2]}.csv", _CORRECTED),)),
    "new_week": Edit(files=((f"data/weekly/sales_{_NEW_WEEK}.csv", _weekly_csv(_NEW_WEEK, 99)),)),
    "horizon": Edit(cell=2, source=SALES[1].replace("H = 4", "H = 6")),
    "chart_title": Edit(cell=5, source=SALES[4].replace("'Revenue'", "'Revenue per week'")),
    "negative_filter": Edit(cell=4, source=SALES[3].replace(">= 0", "> 10")),
}
SALES_TARGETS = {
    "control": (9, 10),
    # (8, 9): look at the forecast first, then the export. The forecast's
    # replay refreshes the series but not the backtest above it, which the
    # forecast does not read; the export does, so the backtest computed from
    # the old series must still be recomputed. (10, 9): the
    # summary re-writes the cleaned file, then the export needs both models.
    "corrected_file": (9, 5, 7, 10, (8, 9), (10, 9)),
    "new_week": (9, 10),
    "horizon": (9, 8),
    "chart_title": (9,),
    "negative_filter": (9, 5, (8, 9)),
}


# ---------------------------------------------------------------------------
# churn
# ---------------------------------------------------------------------------


def _customers(n: int = 120, seed: int = 0, fix_row: bool = False) -> str:
    rs = np.random.RandomState(seed)
    rows = ["customer_id,region,plan,signup,age,monthly_fee"]
    for i in range(n):
        region = ["north", "south", "east", "west"][rs.randint(4)]
        plan = ["basic", "pro", "team"][rs.randint(3)]
        age = "" if i % 17 == 0 else str(int(rs.randint(18, 70)))
        fee = round(float(rs.uniform(5, 60)), 2)
        if fix_row and i == 3:
            fee = 99.0
        rows.append(f"{i},{region},{plan},2024-{1 + i % 12:02d}-{1 + i % 27:02d},{age},{fee}")
    return "\n".join(rows) + "\n"


def _events(n: int = 120, seed: int = 1) -> str:
    rs = np.random.RandomState(seed)
    rows = ["customer_id,kind,count"]
    for i in range(n):
        for kind in ("login", "support", "payment"):
            rows.append(f"{i},{kind},{int(rs.poisson(4 if kind == 'login' else 1))}")
    return "\n".join(rows) + "\n"


CHURN_FILES = (("data/customers.csv", _customers()), ("data/events.csv", _events()))

CHURN = (
    SETUP,
    # 2 -- imports and parameters
    "import json\n"
    "import time\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import roc_auc_score\n"
    "DATA = Path('data')\n"
    "OUT = Path('out')\n"
    "OUT.mkdir(exist_ok=True)\n"
    "REGION = None\n"
    "EVAL_ROWS = 40\n"
    "FEATURES = ['age', 'monthly_fee', 'login', 'support', 'payment', 'is_pro']\n",
    # 3 -- load
    "customers = pd.read_csv(DATA / 'customers.csv')\n"
    "events = pd.read_csv(DATA / 'events.csv')\n"
    "print('loaded', customers.shape, events.shape)\n",
    # 4 -- cleaning chain and the optional region filter
    "cust = customers.copy()\n"
    "cust['signup'] = pd.to_datetime(cust['signup'])\n"
    "cust['region'] = cust['region'].str.upper()\n"
    "cust['age'] = cust['age'].fillna(cust['age'].median())\n"
    "cust['is_pro'] = (cust['plan'] != 'basic').astype(int)\n"
    "if REGION:\n"
    "    cust = cust[cust['region'] == REGION]\n"
    "print('clean', len(cust), cust['region'].nunique())\n",
    # 5 -- features and label
    "def build_counts(ev):\n"
    "    time.sleep(0.5)             # the expensive step of the notebook\n"
    "    open('calls.log', 'a').write('build_counts\\n')\n"
    "    return ev.pivot_table(index='customer_id', columns='kind', values='count', aggfunc='sum').fillna(0)\n"
    "counts = build_counts(events)\n"
    "data = cust.merge(counts, left_on='customer_id', right_index=True)\n"
    "data['churn'] = ((data['support'] > 1) & (data['login'] < 5)).astype(int)\n"
    "print('features', data.shape, int(data['churn'].sum()))\n",
    # 6 -- evaluate, refit, report (the refit shape: the same model
    # object scored on a hold-out before and after it is refitted in place)
    "X, y = data[FEATURES].to_numpy(), data['churn'].to_numpy()\n"
    "half = len(data) // 2\n"
    "m = LogisticRegression(max_iter=500).fit(X[:half], y[:half])\n"
    "X_eval, y_eval = X[-EVAL_ROWS:], y[-EVAL_ROWS:]\n"
    "auc_before = roc_auc_score(y_eval, m.predict_proba(X_eval)[:, 1])\n"
    "m.fit(X, y)\n"
    "auc_after = roc_auc_score(y_eval, m.predict_proba(X_eval)[:, 1])\n"
    "report = {'before': round(auc_before, 6), 'after': round(auc_after, 6)}\n"
    "(OUT / 'report.json').write_text(json.dumps(report))\n"
    "print('report', report)\n",
    # 7 -- importance chart drawn through ax=
    "imp = pd.DataFrame({'feature': FEATURES, 'weight': np.abs(m.coef_[0])}).sort_values('weight')\n"
    "fig, ax = plt.subplots(figsize=(5, 3))\n"
    "imp.plot.barh(x='feature', y='weight', ax=ax, legend=False)\n"
    "ax.set_title('What drives churn')\n"
    "fig.tight_layout()\n"
    "fig.savefig(OUT / 'importance.png', dpi=40)\n"
    "plt.close(fig)\n",
    # 8 -- summary for the one-pager
    "top = imp.iloc[-1]['feature']\nprint('summary', report, 'top driver:', top, 'rows:', len(data))\n",
    # 9 -- a second chart that REBINDS fig/ax (the notebook's last cell: re-running it
    # alone re-ran 9 statements of other cells)
    "fig, ax = plt.subplots(figsize=(4, 3))\n"
    "data.groupby('region')['churn'].mean().plot.bar(ax=ax)\n"
    "ax.set_title(f'churn by region (top driver: {top})')\n"
    "fig.savefig(OUT / 'by_region.png', dpi=40)\n"
    "plt.close(fig)\n"
    "print('saved by_region', len(data))\n",
)

CHURN_EDITS = {
    "control": Edit(),
    "corrected_file": Edit(files=(("data/customers.csv", _customers(fix_row=True)),)),
    "region": Edit(cell=2, source=CHURN[1].replace("REGION = None", "REGION = 'NORTH'")),
    "leak_fix": Edit(cell=2, source=CHURN[1].replace(", 'is_pro']", "]")),
    "max_iter": Edit(cell=6, source=CHURN[5].replace("max_iter=500", "max_iter=500, C=0.3")),
    "chart_title": Edit(cell=7, source=CHURN[6].replace("What drives churn", "Churn drivers")),
    # Only the hold-out changes, not what the model is fitted on: the "before"
    # score must come from the model as it was BEFORE the refit.
    "eval_rows": Edit(cell=2, source=CHURN[1].replace("EVAL_ROWS = 40", "EVAL_ROWS = 70")),
}
CHURN_TARGETS = {
    "control": (8, 9),
    "corrected_file": (8, 6),
    # (9, 8): the region chart first replays the model for its top driver
    # but not the report beside it, which the chart does not read; the
    # summary prints that report, so it must follow.
    "region": (8, (9, 8)),
    "leak_fix": (8, 7),
    "max_iter": (8,),
    "chart_title": (8, 9),
    "eval_rows": (8, 6),
}


def expected_recompute(scenario) -> list[str]:
    """The expensive steps a scenario NEEDS to recompute -- no more.

    Cost is asserted, not just reported: re-running the backtest for a cell
    that reads only the cleaned file (the doubled backtest) or after a
    restart for a chart nothing reads is exactly the kind of waste that made
    the notebook path slower than no cache once.
    """
    if scenario.notebook == "churn":
        return []  # the events file never changes
    done = set().union(*(_sales_needs(scenario.name, c) for c in scenario.first))
    return sorted(_sales_needs(scenario.name, scenario.target) - done)


def _sales_needs(edit: str, cell: int) -> set[str]:
    """What running *cell* after *edit* must recompute, from a first run."""
    if edit in ("corrected_file", "new_week", "negative_filter"):
        return {7: {"backtest"}, 8: {"forecast"}, 9: {"backtest", "forecast"}}.get(cell, set())
    if edit == "horizon" and cell in (8, 9):
        return {"forecast"}
    return set()


SCENARIOS = scenarios_from(
    "sales", SALES, SALES_FILES, SALES_EDITS, SALES_TARGETS, restart=("corrected_file", "new_week")
) + scenarios_from(
    "churn", CHURN, CHURN_FILES, CHURN_EDITS, CHURN_TARGETS, restart=("corrected_file", "region", "eval_rows")
)
