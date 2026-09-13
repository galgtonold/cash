"""Round-22 testers' sessions, shrunk to a small dataset.

``finance_deck`` is r22s3's: a monthly management deck from one ledger file
per month (listed with ``Path.glob``), charts saved through a ``save(fig,
name)`` helper, ``import glob`` in two cells, a slow per-cost-centre forecast.
The month arrives, the charts are restyled, a renamed cost centre is fixed
upstream while looking at a chart first, the next month and a corrected
re-delivery arrive after a restart, and the deck is rebuilt with Restart & Run
All into an emptied output folder.

``demand_forecast`` is r22s4's: daily sales per region in a folder of parts,
a feature cell of column assignments over a cross-joined panel, a model per
region, a rolling-origin backtest built with ``rows.append`` over cut-offs and
a grid, the best parameter picked from it. Holidays are fixed, the grid is
widened in the middle, a lookahead leak is fixed, a weather feature is added,
the late region's missing week arrives as a new file, and Restart & Run All.

The expensive functions sleep past the 0.1 s persistence floor and log their
calls, so what is cached in the tester's notebook is cached here too.
"""
from __future__ import annotations

import numpy as np

from session_harness import (AddFile, ClearDir, Edit, ReplaceFile, Restart, RestartAndRunAll,
                             Run, RunAll, Session)

SETUP = "import cash\n%cash_on"


# ---------------------------------------------------------------------------
# r22s3 -- finance deck
# ---------------------------------------------------------------------------

def _gl_month(month: int, bump: float = 0.0) -> str:
    rs = np.random.RandomState(month)
    rows = ["date,entity,cost_centre,account,amount,currency"]
    centres = ["CC10", "CC20", "CC30", "CC40" if month <= 3 else "CC41"]   # CC40 renamed in April
    for i in range(24):
        entity = "E2" if i % 3 == 0 else "E1"
        account = (4000, 5000, 6000)[i % 3]
        sign = 1 if account == 4000 else -1
        amount = round(sign * float(rs.uniform(100, 900)) + bump, 2)
        currency = "USD" if entity == "E2" else "EUR"
        rows.append(f"2026-{month:02d}-{1 + i:02d},{entity},{centres[i % 4]},{account},{amount},{currency}")
    return "\n".join(rows) + "\n"


_MONTHS = [f"2026-{m:02d}" for m in range(1, 9)]
DECK_FILES = tuple(
    [(f"ledger/gl_2026-{m:02d}.csv", _gl_month(m)) for m in range(1, 7)]
    + [("ledger/adj_2026-06.csv", "month,line,amount\n2026-06,opex,-50.0\n2026-06,revenue,12.5\n"),
       ("mapping.csv", "account,line\n4000,revenue\n5000,cost\n6000,opex\n"),
       ("fx.csv", "month,currency,rate\n" + "".join(
           f"{m},EUR,1.0\n{m},USD,{0.90 + i * 0.01:.2f}\n" for i, m in enumerate(_MONTHS))),
       ("budget.csv", "month,line,budget\n" + "".join(
           f"{m},revenue,{3000 + 50 * i}\n{m},cost,{-2500 - 40 * i}\n{m},opex,{-2600 - 30 * i}\n"
           for i, m in enumerate(_MONTHS)))]
)

DECK = (
    ("setup", SETUP),
    ("imports",
     "import time\n"
     "from pathlib import Path\n"
     "import pandas as pd\n"
     "import matplotlib\n"
     "matplotlib.use('Agg')\n"
     "import matplotlib.pyplot as plt\n"
     "OUT = Path('out')\n"
     "OUT.mkdir(exist_ok=True)\n"
     "def save(fig, name):\n"
     "    fig.tight_layout()\n"
     "    fig.savefig(OUT / name, dpi=40)\n"
     "    plt.close(fig)\n"
     "def load_month(f):\n"
     "    df = pd.read_csv(f)\n"
     "    df['month'] = f.stem[3:]\n"
     "    return df\n"),
    ("load",
     "import glob\n"
     "gl_files = sorted(Path('ledger').glob('gl_*.csv'))\n"
     "gl = pd.concat([load_month(f) for f in gl_files], ignore_index=True)\n"
     "adj_files = sorted(glob.glob('ledger/adj_*.csv'))\n"
     "print('files', len(gl_files), 'lines', len(gl))\n"),
    ("mapping",
     "mapping = pd.read_csv('mapping.csv')\n"
     "RENAMES = {}\n"
     "gl['cc'] = gl['cost_centre'].replace(RENAMES)\n"
     "gl = gl.merge(mapping, on='account')\n"),
    ("fx",
     "fx = pd.read_csv('fx.csv')\n"
     "gl = gl.merge(fx, on=['month', 'currency'], how='left')\n"
     "gl['amount_eur'] = (gl['amount'] * gl['rate']).round(2)\n"),
    ("adjust",
     "import glob\n"
     "adj = pd.concat([pd.read_csv(f) for f in adj_files + sorted(glob.glob('ledger/late_*.csv'))],\n"
     "                ignore_index=True)\n"
     "pnl_lines = gl.groupby(['month', 'line'])['amount_eur'].sum().unstack(fill_value=0.0)\n"
     "for _, a in adj.iterrows():\n"
     "    pnl_lines.loc[a['month'], a['line']] += a['amount']\n"),
    ("reconcile",
     "budget = pd.read_csv('budget.csv').pivot(index='month', columns='line', values='budget')\n"
     "variance = (pnl_lines - budget.loc[pnl_lines.index]).round(2)\n"
     "by_cc = gl.groupby(['cc', 'month'])['amount_eur'].sum().unstack(fill_value=0.0)\n"
     "print('variance total', round(float(variance.sum().sum()), 2))\n"),
    ("forecast",
     "def forecast_cc(series, h=3):\n"
     "    open('calls.log', 'a').write('forecast\\n')\n"
     "    time.sleep(0.15)             # the slow step of the deck\n"
     "    level = series.iloc[0]\n"
     "    for v in series.iloc[1:]:\n"
     "        level = 0.5 * v + 0.5 * level\n"
     "    return [round(float(level), 2)] * h\n"
     "fc = {cc: forecast_cc(by_cc.loc[cc]) for cc in by_cc.index}\n"
     "fc_table = pd.DataFrame(fc, index=['m+1', 'm+2', 'm+3']).T\n"),
    ("charts",
     "fig, axes = plt.subplots(1, 2, figsize=(8, 3))\n"
     "pnl_lines.plot(ax=axes[0], title='P&L by line')\n"
     "variance.sum(axis=1).plot.bar(ax=axes[1], title='Variance to budget')\n"
     "save(fig, 'pnl.png')\n"
     "fig, axes = plt.subplots(2, 2, figsize=(8, 5))\n"
     "for ax, cc in zip(axes.flat, by_cc.index):\n"
     "    by_cc.loc[cc].plot(ax=ax, title=cc)\n"
     "save(fig, 'by_cc.png')\n"),
    ("fc_chart",
     "fig, ax = plt.subplots(figsize=(5, 3))\n"
     "fc_table['m+1'].plot.bar(ax=ax, title='Forecast next month')\n"
     "save(fig, 'forecast.png')\n"),
    ("export",
     "variance.to_csv(OUT / 'variance.csv')\n"
     "fc_table.to_csv(OUT / 'forecast.csv')\n"
     "by_cc.to_csv(OUT / 'by_cc.csv')\n"
     "print('exported', len(variance), len(fc_table))\n"),
    ("summary",
     "print('summary', round(float(pnl_lines.sum().sum()), 2), fc_table.shape,\n"
     "      sorted(by_cc.index), round(float(fc_table['m+1'].sum()), 2))\n"),
)

FINANCE_DECK = Session(
    name="r22s3 finance deck",
    cells=DECK,
    files=DECK_FILES,
    steps=(
        RunAll(),
        # Session A: the new month's ledger file arrives (a Path.glob listing).
        AddFile("ledger/gl_2026-07.csv", _gl_month(7)),
        Run("charts"),
        Run("export"),
        # Session B: restyle the charts -- nothing expensive may re-run for that --
        Edit("charts", lambda s: s.replace("'P&L by line'", "'P&L by line, EUR'")),
        Run("charts", calls={"forecast": 0}),
        # -- then fix the renamed cost centre upstream, look at a chart first,
        # then export (r22s3 F10: the forecast beside the chart went stale).
        Edit("mapping", lambda s: s.replace("RENAMES = {}", "RENAMES = {'CC40': 'CC41'}")),
        Run("charts"),
        Run("export"),
        Run("summary"),
        # Session C, next morning: restart, another month plus a corrected
        # re-delivery of April under the same name, jump to the summary
        # (r22s3 F6: `import glob` in two cells refused the jump).
        Restart(),
        AddFile("ledger/gl_2026-08.csv", _gl_month(8)),
        ReplaceFile("ledger/gl_2026-04.csv", _gl_month(4, bump=7.5)),
        Run("summary"),
        # The deck goes out: an emptied output folder, Restart & Run All
        # (r22s3 F2: charts saved through the helper were not written).
        ClearDir("out"),
        RestartAndRunAll(),
        RunAll(calls={"forecast": 0}),
    ),
)


# ---------------------------------------------------------------------------
# r22s4 -- demand forecast per region
# ---------------------------------------------------------------------------

_DATES = [np.datetime64("2026-01-01") + np.timedelta64(i, "D") for i in range(120)]
_NATIONAL = {"2026-01-01", "2026-04-03", "2026-04-06"}
_REGIONAL = {("R1", "2026-03-19"), ("R2", "2026-02-16"), ("R2", "2026-02-17")}
_LATE_FROM = np.datetime64("2026-04-24")         # R2's last week arrives late


def _sales(late: bool) -> str:
    rs = np.random.RandomState(4)
    rows = ["date,region,family,units"]
    for d in _DATES:
        ds = str(d)
        dow = (int((d - np.datetime64("1970-01-01")) // np.timedelta64(1, "D")) + 3) % 7   # a Thursday
        for r in ("R0", "R1", "R2"):
            for k, fam in enumerate(("F0", "F1", "F2")):
                lam = 20 + 10 * k + (8 if dow >= 5 else 0)
                if ds in _NATIONAL or (r, ds) in _REGIONAL:
                    lam *= 1.8
                units = int(rs.poisson(lam))
                if (r == "R2" and d >= _LATE_FROM) == late:
                    rows.append(f"{ds},{r},{fam},{units}")
    return "\n".join(rows) + "\n"


def _calendar() -> str:
    rows = ["date,region,holiday"]
    rows += [f"{d},ALL,1" for d in sorted(_NATIONAL)]
    rows += [f"{d},{r},1" for r, d in sorted(_REGIONAL)]
    return "\n".join(rows) + "\n"


def _weather() -> str:
    rs = np.random.RandomState(7)
    rows = ["date,region,temp"]
    for d in _DATES:
        for r in ("R0", "R1", "R2"):
            rows.append(f"{d},{r},{round(float(rs.normal(8, 5)), 1)}")
    return "\n".join(rows) + "\n"


FORECAST_FILES = (
    ("sales/part_1.csv", _sales(late=False)),
    ("calendar.csv", _calendar()),
    ("weather.csv", _weather()),
)

FORECAST = (
    ("setup", SETUP),
    ("imports",
     "import glob\n"
     "import time\n"
     "from pathlib import Path\n"
     "import numpy as np\n"
     "import pandas as pd\n"
     "import matplotlib\n"
     "matplotlib.use('Agg')\n"
     "import matplotlib.pyplot as plt\n"
     "from sklearn.linear_model import Ridge\n"
     "OUT = Path('out')\n"
     "OUT.mkdir(exist_ok=True)\n"),
    ("load",
     "files = sorted(glob.glob('sales/part_*.csv'))\n"
     "daily = pd.concat([pd.read_csv(f, parse_dates=['date']) for f in files], ignore_index=True)\n"
     "calendar = pd.read_csv('calendar.csv', parse_dates=['date'])\n"
     "weather = pd.read_csv('weather.csv', parse_dates=['date'])\n"
     "print('rows', len(daily), 'files', len(files))\n"),
    ("features",
     "HORIZON = 7\n"
     "nat = set(calendar.loc[calendar.region == 'ALL', 'date'])\n"
     "keys = daily[['region', 'family']].drop_duplicates()\n"
     "panel = keys.merge(pd.DataFrame({'date': pd.date_range(daily.date.min(), daily.date.max())}), how='cross')\n"
     "panel = (panel.merge(daily, on=['region', 'family', 'date'], how='left')\n"
     "              .sort_values(['region', 'family', 'date']).reset_index(drop=True))\n"
     "panel['units'] = panel['units'].fillna(0.0)\n"
     "panel['holiday'] = panel['date'].isin(nat).astype(int)\n"
     "g = panel.groupby(['region', 'family']).units\n"
     "panel['lag_7'] = g.shift(HORIZON)\n"
     "panel['rmean_7'] = g.transform(lambda s: s.rolling(7, min_periods=1).mean())\n"
     "panel = panel.merge(weather, on=['date', 'region'], how='left')\n"
     "panel['dow'] = panel.date.dt.dayofweek\n"
     "panel = panel.dropna(subset=['lag_7']).reset_index(drop=True)\n"
     "FEATURES = ['lag_7', 'rmean_7', 'holiday', 'dow']\n"
     "print('panel', panel.shape)\n"),
    ("models",
     "def fit_region(frame, alpha):\n"
     "    open('calls.log', 'a').write('fit\\n')\n"
     "    time.sleep(0.15)\n"
     "    return Ridge(alpha=alpha).fit(frame[FEATURES], frame['units'])\n"
     "models = {}\n"
     "for r in sorted(panel.region.unique()):\n"
     "    models[r] = fit_region(panel[panel.region == r], 1.0)\n"),
    ("backtest",
     "GRID = (0.1, 10.0)\n"
     "CUTOFFS = list(pd.to_datetime(sorted(panel.date.unique()))[-21::7])\n"
     "def backtest_one(frame, cutoff, alpha):\n"
     "    open('calls.log', 'a').write('backtest\\n')\n"
     "    time.sleep(0.15)             # the slow loop of the notebook\n"
     "    train = frame[frame.date < cutoff]\n"
     "    test = frame[(frame.date >= cutoff) & (frame.date < cutoff + pd.Timedelta(days=7))]\n"
     "    m = Ridge(alpha=alpha).fit(train[FEATURES], train['units'])\n"
     "    return float(abs(m.predict(test[FEATURES]) - test['units']).sum() / max(test['units'].sum(), 1.0))\n"
     "rows = []\n"
     "for cutoff in CUTOFFS:\n"
     "    for alpha in GRID:\n"
     "        rows.append(dict(cutoff=str(cutoff)[:10], alpha=alpha, wape=backtest_one(panel, cutoff, alpha)))\n"
     "bt = pd.DataFrame(rows)\n"
     "BEST_ALPHA = float(bt.groupby('alpha').wape.mean().idxmin())\n"
     "print('best alpha', BEST_ALPHA, 'cutoffs', len(CUTOFFS))\n"),
    ("forecast",
     "final = Ridge(alpha=BEST_ALPHA).fit(panel[FEATURES], panel['units'])\n"
     "last = panel.sort_values('date').groupby(['region', 'family']).tail(1)\n"
     "forecast = last[['region', 'family']].assign(forecast=final.predict(last[FEATURES]).round(3))\n"
     "forecast.to_csv(OUT / 'forecast.csv', index=False)\n"
     "print('forecast', round(float(forecast.forecast.sum()), 3), 'alpha', BEST_ALPHA)\n"),
    ("metrics",
     "metrics = bt.groupby('alpha').wape.mean().round(6)\n"
     "metrics.to_csv(OUT / 'metrics.csv')\n"
     "print('metrics', metrics.to_dict(), 'best', BEST_ALPHA)\n"),
    ("chart",
     "fig, axes = plt.subplots(1, len(models), figsize=(9, 3))\n"
     "for ax, r in zip(np.atleast_1d(axes), sorted(models)):\n"
     "    panel[panel.region == r].groupby('date').units.sum().plot(ax=ax, title=r)\n"
     "fig.tight_layout()\n"
     "fig.savefig(OUT / 'regions.png', dpi=40)\n"
     "plt.close(fig)\n"),
)

_REGIONAL_HOLIDAYS = (
    "reg = calendar[calendar.region != 'ALL'][['date', 'region']].assign(rh=1)\n"
    "panel = panel.merge(reg, on=['date', 'region'], how='left')\n"
    "panel['holiday'] = (panel['date'].isin(nat) | panel['rh'].eq(1)).astype(int)\n"
)

DEMAND_FORECAST = Session(
    name="r22s4 demand forecast",
    cells=FORECAST,
    files=FORECAST_FILES,
    steps=(
        RunAll(),
        # Monday: the unchanged backtest again -- nothing should be refitted.
        Run("backtest", calls={"backtest": 0},
            gap="a call nested in rows.append(dict(..., wape=f(...))) is not call-cached; "
                "rows.append(f(...)) and w = f(...) are"),
        # A new day: restart and jump straight to the metrics
        # (r22s4 BLOCKING: every restart-then-jump was refused).
        Restart(),
        Run("metrics"),
        # Tuesday: the holiday feature ignored regional holidays; fix it
        # upstream, look at the backtest first.
        Edit("features", lambda s: s.replace(
            "panel['holiday'] = panel['date'].isin(nat).astype(int)\n", _REGIONAL_HOLIDAYS)),
        Run("backtest", calls={"backtest": 6}),
        Run("forecast"),
        Run("metrics"),
        # Widen the grid in the middle.
        Edit("backtest", lambda s: s.replace("GRID = (0.1, 10.0)", "GRID = (0.1, 1.0, 10.0)")),
        Run("metrics"),
        # The rolling mean includes the current day -- a lookahead leak.
        Edit("features", lambda s: s.replace("s.rolling(7, min_periods=1)",
                                             "s.shift(1).rolling(7, min_periods=1)")),
        Run("chart"),
        # Add a weather feature.
        Edit("features", lambda s: s.replace("'holiday', 'dow']", "'holiday', 'dow', 'temp']")),
        Run("forecast"),
        Run("metrics"),
        # Wednesday: the late region's missing week lands as a new file
        # (r22s4 WRONG: the forecast used the parameter tuned on old data).
        AddFile("sales/part_2.csv", _sales(late=True)),
        Run("forecast"),
        Run("metrics"),
        # Final forecasts: an emptied output folder, Restart & Run All.
        ClearDir("out"),
        RestartAndRunAll(),
    ),
)

SESSIONS = (FINANCE_DECK, DEMAND_FORECAST)
