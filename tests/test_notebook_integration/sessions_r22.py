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

``ab_readout`` is r22s1's: an A/B test from assignments, a folder of daily
event files read through a helper, and orders; a bootstrap used by the
overall, CUPED and per-segment estimates; forest charts and a summary export.
Bots and a double-fired day are filtered upstream, the charts are looked at
before the summary, the PM redefines conversion, a segment is added, two days
of events are re-delivered after a morning restart, and Restart & Run All.

``ticket_classifier`` is r22s2's: monthly ticket files, regex cleaning through
a helper, a time split, word + char TF-IDF, a grid search per model in a dict
comprehension, a confusion matrix labelled in a nested ``ax.text`` loop, error
analysis and a model card. The merged queue is mapped upstream, a leaking
signature is cleaned out, the grid is widened, QA relabels old tickets after
a restart, a new month arrives, and Restart & Run All.

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
        # Monday: the unchanged backtest again -- nothing may be refitted
        # (the call sits in rows.append(dict(..., wape=backtest_one(...)))).
        Run("backtest", calls={"backtest": 0}),
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

# ---------------------------------------------------------------------------
# r22s1 -- A/B-test readout
# ---------------------------------------------------------------------------

_AB_START = np.datetime64("2026-08-03")
_AB_DAYS = 8


def _assignments() -> str:
    rs = np.random.RandomState(11)
    rows = ["user_id,variant,assigned_at,platform,country,user_type"]
    for u in range(1, 401):
        day = int(rs.randint(0, 6))
        platform = ("ios", "android", "web", "iOS")[int(rs.randint(0, 4))]   # old SDK: "iOS"
        country = ("DE", "FR", "US", "")[int(rs.randint(0, 4))]
        kind = ("new", "returning")[int(rs.randint(0, 2))]
        variant = "AB"[u % 2]
        ts = f"{_AB_START + np.timedelta64(day, 'D')}T{int(rs.randint(0, 24)):02d}:00:00"
        rows.append(f"{u},{variant},{ts},{platform},{country},{kind}")
        if u % 97 == 0:                                 # bucketed into both arms
            rows.append(f"{u},{'BA'[u % 2]},{ts},{platform},{country},{kind}")
    return "\n".join(rows) + "\n"


def _events_day(day: int, fix: bool = False) -> str:
    rs = np.random.RandomState(100 + day)
    date = _AB_START + np.timedelta64(day, "D")
    rows = ["event_id,user_id,ts,event,revenue"]
    n = 0
    for u in range(1, 401):
        for _ in range(int(rs.poisson(2))):
            n += 1
            kind = ("view", "view", "cart", "purchase")[int(rs.randint(0, 4))]
            revenue = round(float(rs.gamma(2.0, 20.0)) * (1.1 if u % 2 == 0 else 1.0), 2) if kind == "purchase" else ""
            if fix and kind == "purchase" and u % 5 == 0:
                revenue = round(float(revenue) * 0.5, 2)   # the corrected re-delivery
            rows.append(f"{day}-{n},{u},{date}T{int(rs.randint(0, 24)):02d}:30:00,{kind},{revenue}")
    for bot in (7, 8):                                  # bot-like users: absurd volumes
        for k in range(90):
            n += 1
            rows.append(f"{day}-{n},{bot},{date}T{k % 24:02d}:10:00,purchase,{1 + k % 3}.0")
    if day == 4:                                        # tracking double-fired that day
        rows += rows[1:40]
    return "\n".join(rows) + "\n"


def _orders() -> str:
    rs = np.random.RandomState(12)
    rows = ["order_id,user_id,ordered_at,revenue,refunded,refund_amount"]
    for k in range(1, 601):
        u = int(rs.randint(1, 401))
        day = int(rs.randint(-28, _AB_DAYS))            # pre-period and in-test
        revenue = round(float(rs.gamma(2.0, 25.0)), 2)
        refunded = int(rs.rand() < 0.1)
        rows.append(f"{k},{u},{_AB_START + np.timedelta64(day, 'D')}T12:00:00,{revenue},{refunded},"
                    f"{round(revenue * refunded, 2)}")
    return "\n".join(rows) + "\n"


READOUT_FILES = tuple(
    [("data/assignments.csv", _assignments()), ("data/orders.csv", _orders())]
    + [(f"data/events/events_{_AB_START + np.timedelta64(d, 'D')}.csv", _events_day(d))
       for d in range(_AB_DAYS)]
)

READOUT = (
    ("setup", SETUP),
    ("imports",
     "import json, time\n"
     "from pathlib import Path\n"
     "import numpy as np\n"
     "import pandas as pd\n"
     "import matplotlib\n"
     "matplotlib.use('Agg')\n"
     "import matplotlib.pyplot as plt\n"
     "DATA = Path('data')\n"
     "OUT = Path('out')\n"
     "OUT.mkdir(exist_ok=True)\n"
     "START = pd.Timestamp('2026-08-03')\n"),
    ("assign",
     "def read_assignments(path):\n"
     "    return pd.read_csv(path, parse_dates=['assigned_at'])\n"
     "assign_raw = read_assignments(DATA / 'assignments.csv')\n"
     "print(assign_raw.shape, assign_raw['variant'].value_counts().to_dict())\n"),
    ("events",
     "def read_events(files):\n"
     "    return pd.concat([pd.read_csv(f, parse_dates=['ts']) for f in files], ignore_index=True)\n"
     "event_files = sorted((DATA / 'events').glob('events_*.csv'))\n"
     "events_raw = read_events(event_files)\n"
     "print(len(event_files), 'files', events_raw.shape)\n"),
    ("orders",
     "orders_raw = pd.read_csv(DATA / 'orders.csv', parse_dates=['ordered_at'])\n"
     "print('refund rate', round(float(orders_raw['refunded'].mean()), 4))\n"),
    ("checks",
     "arms_per_user = assign_raw.groupby('user_id')['variant'].nunique()\n"
     "both_arms = arms_per_user[arms_per_user > 1].index\n"
     "assign = (assign_raw[~assign_raw['user_id'].isin(both_arms)]\n"
     "          .sort_values('assigned_at').drop_duplicates('user_id', keep='first').reset_index(drop=True))\n"
     "assign['platform'] = assign['platform'].str.lower()\n"
     "assign['country'] = assign['country'].fillna('unknown')\n"
     "print('both arms', len(both_arms), 'clean', len(assign))\n"),
    ("metrics",
     "events = events_raw\n"
     "ev = events.merge(assign[['user_id', 'variant', 'assigned_at']], on='user_id', how='inner')\n"
     "ev = ev[ev['ts'] >= ev['assigned_at']]\n"
     "purchases = ev[ev['event'] == 'purchase']\n"
     "conv_users = purchases.groupby('user_id').size().rename('n_purchases')\n"
     "revenue = purchases.groupby('user_id')['revenue'].sum().rename('revenue')\n"
     "test_orders = orders_raw[orders_raw['ordered_at'] >= START]\n"
     "refunds = test_orders.groupby('user_id')['refund_amount'].sum().rename('refunds')\n"
     "print(len(conv_users), 'converters;', round(float(revenue.sum()), 2), 'revenue')\n"),
    ("users",
     "pre_orders = orders_raw[(orders_raw['ordered_at'] < START)]\n"
     "pre_revenue = pre_orders.groupby('user_id')['revenue'].sum().rename('pre_revenue')\n"
     "users = (assign.set_index('user_id')[['variant', 'platform', 'country', 'user_type']]\n"
     "         .join(conv_users).join(revenue).join(refunds).join(pre_revenue))\n"
     "users[['n_purchases', 'revenue', 'refunds', 'pre_revenue']] = (\n"
     "    users[['n_purchases', 'revenue', 'refunds', 'pre_revenue']].fillna(0))\n"
     "users['converted'] = (users['n_purchases'] > 0).astype(float)\n"
     "users['net_revenue'] = users['revenue'] - users['refunds']\n"
     "print(users.shape)\n"),
    ("bootstrap",
     "METRICS = ['converted', 'revenue', 'net_revenue']\n"
     "def bootstrap_diff(df, metrics, n_boot=200, seed=7):\n"
     "    open('calls.log', 'a').write('bootstrap\\n')\n"
     "    time.sleep(0.12)             # thousands of resamples in the real notebook\n"
     "    rng = np.random.default_rng(seed)\n"
     "    a = df.loc[df['variant'] == 'A', metrics].to_numpy()\n"
     "    b = df.loc[df['variant'] == 'B', metrics].to_numpy()\n"
     "    ia = rng.integers(0, len(a), size=(n_boot, len(a)))\n"
     "    ib = rng.integers(0, len(b), size=(n_boot, len(b)))\n"
     "    rows = []\n"
     "    for j, m in enumerate(metrics):\n"
     "        d = b[:, j][ib].mean(axis=1) - a[:, j][ia].mean(axis=1)\n"
     "        rows.append({'metric': m, 'mean_A': a[:, j].mean(), 'mean_B': b[:, j].mean(),\n"
     "                     'diff': b[:, j].mean() - a[:, j].mean(),\n"
     "                     'lo': np.percentile(d, 2.5), 'hi': np.percentile(d, 97.5)})\n"
     "    return pd.DataFrame(rows)\n"),
    ("overall",
     "overall = bootstrap_diff(users, METRICS)\n"
     "print(overall.round(4).to_string())\n"),
    ("cuped",
     "def cuped(df, metrics, covariate='pre_revenue'):\n"
     "    out = df[['variant']].copy()\n"
     "    x = df[covariate]\n"
     "    for m in metrics:\n"
     "        theta = np.cov(df[m], x)[0, 1] / x.var()\n"
     "        out[m] = df[m] - theta * (x - x.mean())\n"
     "    return out\n"
     "users_cuped = cuped(users, METRICS)\n"
     "overall_cuped = bootstrap_diff(users_cuped, METRICS)\n"
     "print(overall_cuped.round(4).to_string())\n"),
    ("segments",
     "SEGMENTS = ['platform', 'country']\n"
     "def segment_breakdown(df, segments, metrics):\n"
     "    parts = []\n"
     "    for s in segments:\n"
     "        for val, g in df.groupby(s):\n"
     "            r = bootstrap_diff(g, metrics, seed=11)\n"
     "            r.insert(0, 'segment', s)\n"
     "            r.insert(1, 'value', str(val))\n"
     "            r.insert(2, 'n_users', len(g))\n"
     "            parts.append(r)\n"
     "    return pd.concat(parts, ignore_index=True)\n"
     "seg = segment_breakdown(users, SEGMENTS, ['converted', 'net_revenue'])\n"
     "print(len(seg), 'segment rows')\n"),
    ("charts",
     "def forest(ax, df, title):\n"
     "    y = np.arange(len(df))[::-1]\n"
     "    ax.errorbar(df['diff'], y, xerr=[(df['diff'] - df['lo']).abs(), (df['hi'] - df['diff']).abs()], fmt='o')\n"
     "    ax.axvline(0, color='#aa3333', lw=1, ls='--')\n"
     "    ax.set_yticks(y)\n"
     "    ax.set_yticklabels(df['label'])\n"
     "    ax.set_title(title)\n"
     "fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))\n"
     "forest(axes[0], overall_cuped.assign(label=overall_cuped['metric']), 'Overall (CUPED)')\n"
     "s = seg[seg['metric'] == 'converted'].assign(label=lambda d: d['segment'] + '=' + d['value'])\n"
     "forest(axes[1], s, 'Conversion by segment')\n"
     "fig.tight_layout()\n"
     "fig.savefig(OUT / 'readout_lifts.png', dpi=40, facecolor='white')\n"
     "plt.close(fig)\n"
     "fig2, ax2 = plt.subplots(figsize=(6, 3))\n"
     "daily = purchases.groupby([purchases['ts'].dt.date, 'variant'])['revenue'].sum().unstack('variant')\n"
     "daily.plot(ax=ax2, marker='.', title='Purchase revenue per day by arm')\n"
     "fig2.tight_layout()\n"
     "fig2.savefig(OUT / 'daily_revenue.png', dpi=40, facecolor='white')\n"
     "plt.close(fig2)\n"),
    ("summary",
     "summary = pd.concat([overall.assign(scope='overall'), overall_cuped.assign(scope='cuped'),\n"
     "                     seg.assign(scope='segment')], ignore_index=True)\n"
     "summary.to_csv(OUT / 'summary.csv', index=False, float_format='%.10g')\n"
     "record = {'n_users': int(len(users)), 'n_events': int(len(ev)),\n"
     "          'overall': overall.set_index('metric')['diff'].round(8).to_dict()}\n"
     "(OUT / 'metrics.json').write_text(json.dumps(record, indent=1, sort_keys=True))\n"
     "print(summary.shape, 'rows;', record['overall'])\n"),
)

_BOT_AND_DOUBLE_FIRE = (
    "events = events_raw.drop_duplicates('event_id')          # tracking double-fired\n"
    "ev_per_user = events.groupby('user_id').size()\n"
    "bot_users = ev_per_user[ev_per_user > 300].index\n"
    "events = events[~events['user_id'].isin(bot_users)]\n"
)

AB_READOUT = Session(
    name="r22s1 ab readout",
    cells=READOUT,
    files=READOUT_FILES,
    steps=(
        RunAll(),
        # Tuesday: bots and the double-fired day distort revenue; filter them
        # upstream, look at the charts, then the summary (r22s1 WRONG #1:
        # the summary exported the bootstrap from before the filter).
        Edit("metrics", lambda s: s.replace("events = events_raw\n", _BOT_AND_DOUBLE_FIRE)),
        Run("charts"),
        Run("summary"),
        # Next morning.
        Restart(),
        Run("summary"),
        # The PM's new conversion definition: a purchase within 7 days.
        Edit("metrics", lambda s: s.replace(
            "conv_users = purchases.groupby",
            "conv_window = purchases[purchases['ts'] < purchases['assigned_at'] + pd.Timedelta(days=7)]\n"
            "conv_users = conv_window.groupby")),
        Run("segments"),
        Run("summary"),
        # Add a segment.
        Edit("segments", lambda s: s.replace("SEGMENTS = ['platform', 'country']",
                                             "SEGMENTS = ['platform', 'country', 'user_type']")),
        Run("charts"),
        # Wednesday: the morning restore, then two days of events re-delivered
        # under the same names (r22s1 WRONG #2: read inside read_events, after
        # a restore, the old files' result was served).
        Restart(),
        Run("summary"),
        ReplaceFile("data/events/events_2026-08-05.csv", _events_day(2, fix=True)),
        ReplaceFile("data/events/events_2026-08-07.csv", _events_day(4, fix=True)),
        Run("summary"),
        # Before sharing: Restart & Run All into an emptied output folder.
        ClearDir("out"),
        RestartAndRunAll(),
        RunAll(calls={"bootstrap": 0}),
    ),
)


# ---------------------------------------------------------------------------
# r22s2 -- support-ticket queue classifier
# ---------------------------------------------------------------------------

_VOCAB = {
    "billing": ["invoice", "charged twice", "payment failed", "card declined", "billing address"],
    "refunds": ["refund", "money back", "return my order", "credit note", "refund status"],
    "tech": ["error on login", "app crashes", "blank screen", "sync fails", "cannot upload"],
    "account": ["reset password", "change email", "delete my account", "two factor", "profile"],
    "shipping": ["where is my package", "tracking number", "delivery late", "courier", "wrong address"],
    "sales": ["pricing", "upgrade plan", "quote for team", "discount", "enterprise plan"],
}
_DESK = {q: f"the {q} desk" for q in _VOCAB}     # agents' sign-off: leaks the queue
_MERGED_FROM = 6                                  # refunds merged into billing in June


def _tickets_month(month: int, relabel: bool = False) -> str:
    import pandas as pd
    rs = np.random.RandomState(200 + month)
    rows = []
    for k in range(60):
        queue = list(_VOCAB)[k % 6]
        words = [_VOCAB[queue][int(rs.randint(0, 5))] for _ in range(2)]
        noise = [_VOCAB[q][int(rs.randint(0, 5))] for q in rs.choice(list(_VOCAB), 1)]
        body = [f"hello, {words[0]} and {words[1]}", f"also {noise[0]}"]
        if queue == "tech" and k % 2:
            body += ["Traceback (most recent call last)", "ValueError: bad state"]
        if rs.rand() < 0.5:
            body += ["----- original message -----", "thanks for reaching out", _DESK[queue]]
        body.append("best regards")
        label = "billing" if (queue == "refunds" and month >= _MERGED_FROM) else queue
        if relabel and k % 9 == 0:
            label = "account"                     # QA's corrected labels, same text
        rows.append({"id": month * 1000 + k, "created": f"2025-{month:02d}-{1 + k % 28:02d} 10:{k % 60:02d}",
                     "channel": ("email", "chat", "web")[k % 3], "subject": words[0],
                     "body": "\n".join(body), "queue": label})
    return pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")


TICKET_FILES = tuple(
    [(f"tickets/tickets_2025-{m:02d}.csv", _tickets_month(m)) for m in range(1, 9)]
    + [("queues.csv", "queue,team,merged_into\nbilling,finance,\nrefunds,finance,billing\n"
                      "tech,support,\naccount,support,\nshipping,ops,\nsales,sales,\n")]
)

TICKETS = (
    ("setup", SETUP),
    ("imports",
     "import json, re, time\n"
     "from pathlib import Path\n"
     "import numpy as np\n"
     "import pandas as pd\n"
     "import matplotlib\n"
     "matplotlib.use('Agg')\n"
     "import matplotlib.pyplot as plt\n"
     "OUT = Path('out')\n"
     "OUT.mkdir(exist_ok=True)\n"
     "def mark(name):\n"
     "    open('calls.log', 'a').write(name + '\\n')\n"),
    ("load",
     "def read_month(path):\n"
     "    df = pd.read_csv(path)\n"
     "    df['created'] = pd.to_datetime(df['created'])\n"
     "    df['month'] = Path(path).stem.replace('tickets_', '')\n"
     "    return df\n"
     "files = sorted(Path('tickets').glob('tickets_*.csv'))\n"
     "raw = pd.concat([read_month(f) for f in files], ignore_index=True)\n"
     "raw = raw.drop_duplicates(subset=['id']).reset_index(drop=True)\n"
     "queues = pd.read_csv('queues.csv')\n"
     "print(len(files), 'files', raw.shape, sorted(raw.queue.unique()))\n"),
    ("clean",
     "LOG_LINE = re.compile(r'^\\s*(Traceback \\(most recent call last\\)|[A-Za-z]+Error:)')\n"
     "SIG_LINE = re.compile(r'^\\s*(sent from my \\w+|--\\s*$|thanks,?$|best regards,?$)', re.I)\n"
     "def clean_text(subject, body):\n"
     "    lines = [ln for ln in str(body).splitlines() if not (LOG_LINE.match(ln) or SIG_LINE.match(ln))]\n"
     "    t = f'{subject} . ' + ' '.join(lines)\n"
     "    return re.sub(r'\\s+', ' ', t).strip().lower()\n"
     "def clean_frame(df):\n"
     "    out = df.copy()\n"
     "    out['body'] = out['body'].fillna('')\n"
     "    out['text'] = [clean_text(s, b) for s, b in zip(out['subject'], out['body'])]\n"
     "    return out\n"
     "tickets = clean_frame(raw)\n"
     "print('tickets', len(tickets), int(tickets['text'].str.len().mean()))\n"),
    ("split",
     "SPLIT = '2025-07'\n"
     "train = tickets[tickets['month'] < SPLIT].reset_index(drop=True)\n"
     "test = tickets[tickets['month'] >= SPLIT].reset_index(drop=True)\n"
     "print('train', train.shape, 'test', test.shape)\n"),
    ("features",
     "from sklearn.feature_extraction.text import TfidfVectorizer\n"
     "from scipy.sparse import hstack\n"
     "def build_features(train_text, test_text):\n"
     "    mark('tfidf')\n"
     "    time.sleep(0.2)              # word + char n-grams over 250k tickets\n"
     "    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)\n"
     "    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, sublinear_tf=True)\n"
     "    Xtr = hstack([word_vec.fit_transform(train_text), char_vec.fit_transform(train_text)]).tocsr()\n"
     "    Xte = hstack([word_vec.transform(test_text), char_vec.transform(test_text)]).tocsr()\n"
     "    return word_vec, char_vec, Xtr, Xte\n"
     "word_vec, char_vec, X_train, X_test = build_features(train['text'], test['text'])\n"
     "y_train, y_test = train['queue'].to_numpy(dtype=object), test['queue'].to_numpy(dtype=object)\n"
     "print(X_train.shape, X_test.shape)\n"),
    ("cv",
     "from sklearn.model_selection import GridSearchCV, StratifiedKFold\n"
     "from sklearn.linear_model import LogisticRegression, SGDClassifier\n"
     "GRIDS = {'logreg': (LogisticRegression(max_iter=300), {'C': [1.0, 4.0]}),\n"
     "         'linsvm': (SGDClassifier(loss='hinge', max_iter=50, tol=1e-4, random_state=0),\n"
     "                    {'alpha': [1e-4, 1e-3]})}\n"
     "def tune(name, est, grid, X, y):\n"
     "    mark('tune')\n"
     "    time.sleep(0.12)\n"
     "    gs = GridSearchCV(est, grid, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring='f1_macro')\n"
     "    return gs.fit(X, y)\n"
     "searches = {name: tune(name, est, grid, X_train, y_train) for name, (est, grid) in GRIDS.items()}\n"
     "cv_table = pd.DataFrame([{'model': n, 'best': json.dumps(s.best_params_), 'cv_f1': round(s.best_score_, 4)}\n"
     "                         for n, s in searches.items()])\n"
     "print(cv_table.to_string())\n"),
    ("fit",
     "from sklearn.base import clone\n"
     "from sklearn.metrics import accuracy_score, f1_score\n"
     "def fit_full(search, X, y):\n"
     "    mark('fit')\n"
     "    return clone(search.best_estimator_).fit(X, y)\n"
     "models = {name: fit_full(s, X_train, y_train) for name, s in searches.items()}\n"
     "preds = {name: m.predict(X_test) for name, m in models.items()}\n"
     "scores = pd.DataFrame([{'model': n, 'accuracy': accuracy_score(y_test, p),\n"
     "                        'f1_macro': f1_score(y_test, p, average='macro')} for n, p in preds.items()]).round(4)\n"
     "print(scores.to_string())\n"),
    ("confusion",
     "from sklearn.metrics import confusion_matrix, precision_recall_fscore_support\n"
     "BEST = 'linsvm'\n"
     "labels = sorted(set(y_train) | set(y_test))\n"
     "y_pred = preds[BEST]\n"
     "cm = confusion_matrix(y_test, y_pred, labels=labels)\n"
     "p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)\n"
     "per_queue = pd.DataFrame({'queue': labels, 'precision': p, 'recall': r, 'f1': f, 'support': s}).round(4)\n"
     "per_queue.to_csv(OUT / 'per_queue.csv', index=False)\n"
     "fig, ax = plt.subplots(figsize=(6, 5))\n"
     "ax.imshow(cm, cmap='Blues')\n"
     "ax.set_xticks(range(len(labels)), labels, rotation=60, ha='right')\n"
     "ax.set_yticks(range(len(labels)), labels)\n"
     "for i in range(len(labels)):\n"
     "    for j in range(len(labels)):\n"
     "        if cm[i, j]:\n"
     "            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7)\n"
     "fig.tight_layout()\n"
     "fig.savefig(OUT / 'confusion_matrix.png', dpi=40, metadata={'Software': None})\n"
     "plt.close(fig)\n"
     "per_queue\n"),
    ("errors",
     "errs = pd.DataFrame({'true': y_test, 'pred': y_pred})\n"
     "pairs = (errs[errs.true != errs.pred].value_counts().rename('n').reset_index()\n"
     "         .sort_values(['n', 'true', 'pred'], ascending=[False, True, True]).head(5).reset_index(drop=True))\n"
     "examples = {}\n"
     "for t_, p_ in pairs[['true', 'pred']].itertuples(index=False):\n"
     "    hit = test[(test.queue == t_) & (y_pred == p_)]\n"
     "    examples[f'{t_} -> {p_}'] = hit['text'].head(2).str[:80].tolist()\n"
     "pairs.to_csv(OUT / 'confused_pairs.csv', index=False)\n"
     "with open(OUT / 'confused_examples.json', 'w', encoding='utf-8') as fh:\n"
     "    json.dump(examples, fh, indent=1)\n"
     "print(len(pairs), 'confused pairs')\n"),
    ("card",
     "best = searches[BEST]\n"
     "card = pd.DataFrame([('model', f'{BEST} {json.dumps(best.best_params_)}'),\n"
     "                     ('train rows', str(X_train.shape[0])), ('test rows', str(X_test.shape[0])),\n"
     "                     ('features', str(X_train.shape[1])),\n"
     "                     ('test macro F1', f\"{scores.set_index('model').loc[BEST, 'f1_macro']:.4f}\")],\n"
     "                    columns=['field', 'value'])\n"
     "card.to_csv(OUT / 'model_card.csv', index=False)\n"
     "print(card.to_string(index=False))\n"),
)

_MERGE = ("merge_map = queues.dropna(subset=['merged_into']).set_index('queue')['merged_into'].to_dict()\n"
          "raw['queue'] = raw['queue'].replace(merge_map)\n")
_QUOTED = "QUOTED = re.compile(r'^-{3,}\\s*original message\\s*-{3,}', re.I | re.M)\n"

TICKET_CLASSIFIER = Session(
    name="r22s2 ticket classifier",
    cells=TICKETS,
    files=TICKET_FILES,
    steps=(
        RunAll(),
        # Tuesday: refunds was merged into billing half-way; map the old queue
        # upstream, then run the cross-validation cell.
        Edit("load", lambda s: s.replace("print(len(files)", _MERGE + "print(len(files)")),
        Run("cv"),
        Run("card"),
        # Quoted agent replies carry the team's sign-off, which leaks the queue:
        # fix the cleaning where it is defined, check the model card.
        Edit("clean", lambda s: s.replace(
            "def clean_text(subject, body):\n    lines = [ln for ln in str(body).splitlines()",
            _QUOTED + "def clean_text(subject, body):\n"
            "    body = QUOTED.split(str(body), maxsplit=1)[0]\n"
            "    lines = [ln for ln in body.splitlines()")),
        Run("card"),
        # Widen the grid by one value.
        Edit("cv", lambda s: s.replace("{'alpha': [1e-4, 1e-3]}", "{'alpha': [1e-4, 1e-3, 1e-2]}")),
        Run("card"),
        # Wednesday, after a restart: QA relabels a batch of old tickets (same
        # file name, same text) -- the vectoriser has nothing new to learn.
        Restart(),
        ReplaceFile("tickets/tickets_2025-03.csv", _tickets_month(3, relabel=True)),
        Run("card"),
        # A new month arrives; final numbers.
        AddFile("tickets/tickets_2025-09.csv", _tickets_month(9)),
        Run("confusion"),
        Run("card"),
        ClearDir("out"),
        RestartAndRunAll(),
        RunAll(calls={"tfidf": 0, "tune": 0}),
    ),
)


SESSIONS = (FINANCE_DECK, DEMAND_FORECAST, AB_READOUT, TICKET_CLASSIFIER)
