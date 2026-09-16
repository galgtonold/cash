"""Round-24 testers' sessions, shrunk to a small dataset.

``machine_alerts`` is r24s3's: hourly sensor files per machine, a cleaning cell
that blanks the hours after each maintenance event (``BLANK_AFTER_H``), a
detector fitted per machine, a sweep over windows and thresholds built with
``sweep_rows.append`` (its scorer reads ``clean_h`` as a global), the best
setting picked in the next cell, and an analysis of the alarm episodes below
it. The blanking window is widened, the cleaning cell and the sweep are re-run,
and the analysis is looked at without running the pick in between (r24s3
WRONG: the analysis kept the pre-edit episodes, silently).

``region_pack`` is r24s2's: weekly sales per region, the house style set once
with ``plt.rcParams.update``, one chart per region saved into a pack folder.
Next morning the kernel is restarted and the charts are redrawn directly
(r24s2 WRONG: every chart came out in matplotlib's default style), the style is
changed, and the pack is rebuilt from an emptied folder.

``doc_export`` is r24s4's: a folder of documents, a sample limit while
developing, token counts exported per document with the file sizes printed
through ``p.stat().st_size``. The sample limit is lifted and the export
re-run (r24s4 WRONG: the printed sizes were the sample run's), a document is
re-delivered, and Restart & Run All.

The expensive functions sleep past the 0.1 s persistence floor and log their
calls, so what is cached in the tester's notebook is cached here too.
"""
from __future__ import annotations

import numpy as np

from session_harness import (AddFile, ClearDir, Edit, ReplaceFile, Restart, RestartAndRunAll,
                             Run, RunAll, Session)

SETUP = "import cash\n%cash_on"


# ---------------------------------------------------------------------------
# r24s3 -- machine failure alerts
# ---------------------------------------------------------------------------

def _sensor_file(m: int) -> str:
    rs = np.random.RandomState(100 + m)
    rows = ["hour,temp,vibration"]
    for h in range(24 * 12):
        drift = 0.02 * max(0, h - 200) if m % 2 == 0 else 0.0     # the even machines degrade
        rows.append(f"{h},{60 + rs.normal(0, 1):.3f},{1 + drift + rs.normal(0, 0.2):.3f}")
    return "\n".join(rows) + "\n"


ALERT_FILES = tuple(
    [(f"sensors/M{m:02d}.csv", _sensor_file(m)) for m in range(4)]
    + [("maintenance.csv", "machine,hour\nM00,40\nM01,90\nM02,150\nM03,60\nM02,230\n"),
       ("failures.csv", "machine,hour\nM00,270\nM02,260\n")]
)

ALERTS = (
    ("setup", SETUP),
    ("imports",
     "import time\n"
     "from pathlib import Path\n"
     "import numpy as np\n"
     "import pandas as pd\n"),
    ("load",
     "hourly = {p.stem: pd.read_csv(p, index_col='hour') for p in sorted(Path('sensors').glob('M*.csv'))}\n"
     "mlog = pd.read_csv('maintenance.csv')\n"
     "failures = pd.read_csv('failures.csv')\n"
     "ids = sorted(hourly)\n"
     "print('machines', len(ids))\n"),
    ("clean",
     "BLANK_AFTER_H = 24\n"
     "events = mlog.groupby('machine').hour.apply(list).to_dict()\n"
     "def blank_after(h, times):\n"
     "    mask = pd.Series(False, index=h.index)\n"
     "    for t in times:\n"
     "        mask |= (h.index > t) & (h.index <= t + BLANK_AFTER_H)\n"
     "    return h.mask(mask)\n"
     "clean_h = {mid: blank_after(h, events.get(mid, [])) for mid, h in hourly.items()}\n"
     "print('missing', round(float(np.mean([h.isna().any(axis=1).mean() for h in clean_h.values()])), 4))\n"),
    ("detector",
     "def fit_detector(h, w):\n"
     "    open('calls.log', 'a').write('fit\\n')\n"
     "    time.sleep(0.12)\n"
     "    r = h['vibration'].interpolate(limit_direction='both').rolling(w, min_periods=1).mean()\n"
     "    med = r.median()\n"
     "    iqr = r.quantile(0.75) - r.quantile(0.25)\n"
     "    return (r - med) / (iqr if iqr else 1.0)\n"),
    ("sweep",
     "WINDOWS = [6, 24]\n"
     "THRESHOLDS = [2.0, 4.0]\n"
     "def score_machine(mid, w):\n"
     "    return fit_detector(clean_h[mid], w)\n"
     "def episodes(s, thr):\n"
     "    a = s > thr\n"
     "    return list(s.index[a & ~a.shift(1, fill_value=False)])\n"
     "def evaluate(sc, thr):\n"
     "    eps = {mid: episodes(s, thr) for mid, s in sc.items()}\n"
     "    n = sum(len(e) for e in eps.values())\n"
     "    tp = sum(1 for mid, e in eps.items() for x in e\n"
     "             if ((failures.machine == mid) & (failures.hour > x) & (failures.hour <= x + 72)).any())\n"
     "    return {'threshold': thr, 'episodes': n, 'precision': tp / n if n else 0.0}\n"
     "sweep_rows = []\n"
     "for w in WINDOWS:\n"
     "    sc = {mid: score_machine(mid, w) for mid in ids}\n"
     "    for thr in THRESHOLDS:\n"
     "        sweep_rows.append({'window_h': w, **evaluate(sc, thr)})\n"
     "sweep = pd.DataFrame(sweep_rows)\n"
     "print(sweep.round(3).to_string())\n"),
    ("pick",
     "best = sweep.sort_values(['precision', 'episodes'], ascending=False).iloc[0]\n"
     "BEST_W, BEST_THR = int(best.window_h), float(best.threshold)\n"
     "best_scores = {mid: score_machine(mid, BEST_W) for mid in ids}\n"
     "print('chosen', BEST_W, BEST_THR)\n"),
    ("analysis",
     "rows = []\n"
     "for mid, s in best_scores.items():\n"
     "    for e in episodes(s, BEST_THR):\n"
     "        rows.append({'machine': mid, 'start': e})\n"
     "ep = pd.DataFrame(rows, columns=['machine', 'start'])\n"
     "print(len(ep), 'episodes', round(float(sum(s.sum() for s in best_scores.values())), 3))\n"),
)

MACHINE_ALERTS = Session(
    name="r24s3 machine alerts",
    cells=ALERTS,
    files=ALERT_FILES,
    steps=(
        RunAll(),
        # Widen the blanking window, re-run the cleaning and the sweep, then
        # look at the analysis without running the pick in between
        # (r24s3 WRONG: the analysis showed the pre-edit episodes).
        Edit("clean", lambda s: s.replace("BLANK_AFTER_H = 24", "BLANK_AFTER_H = 60")),
        Run("clean"),
        Run("sweep"),
        Run("analysis"),
        Run("pick", calls={"fit": 0}),
        # Next day: restart and go straight to the analysis.
        Restart(),
        Run("analysis"),
        RestartAndRunAll(),
        RunAll(calls={"fit": 0}),
    ),
)


# ---------------------------------------------------------------------------
# r24s2 -- regional sales pack
# ---------------------------------------------------------------------------

def _sales() -> str:
    rs = np.random.RandomState(24)
    rows = ["week,region,sales,plan"]
    for region in ("north", "south", "east"):
        for week in range(1, 13):
            plan = 100 + 5 * week
            rows.append(f"{week},{region},{plan + rs.normal(0, 8):.2f},{plan}")
    return "\n".join(rows) + "\n"


PACK = (
    ("setup", SETUP),
    ("imports",
     "import time\n"
     "from pathlib import Path\n"
     "import pandas as pd\n"
     "import matplotlib\n"
     "matplotlib.use('Agg')\n"
     "import matplotlib.pyplot as plt\n"
     "PACK = Path('pack')\n"
     "PACK.mkdir(exist_ok=True)\n"),
    ("style",
     "plt.rcParams.update({'axes.grid': True, 'grid.color': '#CCCCCC', 'axes.facecolor': '#F4F4F4',\n"
     "                     'axes.prop_cycle': matplotlib.cycler(color=['#1F4E79', '#C00000'])})\n"),
    ("load",
     "sales = pd.read_csv('sales.csv')\n"
     "print('rows', len(sales))\n"),
    ("vs_plan",
     "def vs_plan(df):\n"
     "    open('calls.log', 'a').write('vs_plan\\n')\n"
     "    time.sleep(0.12)\n"
     "    out = df.copy()\n"
     "    out['gap'] = (out['sales'] - out['plan']).round(2)\n"
     "    return out\n"
     "by_region = {r: vs_plan(g) for r, g in sales.groupby('region')}\n"
     "print({r: round(float(g['gap'].sum()), 2) for r, g in by_region.items()})\n"),
    ("charts",
     "for region, g in by_region.items():\n"
     "    fig, ax = plt.subplots(figsize=(4, 2.5))\n"
     "    ax.plot(g['week'], g['sales'], label='sales')\n"
     "    ax.plot(g['week'], g['plan'], label='plan')\n"
     "    ax.set_title(region)\n"
     "    ax.legend()\n"
     "    fig.savefig(PACK / f'{region}.png', dpi=40)\n"
     "    plt.close(fig)\n"
     "print(sorted(p.name for p in PACK.glob('*.png')))\n"),
)

REGION_PACK = Session(
    name="r24s2 region pack",
    cells=PACK,
    files=(("sales.csv", _sales()),),
    steps=(
        RunAll(),
        # Next morning: restart and redraw the charts directly
        # (r24s2 WRONG: every chart came out in the default style).
        Restart(),
        Run("charts", calls={"vs_plan": 0}),
        # The style is changed; redraw.
        Edit("style", lambda s: s.replace("'#F4F4F4'", "'#FFFFFF'")),
        Run("charts", calls={"vs_plan": 0}),
        # The pack goes out from an emptied folder.
        ClearDir("pack"),
        RestartAndRunAll(),
        RunAll(calls={"vs_plan": 0}),
    ),
)


# ---------------------------------------------------------------------------
# r24s4 -- document export
# ---------------------------------------------------------------------------

_WORDS = ("invoice", "delivery", "contract", "payment", "late", "refund", "order", "account")


def _doc(i: int, extra: int = 0) -> str:
    rs = np.random.RandomState(400 + i)
    return " ".join(_WORDS[k] for k in rs.randint(0, len(_WORDS), 30 + 7 * i + extra)) + "\n"


DOC_FILES = tuple((f"docs/doc_{i:02d}.txt", _doc(i)) for i in range(8))

EXPORT = (
    ("setup", SETUP),
    ("imports",
     "import time\n"
     "from collections import Counter\n"
     "from pathlib import Path\n"
     "import pandas as pd\n"
     "OUT = Path('out')\n"
     "OUT.mkdir(exist_ok=True)\n"),
    ("load",
     "SAMPLE = 3\n"
     "paths = sorted(Path('docs').glob('*.txt'))[:SAMPLE]\n"
     "texts = {p.stem: p.read_text() for p in paths}\n"
     "print('docs', len(texts))\n"),
    ("count",
     "def count_tokens(text):\n"
     "    open('calls.log', 'a').write('count\\n')\n"
     "    time.sleep(0.12)\n"
     "    return Counter(text.split())\n"
     "counts = {name: count_tokens(t) for name, t in texts.items()}\n"),
    ("export",
     "table = pd.DataFrame(counts).T.fillna(0).astype(int).sort_index(axis=1)\n"
     "table.to_csv(OUT / 'token_counts.csv')\n"
     "pd.Series({n: sum(c.values()) for n, c in counts.items()}, name='tokens').to_csv(OUT / 'totals.csv')\n"
     "print({p.name: p.stat().st_size for p in sorted(OUT.glob('*.csv'))})\n"),
)

DOC_EXPORT = Session(
    name="r24s4 doc export",
    cells=EXPORT,
    files=DOC_FILES,
    steps=(
        RunAll(),
        # Lift the sample limit and export again
        # (r24s4 WRONG: the printed sizes were the sample run's).
        Edit("load", lambda s: s.replace("SAMPLE = 3", "SAMPLE = None")),
        Run("load"),
        Run("export"),
        # A document is re-delivered under the same name.
        ReplaceFile("docs/doc_02.txt", _doc(2, extra=11)),
        Run("export"),
        Restart(),
        Run("export"),
        ClearDir("out"),
        RestartAndRunAll(),
        RunAll(calls={"count": 0}),
    ),
)


SESSIONS = (MACHINE_ALERTS, REGION_PACK, DOC_EXPORT)
