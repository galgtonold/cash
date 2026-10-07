"""The speed set: scenarios from real use, each timed plain and with cash.

Decorator scenarios call a function from ``_speed_fixtures`` directly (plain)
and through ``Cash(cache_dir=...).cache`` (cash). Notebook scenarios run the
same cells in a fresh in-process IPython shell with and without ``%cash_on``
(see ``_speed_harness.Notebook``).

Where the scenarios come from: the everyday decorator values; bug hunt 4's
performance findings (a comprehension calling a function per element, cells
building records, a module table read before every cell, seeded randomness,
cost growing with the notebook's length); the slow cells of the JuNE
workflow replays (file loops, model fits, pandas pipelines).

A scenario's sizes are the same in ``--quick`` and full runs, so their ratios
compare; ``--quick`` only takes fewer samples.
"""

from __future__ import annotations

import importlib
import shutil
from collections.abc import Callable

from benchmarks import _speed_fixtures as fx
from benchmarks._speed_harness import Context, Notebook, Scenario, Timing, on_sys_path, stopwatch, total

SCENARIOS: list[Scenario] = []


def scenario(name: str, group: str, what: str, *, needs: tuple[str, ...] = (), metrics=None, clock: str = "auto"):
    def register(fn: Callable[[str, Context], dict[str, Timing]]):
        SCENARIOS.append(
            Scenario(name, group, what, fn, clock=clock, needs=needs, metrics=tuple(metrics or (("", ""),)))
        )
        return fn

    return register


# --------------------------------------------------------------------------
# Decorator
# --------------------------------------------------------------------------


def _cash(ctx: Context, fresh: bool = False):
    """The scenario's Cash over its own cache folder; ``fresh`` builds a new
    instance over the same folder (an empty RAM tier, the entries on disk)."""
    from cash.core import Cash

    if fresh or "cash" not in ctx.state:
        instance = Cash(cache_dir=str(ctx.tmp / ".cash"), register_magic=False)
        if fresh:
            return instance
        ctx.state["cash"] = instance
    return ctx.state["cash"]


def _decorated(ctx: Context, fn):
    memo = ctx.state.setdefault("decorated", {})
    if fn not in memo:
        memo[fn] = _cash(ctx).cache(fn)
    return memo[fn]


def _hits(f) -> int:
    try:
        return int(f.cache_info()["hits"])
    except (AttributeError, KeyError, TypeError):
        return -1


def per_call(t: Timing, calls: int) -> Timing:
    return Timing(t.cpu / calls, t.wall / calls)


# Decorator rows are per call. The plain side makes more calls than the cash
# side where its body is cheap, so neither sample is too short to time well.


def _hit_scenario(name: str, fn, calls: tuple[int, int], what: str):
    plain_calls, cash_calls = calls

    @scenario(f"dec_hit_{name}", "decorator", what)
    def sample(side: str, ctx: Context) -> dict[str, Timing]:
        if side == "plain":
            with stopwatch() as t:
                for _ in range(plain_calls):
                    fn(1)
            return {"": per_call(t[0], plain_calls)}
        f = _decorated(ctx, fn)
        f(1)
        before = _hits(f)
        with stopwatch() as t:
            for _ in range(cash_calls):
                f(1)
        if before >= 0 and _hits(f) - before != cash_calls:
            ctx.note(f"only {_hits(f) - before}/{cash_calls} hits")
        return {"": per_call(t[0], cash_calls)}


def _miss_scenario(name: str, fn, calls: tuple[int, int], what: str):
    plain_calls, cash_calls = calls

    @scenario(f"dec_miss_{name}", "decorator", what)
    def sample(side: str, ctx: Context) -> dict[str, Timing]:
        n_calls = plain_calls if side == "plain" else cash_calls
        f = fn if side == "plain" else _decorated(ctx, fn)
        args = [ctx.fresh() for _ in range(n_calls)]
        with stopwatch() as t:
            for n in args:
                f(n)
        return {"": per_call(t[0], n_calls)}


def _disk_hit_scenario(name: str, fn, calls: tuple[int, int], what: str):
    plain_calls, cash_calls = calls

    @scenario(f"dec_disk_hit_{name}", "decorator", what)
    def sample(side: str, ctx: Context) -> dict[str, Timing]:
        if side == "plain":
            with stopwatch() as t:
                for _ in range(plain_calls):
                    fn(1)
            return {"": per_call(t[0], plain_calls)}
        if not ctx.state.get("primed"):
            _cash(ctx, fresh=True).cache(fn)(1)
            ctx.state["primed"] = True
        wrapped = [_cash(ctx, fresh=True).cache(fn) for _ in range(cash_calls)]
        with stopwatch() as t:
            for f in wrapped:
                f(1)
        if sum(max(_hits(f), 0) for f in wrapped) != cash_calls:
            ctx.note("not every call was a hit")
        return {"": per_call(t[0], cash_calls)}


# (plain calls, cash calls) per sample
_hit_scenario("small_dict", fx.small_dict, (5000, 500), "RAM hit, 10-key dict result")
_hit_scenario("nested_records", fx.nested_records, (20, 20), "RAM hit, 2000 nested record dicts")
_hit_scenario("numpy", fx.numpy_array, (10, 20), "RAM hit, 1M-float array")
_hit_scenario("pandas", fx.pandas_frame, (5, 20), "RAM hit, 50k-row frame")
_hit_scenario("dataclass_list", fx.dataclass_list, (20, 20), "RAM hit, 2000 dataclass instances")
_miss_scenario("small_dict", fx.small_dict, (5000, 100), "miss + store, 10-key dict result")
_miss_scenario("nested_records", fx.nested_records, (20, 10), "miss + store, 2000 nested record dicts")
_miss_scenario("pandas", fx.pandas_frame, (5, 5), "miss + store, 50k-row frame")
_disk_hit_scenario("nested_records", fx.nested_records, (20, 5), "disk hit (empty RAM tier), nested records")
_disk_hit_scenario("pandas", fx.pandas_frame, (5, 5), "disk hit (empty RAM tier), 50k-row frame")


_FIRST_CALL_MODULE = """import math

SCALE = {u}


def helper(x):
    return math.sqrt(x) * SCALE


def target(n):
    total = 0.0
    for i in range(n):
        total += helper(i)
    return {{"n": n, "total": total}}
"""


@scenario("dec_first_call", "decorator", "import a module, decorate, first call (code analysis, key, store)")
def _first_call(side: str, ctx: Context) -> dict[str, Timing]:
    folder = ctx.tmp / "mods"
    folder.mkdir(exist_ok=True)
    names = []
    for _ in range(20 if side == "plain" else 5):
        u = ctx.fresh()
        name = f"speed_first_call_{u}"
        (folder / f"{name}.py").write_text(_FIRST_CALL_MODULE.format(u=u), encoding="utf-8")
        names.append(name)
    importlib.invalidate_caches()
    instance = _cash(ctx) if side == "cash" else None
    with on_sys_path(folder, tuple(names)):
        with stopwatch() as t:
            for name in names:
                target = importlib.import_module(name).target
                (instance.cache(target) if instance else target)(50)
    return {"": per_call(t[0], len(names))}


@scenario("dec_keying", "decorator", "RAM hit keyed on an array, a frame, a config dict and a name list")
def _keying(side: str, ctx: Context) -> dict[str, Timing]:
    import numpy as np
    import pandas as pd

    if "args" not in ctx.state:
        rng = np.random.default_rng(0)
        ctx.state["args"] = (
            rng.random(100_000),
            pd.DataFrame({"a": rng.random(10_000), "b": rng.integers(0, 9, 10_000)}),
            {"lr": 0.1, "layers": [64, 32], "name": "exp1"},
            [f"feature_{i}" for i in range(100)],
        )
    args = ctx.state["args"]
    f = fx.keyed_args if side == "plain" else _decorated(ctx, fx.keyed_args)
    f(*args)
    calls = 5000 if side == "plain" else 50
    with stopwatch() as t:
        for _ in range(calls):
            f(*args)
    return {"": per_call(t[0], calls)}


# --------------------------------------------------------------------------
# Notebook
# --------------------------------------------------------------------------


def _notebook(side: str, ctx: Context, cells: list[str]) -> Notebook:
    workdir = ctx.tmp / f"{side}-{ctx.fresh()}"
    nb = Notebook(cells, cash_on=(side == "cash"), workdir=workdir)
    previous = ctx.state.pop("cleanup", None)
    if previous:
        previous()

    def cleanup():
        nb.close()
        shutil.rmtree(workdir, ignore_errors=True)

    ctx.state["cleanup"] = cleanup
    return nb


def _same_value(side: str, ctx: Context, nb: Notebook, expr: str) -> None:
    """Note in the report when cash's value differs from the plain run's."""
    value = repr(nb.peek(expr))
    key = f"value:{expr}"
    if side == "plain":
        ctx.state[key] = value
    elif key in ctx.state and ctx.state[key] != value:
        ctx.note(f"VALUE DIFFERS: {expr}")


EVERYDAY = [
    "import json, math\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    "rng = np.random.default_rng(0)\nN = 5_000",
    "df = pd.DataFrame({'user': rng.integers(0, 200, N), 'amount': rng.random(N) * 100, 'day': rng.integers(1, 31, N)})",
    "df.head()",
    "df['amount_eur'] = df['amount'] * 0.92",
    "by_user = df.groupby('user')['amount'].sum()",
    "top = by_user.sort_values(ascending=False).head(10)\ntop",
    "def describe(s):\n    return {'mean': float(s.mean()), 'max': float(s.max())}",
    "stats = describe(df['amount'])\nprint(stats)",
    "daily = df.pivot_table(index='day', values='amount', aggfunc='mean')",
    "fig, ax = plt.subplots()\nax.plot(daily.index, daily['amount'])\nax.set_title('daily mean')",
    "names = [f'user_{i}' for i in range(200)]",
    "lookup = dict(zip(range(200), names))",
    "df['name'] = df['user'].map(lookup)",
    "big = df[df['amount'] > 90]\nlen(big)",
    "arr = rng.random((200, 50))",
    "cov = np.cov(arr, rowvar=False)",
    "eig = np.linalg.eigvalsh(cov)",
    "print(f'largest eigenvalue {eig[-1]:.3f}')",
    "cfg = {'lr': 0.1, 'epochs': 3, 'layers': [64, 32]}",
    "cfg_json = json.dumps(cfg)",
    "merged = df.merge(by_user.rename('total').reset_index(), on='user')",
    "share = merged['amount'] / merged['total']",
    "summary = {'rows': len(df), 'users': int(df['user'].nunique()), 'share_max': float(share.max())}",
    "summary",
    "plt.close('all')",
]


@scenario(
    "nb_everyday",
    "notebook",
    "26 everyday cells (pandas, numpy, a plot, prints)",
    needs=("pandas", "matplotlib"),
    metrics=(("first", "everyday notebook, first Run All"), ("rerun", "everyday notebook, second Run All")),
)
def _everyday(side: str, ctx: Context) -> dict[str, Timing]:
    nb = _notebook(side, ctx, EVERYDAY)
    first = total(nb.run())
    rerun = total(nb.run())
    _same_value(side, ctx, nb, "summary")
    return {"first": first, "rerun": rerun}


@scenario(
    "nb_slow_cell_rerun",
    "notebook",
    "re-run of a 0.2 s pure-Python cell (should be a hit, ratio far below 1)",
)
def _slow_cell(side: str, ctx: Context) -> dict[str, Timing]:
    cells = ["N = 400_000", "total = sum(i * i % 7 for i in range(N * 3))", "out = total + 1"]
    nb = _notebook(side, ctx, cells)
    nb.run()
    rerun = nb.run([2])
    _same_value(side, ctx, nb, "total")
    return {"": rerun[0]}


def _long_cells(n: int) -> list[str]:
    cells = ["import numpy as np"]
    for i in range(1, n):
        kind = i % 5
        if kind == 0:
            cells.append(f"v{i} = {i} * 2")
        elif kind == 1:
            cells.append(f"def fn{i}(x):\n    return x + {i}")
        elif kind == 2:
            cells.append(f"print('step', {i})")
        elif kind == 3:
            cells.append(f"a{i} = np.arange(10) + {i}")
        else:
            cells.append(f"d{i} = {{'k': {i}, 'v': [{i}, {i} + 1]}}")
    return cells


LONG_N = 300


@scenario(
    "nb_long",
    "notebook",
    f"{LONG_N} small cells, Run All",
    metrics=(
        ("cell20", "per cell, cells 11-30 of a long notebook"),
        (f"cell{LONG_N}", f"per cell, cells {LONG_N - 19}-{LONG_N} of a long notebook"),
    ),
)
def _long(side: str, ctx: Context) -> dict[str, Timing]:
    nb = _notebook(side, ctx, _long_cells(LONG_N))
    t = nb.run()
    early, late = total(t[10:30]), total(t[-20:])
    return {"cell20": Timing(early.cpu / 20, early.wall / 20), f"cell{LONG_N}": Timing(late.cpu / 20, late.wall / 20)}


@scenario(
    "nb_cheap_calls",
    "notebook",
    "a comprehension or loop calling a function per element, 20k elements, first run",
    metrics=(
        ("user_fn", "[f(i) for i in range(20k)], first run"),
        ("json_dumps", "[json.dumps({...}) for i in range(20k)], first run"),
        ("for_append", "for-loop appending f(i), 20k, first run"),
    ),
)
def _cheap_calls(side: str, ctx: Context) -> dict[str, Timing]:
    cells = [
        "import json\nN = 20_000\ndef f(i):\n    return i * 2",
        "d = [f(i) for i in range(N)]",
        "a = [json.dumps({'i': i}) for i in range(N)]",
        "g = []\nfor i in range(N):\n    g.append(f(i))",
    ]
    nb = _notebook(side, ctx, cells)
    t = nb.run()
    _same_value(side, ctx, nb, "(sum(d), a[-1], sum(g))")
    return {"user_fn": t[1], "json_dumps": t[2], "for_append": t[3]}


@scenario(
    "nb_records",
    "notebook",
    "cells building plain Python data (records, an index, a dict of lists), 50k elements",
    metrics=(
        ("first", "build records/index/groups, first run (3 cells)"),
        ("rerun_recs", "re-run: list of 50k record dicts"),
        ("rerun_index", "re-run: dict index of 50k keys"),
        ("rerun_groups", "re-run: dict of 16k small lists"),
    ),
)
def _records(side: str, ctx: Context) -> dict[str, Timing]:
    cells = [
        "N = 50_000",
        "recs = [{'id': i, 'user': 'u', 'amount': i * 0.5} for i in range(N)]",
        "index = {f'k{i}': i for i in range(N)}",
        "groups = {f'g{i}': [i, i + 1] for i in range(N // 3)}",
    ]
    nb = _notebook(side, ctx, cells)
    first = nb.run()
    again = nb.run()
    _same_value(side, ctx, nb, "(len(recs), recs[-1], index['k7'], groups['g9'])")
    return {"first": total(first[1:]), "rerun_recs": again[1], "rerun_index": again[2], "rerun_groups": again[3]}


_HELPERS = """import numpy as np
EMB = np.random.default_rng(0).random((250_000, 8))   # 16 MB table loaded at import


def score(i):
    return float(EMB[i].sum())
"""


@scenario(
    "nb_module_data",
    "notebook",
    "per trivial cell, after a cell read a module's 16 MB table through its function",
)
def _module_data(side: str, ctx: Context) -> dict[str, Timing]:
    u = ctx.fresh()
    name = f"speed_helpers_{u}"
    workdir = ctx.tmp / f"mod-{u}"
    workdir.mkdir()
    (workdir / f"{name}.py").write_text(_HELPERS, encoding="utf-8")
    importlib.invalidate_caches()
    cells = [f"import {name}", f"v = {name}.score(3)"] + [f"x{i} = {i}" for i in range(20)]
    with on_sys_path(workdir, (name,)):
        nb = _notebook(side, ctx, cells)
        t = nb.run()
    trivial = total(t[2:])
    return {"": Timing(trivial.cpu / 20, trivial.wall / 20)}


@scenario(
    "nb_seeded_rerun",
    "notebook",
    "second Run All of np.random.seed(42) + slow seeded draws (should hit)",
)
def _seeded(side: str, ctx: Context) -> dict[str, Timing]:
    cells = [
        "import numpy as np\nnp.random.seed(42)",
        "def make_data(n):\n    s = 0\n    for i in range(300_000):\n        s += i % 3\n    return np.random.normal(size=n) + s * 0",
        "X = make_data(1000)",
        "noise = np.random.normal(size=1000) + sum(i % 5 for i in range(300_000)) * 0",
        "m = float(X.mean() + noise.mean())",
    ]
    nb = _notebook(side, ctx, cells)
    nb.run()
    rerun = total(nb.run())
    _same_value(side, ctx, nb, "m")
    return {"": rerun}


def _data_folder(ctx: Context):
    folder = ctx.tmp / "data"
    if not folder.exists():
        folder.mkdir()
        for k in range(100):
            text = "\n".join(f"{k},{i},event_{i % 13},value={i * 0.25}" for i in range(200))
            (folder / f"log_{k:03d}.txt").write_text(text + "\n", encoding="utf-8")
        for k in range(20):
            rows = "\n".join(f"{i},{k},{i * 1.5},u{i % 40}" for i in range(500))
            (folder / f"part_{k:02d}.csv").write_text("id,part,amount,user\n" + rows + "\n", encoding="utf-8")
    return folder


@scenario(
    "nb_file_loop",
    "notebook",
    "a loop reading 100 text files plus 20 CSVs with pandas",
    needs=("pandas",),
    metrics=(("first", "file-reading cells, first run"), ("rerun", "file-reading cells, re-run")),
)
def _file_loop(side: str, ctx: Context) -> dict[str, Timing]:
    folder = _data_folder(ctx)
    cells = [
        f"from pathlib import Path\nimport pandas as pd\nDATA = Path({str(folder)!r})",
        "lines = []\nfor p in sorted(DATA.glob('*.txt')):\n    with open(p, encoding='utf-8') as fh:\n        lines.extend(fh.read().splitlines())",
        "frames = [pd.read_csv(p) for p in sorted(DATA.glob('*.csv'))]",
        "table = pd.concat(frames, ignore_index=True)",
        "n = len(lines) + len(table)",
    ]
    nb = _notebook(side, ctx, cells)
    first = total(nb.run()[1:])
    rerun = total(nb.run()[1:])
    _same_value(side, ctx, nb, "n")
    return {"first": first, "rerun": rerun}


@scenario(
    "nb_sklearn_fit",
    "notebook",
    "fit a logistic regression and a 30-tree random forest on 2000x20",
    needs=("sklearn",),
    metrics=(("first", "model-fitting cells, first run"), ("rerun", "model-fitting cells, re-run")),
)
def _sklearn(side: str, ctx: Context) -> dict[str, Timing]:
    cells = [
        "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier",
        "rng = np.random.default_rng(0)\nX = rng.random((2000, 20))\ny = (X[:, 0] + X[:, 1] > 1).astype(int)",
        "lr = LogisticRegression(max_iter=300).fit(X, y)",
        "rf = RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=1).fit(X, y)",
        "score = lr.score(X, y) + rf.score(X, y)",
    ]
    nb = _notebook(side, ctx, cells)
    first = total(nb.run()[1:])
    rerun = total(nb.run()[1:])
    _same_value(side, ctx, nb, "score")
    return {"first": first, "rerun": rerun}
