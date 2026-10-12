"""What a verdict learned from earlier runs is learned again for, on new inputs.

An empty ``[heapq.heappush(h, x) for x in pending]`` changed nothing, which
says nothing about the next items, so what a statement or a loop hands
straight to library code is watched again. What only a helper of the user's
is handed (``bt = run_backtest(series, 3)``), or what a draw on an Axes is
handed as data (``ax.bar(names, totals)``), keeps the verdict of its first
run: counting it as changed whenever the inputs move made every reader of
``series`` wait for the backtest, and re-drew a chart nothing reads.

The helper's own handoffs leave alone the globals no call changes in place
(a ``Path``) and a random generator, whose draws are followed as a carrier's
are: ``def slow_col(n): return rng.random(n)`` must not make ``rng`` a
changed global of every statement calling it.

A loop's trust is dropped when a data input changed since it last ran, also
when the loop lists that input among what it may change.
"""

from __future__ import annotations

import ast
import heapq
import io
import pathlib
import random
from types import SimpleNamespace

import numpy as np
import pytest

from cash.analysis.namespace_effects import callee_handed_globals, rewatch_exempt
from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.loop_rules import LoopRules
from cash.source_norm import exact_source_digest

SOURCES = {
    "score": "def score(feats, thr):\n    return [f for f in feats if f > thr]\n",
    "run_backtest": "def run_backtest(series, n):\n    return list(series.iterrows())\n",
    "slow_col": "def slow_col(n):\n    return rng.random(n)\n",
    "seeded": "def seeded(n):\n    return random.random() * n\n",
    "log": "def log(msg):\n    with LOG.open('a') as fh:\n        fh.write(msg)\n",
    "push_global": "def push_global(x):\n    heapq.heappush(H, x)\n",
}


def _namespace() -> dict:
    ns: dict = {"heapq": heapq, "random": random, "np": np}
    for source in SOURCES.values():
        exec(source, ns)  # the helpers under test
    ns.update(
        data=[3, 1],
        n=2,
        h=[],
        H=[],
        pending=[5],
        rows=[],
        rng=np.random.default_rng(0),
        LOG=pathlib.Path("unused.log"),
        buf=io.BytesIO(),
    )
    return ns


def _exempt(stmt: str, ns: dict | None = None) -> set[str]:
    return set(rewatch_exempt(ast.parse(stmt), ns if ns is not None else _namespace(), SOURCES.get))


@pytest.mark.parametrize(
    ("stmt", "exempt"),
    [
        ("bt = run_backtest(data, 3)", {"data"}),
        ("out = score(data, n)", {"data", "n"}),
        ("print(score(data, n))", {"data", "n"}),
        ("rows.append(dict(k=n, err=score(data, n)))", {"data"}),
        ("for k in pending:\n    rows.append(score(data, k))", {"data"}),
    ],
)
def test_what_only_a_helper_is_handed_keeps_its_verdict(stmt, exempt):
    assert _exempt(stmt) == exempt


@pytest.mark.parametrize(
    "stmt",
    [
        "_ = [heapq.heappush(h, x) for x in pending]",
        "np.copyto(data, h)",
        # What the helper hands back is changed in the statement itself.
        "score(data, n)[0].append(1)",
        "np.copyto(score(data, n), h)",
        "a, b = score(data, n)",
    ],
)
def test_what_reaches_library_code_is_watched_again(stmt):
    assert "data" not in _exempt(stmt)
    assert "h" not in _exempt(stmt)


def test_a_name_also_handed_elsewhere_is_watched_again():
    assert _exempt("out = score(data, n); data.sort()") == {"n"}


def test_a_draw_reads_its_data_but_not_a_buffer():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        ns = _namespace()
        ns.update(fig=fig, ax=ax, names=["a", "b"], totals=[1, 2])
        assert _exempt("ax.bar(names, totals)", ns) == {"names", "totals"}
        assert _exempt("fig.savefig(buf)", ns) == set()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("stmt", ["c = slow_col(n)", "s = seeded(n)", "log('x')"])
def test_a_helper_hands_on_no_generator_module_or_path(stmt):
    assert callee_handed_globals(ast.parse(stmt), _namespace(), SOURCES.get) == frozenset()


def test_a_helper_still_hands_on_a_global_list():
    assert callee_handed_globals(ast.parse("push_global(1)"), _namespace(), SOURCES.get) == frozenset({"H"})


LOOP = "for k in (2, 5, 20):\n    rows.append(score(raw, k))"


def _rules(entry_lineages: dict[str, str]) -> LoopRules:
    tracking = SimpleNamespace(
        control_outcomes={exact_source_digest(LOOP): (entry_lineages, {}, (), "")},
        variable_lineage={},
    )
    return LoopRules(SimpleNamespace(user_ns={}), tracking)


def _trace(raw_now: str) -> list[TraceEntry]:
    return [
        TraceEntry("raw = load()", {"raw"}, set(), {}, {"raw": raw_now}, False),
        TraceEntry("rows = []", {"rows"}, set(), {}, {"rows": "r0"}, False),
        # The loop lists `raw` among what it may change (it is handed to a
        # helper), so `raw` is an output of the loop and loop-derived too.
        TraceEntry(LOOP, {"rows", "k", "raw"}, {"rows", "raw", "score"}, {"rows": "r0", "raw": raw_now}, {}, False),
    ]


def test_a_loop_whose_data_changed_loses_its_trust_when_it_lists_the_data():
    rules = _rules({"rows": "r0", "raw": "old"})
    changed = rules.loops_reading_changed_data({"rows"}, _trace("new"), {"k"}, {"rows", "k", "raw"})
    assert changed == {"rows"}


def test_a_loop_on_the_same_data_keeps_its_trust():
    rules = _rules({"rows": "r0", "raw": "same"})
    assert rules.loops_reading_changed_data({"rows"}, _trace("same"), {"k"}, {"rows", "k", "raw"}) == set()
