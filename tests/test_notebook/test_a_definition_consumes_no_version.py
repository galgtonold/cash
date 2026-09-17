"""A ``def`` does not consume a version of the globals its body reads.

Round 25's r25s5: after the summary cell had repaired the chain, running the
export cell re-ran the back-test cell again -- ``bt = []``, its loop, and both
``backtest`` statements. The export cell needed ``def plot_region`` re-run, and
the shadowed-variable pass read the def's recorded ``backtest`` as a consumed,
non-final version, so it scheduled the producers of both versions. A function
reads its globals when it is CALLED; at definition time it consumes nothing,
and the call site's own inputs cover what it will read.
"""
from __future__ import annotations

import types

from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner


def _planner() -> ReexecutionPlanner:
    vl = types.SimpleNamespace(shell=types.SimpleNamespace(user_ns={}))
    return ReexecutionPlanner(vl, classifier=None, debug=False)


def _entry(stmt, outputs=(), inputs=(), consumed=None, produced=None):
    return (stmt, set(outputs), list(inputs), dict(consumed or {}), dict(produced or {}), None)


TRACE = [
    _entry("bt = []", ("bt",), (), None, {"bt": "B0"}),                                               # 0
    _entry("backtest = pd.concat(bt)", ("backtest",), ("bt",), {"bt": "B0"}, {"backtest": "L1"}),      # 1
    _entry("backtest['abs_err'] = backtest.pred", ("backtest",), ("backtest",), {"backtest": "L1"},
           {"backtest": "L2"}),                                                                        # 2
    _entry("def plot_region(region):\n    return backtest[backtest.region == region]",
           ("plot_region",), ("backtest",), {"backtest": "L1"}, {"plot_region": "P"}),                # 3
]


def test_rerunning_a_def_does_not_rebuild_the_globals_its_body_reads():
    scheduled = _planner()._complete_shadowed_var_producers([3], TRACE, {"backtest": "L2"})
    assert scheduled == [3], scheduled


def test_a_statement_that_consumed_an_older_version_still_gets_its_producers():
    trace = TRACE[:3] + [_entry("n = len(backtest)", ("n",), ("backtest",), {"backtest": "L1"}, {"n": "N"})]
    scheduled = _planner()._complete_shadowed_var_producers([3], trace, {"backtest": "L2"})
    assert set(scheduled) == {1, 2, 3}, scheduled
