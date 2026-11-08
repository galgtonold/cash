"""A loop changing the items of a list in place runs as one unit and is stored.

``for est in estimators: est.fit(X)`` changes the objects ``estimators``
holds. Run per iteration, each pass changes its loop variable in place and
runs every time. Run as one unit, the loop stores ``estimators`` with what
it changed, and a hit refills the list with the stored items.

The fitted items hold more than the share check reads (`_DEPTH_READ`
levels), where the unfitted ones did not: the classes with state of the
notebook are taken before the loop runs, so one it never reaches does not
read as one it changed.
"""

from __future__ import annotations

import ast

import pytest

from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    "loop, expected",
    [
        ("for est in estimators:\n    est.fit(X)", {"est"}),
        ("for r in records:\n    r['tax'] = 1", {"r"}),
        ("for i, est in enumerate(estimators):\n    est.n = i", {"est"}),
        ("for line in lines:\n    line = line.strip()", set()),
        ("for est in estimators:\n    print(est)", set()),
    ],
    ids=["a method", "an item assigned", "enumerate", "only rebound", "only read"],
)
def test_the_targets_a_loop_changes(loop, expected):
    from cash.analysis.mutation_effects import changed_loop_targets

    node = ast.parse(loop).body[0]
    assert changed_loop_targets(node, {}, lambda name: False) == expected


SETUP = (
    "import time\n"
    "class Other:\n    registry = []\n"
    "class Model:\n"
    "    def __init__(self, k):\n        self.k = k\n"
    "    def fit(self, X):\n"
    "        time.sleep(0.05)\n"
    "        self.w = [[[[[x * self.k for x in X]]]]]\n"
    "        return self\n"
    "X = [1, 2, 3]"
)
MAKE = "estimators = [Model(1), Model(2)]"
LOOP = "for est in estimators:\n    est.fit(X)"


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _loop_statuses(cash_magics):
    return [(str(s["status"]), s.get("uncacheable_reasons")) for s in cash_magics._last_cell_metrics["statements"]]


@pytest.mark.usefixtures("store_everything")
def test_the_loop_is_stored_and_a_hit_refills_the_list(cash_magics, mock_shell):
    cells = [SETUP, MAKE, LOOP]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = _loop_statuses(cash_magics)
    assert len(first) == 1, first  # one unit, not one entry per pass

    ns = mock_shell.user_ns
    held = ns["estimators"]
    run_cash_cell(cash_magics, MAKE, cells=cells)
    run_cash_cell(cash_magics, LOOP, cells=cells)
    assert "RESTORED" in str(_loop_statuses(cash_magics)) or "CACHED" in str(_loop_statuses(cash_magics))
    assert ns["estimators"] is not held
    assert [m.w[0][0][0][0] for m in ns["estimators"]] == [[1, 2, 3], [2, 4, 6]]
    assert ns["est"] is ns["estimators"][-1]


@pytest.mark.usefixtures("store_everything")
def test_a_loop_over_a_tuple_keeps_its_passes(cash_magics, mock_shell):
    """A tuple is no container a hit refills: the loop keeps one entry per
    pass, as before."""
    cells = [SETUP, MAKE.replace("[", "(").replace("]", ")"), LOOP]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert len(_loop_statuses(cash_magics)) == 2
