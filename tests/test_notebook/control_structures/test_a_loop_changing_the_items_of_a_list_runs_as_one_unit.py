"""A loop changing the items of a list in place runs as one unit and is stored.

``for est in estimators: est.fit(X)`` changes the objects ``estimators``
holds. Run per iteration, each pass changes its loop variable in place and
runs every time. Run as one unit, the loop stores ``estimators`` with what
it changed as an output on every run, and a hit binds it, with every
variable sharing its objects, to the stored copies.

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
def test_the_loop_is_stored_and_a_hit_restores_the_list(cash_magics, mock_shell):
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
    """Items out of a tuple are not walked as the items of a list: the loop
    keeps one entry per pass, as before."""
    cells = [SETUP, MAKE.replace("[", "(").replace("]", ")"), LOOP]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert len(_loop_statuses(cash_magics)) == 2


MODELS = "models = {'a': Model(1), 'b': Model(2)}"


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize(
    "loop",
    ["for est in models.values():\n    est.fit(X)", "for name, est in models.items():\n    est.fit(X)"],
    ids=["values", "items"],
)
def test_a_loop_over_a_dicts_values_is_stored(cash_magics, mock_shell, loop):
    """``.values()`` and ``.items()`` of a plain dict run no code of their
    own: the unit evaluates the header, and a hit binds ``models`` to the
    stored dict of fitted models."""
    cells = [SETUP, MODELS, loop]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert len(_loop_statuses(cash_magics)) == 1

    ns = mock_shell.user_ns
    run_cash_cell(cash_magics, MODELS, cells=cells)
    run_cash_cell(cash_magics, loop, cells=cells)
    assert "RESTORED" in str(_loop_statuses(cash_magics)), _loop_statuses(cash_magics)
    assert {k: m.w[0][0][0][0] for k, m in ns["models"].items()} == {"a": [1, 2, 3], "b": [2, 4, 6]}
    assert ns["est"] is ns["models"]["b"]


@pytest.mark.usefixtures("store_everything")
def test_a_dict_subclass_view_keeps_its_passes(cash_magics, mock_shell):
    """A subclass of dict may define ``values()`` its own way: only a dict
    of the builtin type is walked, and the loop keeps one entry per pass."""
    make = "class Models(dict):\n    pass\n" + MODELS.replace("{", "Models({").replace("}", "})")
    loop = "for est in models.values():\n    est.fit(X)"
    cells = [SETUP, make, loop]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert len(_loop_statuses(cash_magics)) == 2
