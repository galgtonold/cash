"""A loop that handed data to a library call is not run again for a reader of that data.

``for name, model in models.items(): model.fit(X_train, y_train)`` compares
models in a cell above. After an edit above rebuilt ``X_train``, a cell
below that only reads ``X_train`` ran the whole loop again first: on new
inputs the simulation counted every name the loop hands to library code as
changed, a rule meant for a loop that first ran on no items and so never
handed anything. A run where ``fit`` was handed ``X_train`` and left it
unchanged now says so; a name never handed, or handed only under a branch a
run may skip, is still counted.
"""

from __future__ import annotations

import ast
import sys

import pytest

from tests._cell_driver import run_cash_cell

LIB = """
FITS = []


class Counting:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        FITS.append(self.name)
        self.n_ = len(X)
        return self
"""

DATA = "X_train = [1, 2, 3]\ny_train = [0, 1, 0]"
NEW_DATA = "X_train = [1, 2, 3, 4]\ny_train = [0, 1, 0, 1]"
COMPARE = (
    "models = {'a': Counting('a'), 'b': Counting('b')}\n"
    "scores = {}\n"
    "for name, model in models.items():\n"
    "    model.fit(X_train, y_train)\n"
    "    scores[name] = model.n_"
)


@pytest.fixture
def fitlib(tmp_path, monkeypatch, cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    (tmp_path / "fitlib_loopw.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("fitlib_loopw", None)


def _fits_after_an_edit_above(cash_magics, compare: str, setup: str = "") -> list[str]:
    """Run the notebook, rebuild the data above, then run only the reader
    below the loop; what was fitted on the way."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [f"from fitlib_loopw import Counting\n{setup}", DATA, compare, "total = sum(X_train)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    fits = sys.modules["fitlib_loopw"].FITS
    fits.clear()
    cells[1] = NEW_DATA
    run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[3], cells=cells)
    assert cash_magics.shell.user_ns["total"] == 10
    return list(fits)


#: Run pass by pass: what each statement was handed is known from the
#: statements seen running.
PER_PASS = (
    "scores = {}\n"
    "for name in ['a', 'b']:\n"
    "    fitted = Counting(name).fit(X_train, y_train)\n"
    "    scores[name] = fitted.n_"
)


@pytest.mark.usefixtures("fitlib")
@pytest.mark.parametrize("compare", [COMPARE, PER_PASS], ids=["one-unit", "per-pass"])
def test_a_reader_of_the_data_does_not_run_the_loop_again(cash_magics, compare):
    assert _fits_after_an_edit_above(cash_magics, compare) == []


@pytest.mark.usefixtures("fitlib")
def test_a_loop_run_as_one_unit_does_not_run_again_either(cash_magics):
    # Run as one unit, a pass began when a target holds another object.
    compare = COMPARE + "\n    if stop:\n        break"
    assert _fits_after_an_edit_above(cash_magics, compare, setup="stop = False") == []


@pytest.mark.usefixtures("fitlib")
def test_data_handed_only_under_an_untaken_branch_still_runs_the_loop(cash_magics):
    compare = (
        "models = {'a': Counting('a')}\n"
        "for name, model in models.items():\n"
        "    model.fit(y_train, y_train)\n"
        "    if refit:\n"
        "        model.fit(X_train, y_train)"
    )
    assert _fits_after_an_edit_above(cash_magics, compare, setup="refit = False") == ["a"]


@pytest.mark.usefixtures("fitlib")
@pytest.mark.parametrize(("models", "moved"), [("{}", True), ("{'a': Counting('a')}", False)])
def test_a_loop_that_first_ran_on_no_items_is_still_counted(cash_magics, cash_instance, mock_shell, models, moved):
    from cash.notebook.upstream import NotebookSimulator

    loop = "for name, model in models.items():\n    model.fit(X_train, y_train)"
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = ["from fitlib_loopw import Counting", DATA, f"models = {models}\n{loop}"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    simulation = NotebookSimulator(mock_shell, cash_instance, cash_magics.tracking_state).virtual_lineage
    node = ast.parse(loop).body[0]
    assert "X_train" not in simulation.controls.collect_mutations(node, set(), set(), set(), lambda: False)
    changed = simulation.controls.collect_mutations(node, set(), set(), set(), lambda: True)
    assert ("X_train" in changed) is moved


def test_what_a_statement_hands_counts_only_where_every_call_in_it_runs():
    from cash.notebook.loop_watch import surely_handed

    class Model:
        def fit(self, X, y):
            return self

    namespace = {"models": {"a": Model()}, "X": [1], "y": [0], "Z": [2], "flag": False, "model": Model()}
    code = (
        "for name, model in models.items():\n"
        "    model.fit(X, y)\n"
        "    model.fit(Z, y) if flag else None\n"
        "    if flag:\n"
        "        model.fit(y, y)\n"
        "        continue\n"
        "    model.fit(X, X)"
    )
    handed, reach = surely_handed(ast.parse(code).body[0], namespace, lambda _name: False, None)
    # `y` is handed under the `if` too, `Z` only under a conditional
    # expression; the last statement runs only on a pass that did not
    # `continue`.
    assert handed == {0: frozenset({"X"}), 3: frozenset({"X"})}
    assert reach == 3
    while_loop = ast.parse("while flag:\n    model.fit(X, y)").body[0]
    assert surely_handed(while_loop, namespace, lambda _name: False, None) == ({}, 0)
