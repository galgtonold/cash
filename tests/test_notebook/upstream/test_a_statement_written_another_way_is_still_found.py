"""A statement is found in a cell however the cell writes it.

cash records the statement that last changed a variable unparsed
(``df.drop(columns=['b'], inplace=True)``). The checks that ask which cell
holds it searched the cell text for that form, so a cell writing
``df.drop(columns=["b"], inplace=True)`` (other quotes, other spacing, a
comment) did not hold it, and the check took the statement for one in another
cell, or in none. Each test below writes the statement another way than cash
records it.
"""

from __future__ import annotations

import warnings

import pytest

from cash.analysis.mutation_effects import CellEffects
from cash.notebook.lineage_store import tag_value
from tests._cell_driver import run_cash_cell


def _run_all(cash_magics, cells):
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)


def test_a_cell_above_changing_a_frame_in_place_runs_once_in_a_run_from_the_top(cash_magics):
    """``df["b"] = ...`` above, ``df.drop(columns=["b"], inplace=True)``
    here: the last write is in the cell above, so the frame this cell starts
    from is the one that cell left. Not found there, the first run of this
    cell took the frame for its own earlier output and rebuilt it, running
    the cell above a second time."""
    pytest.importorskip("pandas")
    cells = [
        "import pandas as pd\nRUNS = []\ndef twice(s):\n    RUNS.append(1)\n    return s * 2\n"
        "df = pd.DataFrame({'a': [1, 2]})",
        'df["b"] = twice(df["a"])',
        'df.drop(columns=["b"], inplace=True)',
    ]
    _run_all(cash_magics, cells)
    ns = cash_magics.shell.user_ns
    assert ns["RUNS"] == [1], "the cell above ran again in a run from the top"
    assert ns["df"].to_dict("list") == {"a": [1, 2]}


def test_a_name_made_above_is_no_partner_below(cash_magics):
    """``cfg = {"f": feats}`` is in the cell above: re-running the cell that
    appends to ``feats`` rebuilds both, and no name below shares the list.
    Not found above, ``cfg`` was taken for one a cell below made, and
    NOTEBOOK-SHARED-KEPT said a cell below shares ``feats``."""
    cells = ['feats = []\ncfg = {"f": feats}  # the settings', "feats.append(1)", "n = 1"]
    _run_all(cash_magics, cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[1], cells=cells)
    said = [str(w.message) for w in caught if "NOTEBOOK-SHARED-KEPT" in str(w.message)]
    ns = cash_magics.shell.user_ns
    assert ns["feats"] == [1]
    assert ns["cfg"]["f"] is ns["feats"]
    assert not said, said


def test_a_cell_rebinding_its_own_input_is_its_producer(cash_magics, monkeypatch):
    """The stale-value guard keeps a reassigned input the cells above made
    anew (an unsaved edit above), but never one this cell made itself: re-run
    on its own output, ``df = df.assign(a=df["a"] * 2)`` would double ``a``
    again. The statement was looked for in the cell as text, so written with
    other quotes it was taken for one above, and the input kept."""
    guard = cash_magics._upstream_checker.simulator.stale_values
    state = cash_magics.tracking_state

    class Frame:
        pass

    df = Frame()
    tag_value(df, "live")
    cash_magics.shell.user_ns["df"] = df
    state.lineage.record("df", "recorded")
    state.executed_cell_codes["df"] = "df = df.assign(a=df['a'] * 2)"
    monkeypatch.setattr(guard.unsaved_edits, "is_valid_extension", lambda *a, **k: True)
    cells = ["df = make()", 'df = df.assign(a=df["a"]  *  2)  # double it']
    effects = CellEffects(outputs=frozenset({"df"}), reassigned=frozenset({"df"}))
    broken: set[str] = set()

    guard.mark_stale_value_inputs_broken({"df"}, effects, broken, cells, 1, {"df": "virtual"})

    assert "df" in broken


def test_a_function_holding_a_value_changed_below_is_said_to_drop_it(cash_magics):
    """The cell reads ``draw``, which holds ``BIG``, and a cell below
    changed ``BIG`` in place: NOTEBOOK-CHANGED-BELOW says the cell gets it
    without the change. The cells below were searched as text for
    ``BIG[0] = 'x'``, which a cell writing ``BIG[0] = "x"`` does not hold,
    and nothing was said."""
    from cash.notebook.upstream._types import CellCheck

    classifier = cash_magics._upstream_checker.simulator.classifier
    ns = cash_magics.shell.user_ns
    big = [1, 2]

    def draw(k, rows=big):
        return rows[k]

    ns["BIG"], ns["draw"] = big, draw
    cash_magics.tracking_state.executed_cell_codes["BIG"] = "BIG[0] = 'x'"
    cells = ["BIG = [1, 2]\ndef draw(k, rows=BIG): ...", "y = draw(1)", 'BIG[0] = "x"  # first']
    check = CellCheck(current_cell_idx=1, notebook_cells=cells)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier._warn_held_changed_below({"draw"}, check)

    said = [str(w.message) for w in caught if "NOTEBOOK-CHANGED-BELOW" in str(w.message)]
    assert said and "reads 'draw', which holds 'BIG'" in said[0], said
