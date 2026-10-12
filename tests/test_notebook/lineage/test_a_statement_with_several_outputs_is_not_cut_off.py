"""The outputs of one statement share its lineage, so none of them takes an
earlier value's lineage because its content came out the same.

``train, test = df[df.day < 7], df[df.day >= 7]`` re-run after rows were
appended: ``train`` holds the same rows as before, ``test`` does not. Giving
``train`` its old lineage named the statement's new lineage by the old one,
and ``test``, built by the same formula, took it too: ``score(test)`` was
served for the old rows. A value rebuilt equal to one output (``test =
train.copy()``) took the lineage the other output had shared as well.
"""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("pandas")

SETUP = (
    "import time\n"
    "import numpy as np, pandas as pd\n"
    "def score(d):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return float(d.x.sum())\n"
)
DATA = "df = pd.DataFrame({'day': np.arange(10), 'x': np.arange(10.0)})"
SPLIT = "train, test = df[df.day < 7], df[df.day >= 7]"


@pytest.mark.parametrize(
    ("edited", "reader", "expected"),
    [
        # Rows appended: train comes out the same, test does not.
        ("df = pd.DataFrame({'day': np.arange(12), 'x': np.arange(12.0)})", "s = score(test)", 7 + 8 + 9 + 10 + 11.0),
        # A value in the first rows fixed: test comes out the same, train does not.
        (
            "df = pd.DataFrame({'day': np.arange(10), 'x': np.r_[100.0, np.arange(1.0, 10.0)]})",
            "s = score(train)",
            100.0 + sum(range(1, 7)),
        ),
    ],
    ids=["train_unchanged", "test_unchanged"],
)
def test_the_changed_output_is_not_served_for_its_old_value(cash_magics, mock_shell, edited, reader, expected):
    """Both halves are tried: which output the statement records first
    follows set order, which string hashing varies per process."""
    cells = [SETUP, DATA, SPLIT, reader]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    cells[1] = edited
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["s"] == expected


def test_a_value_rebuilt_equal_to_one_output_is_not_served_the_other(cash_magics, mock_shell):
    cells = [SETUP, DATA, SPLIT, "s = score(test)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["s"] == 24.0

    cells[2] = SPLIT + "\ntest = train.copy()"
    for cell in cells[2:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["s"] == 21.0


def test_an_input_the_statement_also_changes_is_not_cut_off_with_its_result(cash_magics, mock_shell):
    """``head = add_total(raw)`` adds a column to ``raw`` too: ``head``
    coming out the same must not give the new ``raw`` its old lineage."""
    cells = [
        SETUP + "def add_total(d):\n    d['t'] = d.x * 2\n    return d[['day']].head(2)\n",
        DATA,
        "head = add_total(df)",
        "s = score(df)",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["s"] == 45.0

    cells[1] = "df = pd.DataFrame({'day': np.arange(12), 'x': np.arange(12.0)})"
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["s"] == 66.0


def test_a_single_output_rebuilt_the_same_still_keeps_what_was_built_from_it(cash_magics, mock_shell):
    """Positive control: one output equal to before keeps its lineage."""
    cells = [SETUP, DATA, "train = df[df.day < 7]", "s = score(train)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = cash_magics.tracking_state.variable_lineage["s"]

    cells[1] = "df = pd.DataFrame({'day': np.arange(12), 'x': np.arange(12.0)})"
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["s"] == 21.0
    assert cash_magics.tracking_state.variable_lineage["s"] == before
