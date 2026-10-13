"""A value restored from the cache counts for the same-value cutoff.

The cutoff gives a frame, series or array that a statement executed its
earlier lineage when an equal value was seen before. Only executed values
were seen: after a restart, where most values come back from the cache, a
value rebuilt equal to a restored one (an A/B notebook restricting its frame
back to day 1 after day 2 arrived) took a new lineage, and the 9 s bootstrap
over it ran again. An entry now keeps the digest its value had when it ran.
"""

import pytest

from cash.notebook.ipython.magics import CashMagics
from cash.notebook.value_cutoff import ValueCutoff
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
PICK = "train = df[df.day < 7][['x']]  # @cash:persist"
SCORE = "s = score(train)"


def test_a_restored_value_lends_its_lineage_to_an_equal_one(cash_magics, mock_shell, cash_instance):
    cells = [SETUP, DATA, PICK, SCORE]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = dict(cash_magics.tracking_state.variable_lineage)

    # A restarted kernel over the same cache: a second CashMagics (the
    # fixtures' shell and Cash, an empty namespace), which has seen nothing.
    for name in ("df", "train", "s", "score", "np", "pd", "time"):
        mock_shell.user_ns.pop(name, None)
    restarted = CashMagics(mock_shell, cash_instance)
    for cell in cells:
        run_cash_cell(restarted, cell, cells=cells)
    state = restarted.tracking_state
    assert state.variable_lineage["train"] == first["train"]  # restored, not run

    # `df` changes in a column `train` leaves out: `train` runs and comes out
    # equal to the restored one.
    cells[1] = "df = pd.DataFrame({'day': np.arange(10), 'x': np.arange(10.0), 'y': 1})"
    for cell in cells[1:]:
        run_cash_cell(restarted, cell, cells=cells)

    assert mock_shell.user_ns["s"] == 21.0
    assert state.variable_lineage["train"] == first["train"]
    assert state.variable_lineage["s"] == first["s"]


def test_a_changed_value_does_not_take_the_restored_lineage(cash_magics, mock_shell, cash_instance):
    """Negative control: a different ``train`` keeps a lineage of its own."""
    cells = [SETUP, DATA, PICK, SCORE]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = dict(cash_magics.tracking_state.variable_lineage)

    for name in ("df", "train", "s", "score", "np", "pd", "time"):
        mock_shell.user_ns.pop(name, None)
    restarted = CashMagics(mock_shell, cash_instance)
    for cell in cells:
        run_cash_cell(restarted, cell, cells=cells)

    cells[1] = "df = pd.DataFrame({'day': np.arange(10), 'x': np.arange(1.0, 11.0)})"
    for cell in cells[1:]:
        run_cash_cell(restarted, cell, cells=cells)

    assert mock_shell.user_ns["s"] == 28.0
    assert restarted.tracking_state.variable_lineage["train"] != first["train"]


def test_the_first_value_seen_keeps_its_lineage():
    cutoff = ValueCutoff()
    np = pytest.importorskip("numpy")
    seen = cutoff.lineage_for(np.arange(5.0), "a" * 64, _content)
    assert seen == "a" * 64
    digest = cutoff.digest_of(seen)
    assert digest is not None

    # Restored later with another lineage: the one seen first stays.
    cutoff.remember(digest, "b" * 64)
    assert cutoff.lineage_for(np.arange(5.0), "c" * 64, _content) == "a" * 64


def test_a_remembered_value_lends_its_lineage():
    cutoff = ValueCutoff()
    np = pytest.importorskip("numpy")
    probe = ValueCutoff()
    probe.lineage_for(np.arange(5.0), "a" * 64, _content)
    digest = probe.digest_of("a" * 64)

    cutoff.remember(digest, "b" * 64)

    assert cutoff.lineage_for(np.arange(5.0), "c" * 64, _content) == "b" * 64
    assert cutoff.lineage_for(np.arange(6.0), "d" * 64, _content) == "d" * 64


def _content(value):
    return value.tobytes().hex()
