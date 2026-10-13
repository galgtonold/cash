"""A statement binding several values that all come out the same as an
earlier run's keeps the lineage they had, so what was built on them is served.

``U, V, hist = recsys.als_fit(C, ..., tol=1e-3)`` returned the factors the fit
without ``tol`` had, and ``evaluate(U, V, R, test_item)`` ran again (3.7 s):
the same-value cutoff skipped every statement with several outputs, since
they share one lineage and a cutoff of one would name the others too
(`test_a_statement_with_several_outputs_is_not_cut_off.py`). All of them
together may take the lineage they shared, when an earlier statement bound
the same names to values with the same content.
"""

import pytest

from cash.notebook.value_cutoff import ValueCutoff
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")


def _setup(log):
    return (
        "import os, time\n"
        "import numpy as np\n"
        "def als_fit(n, tol=None, bump=0.0):\n"
        "    return np.arange(n) * 1.0, np.ones(n) + bump, [0.5, 0.25]\n"
        "def evaluate(U, V):\n"
        f"    fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return float(U @ V)\n"
    )


def _runs(log) -> int:
    return len(log.read_bytes()) if log.exists() else 0


def _run(magics, cells, start=0):
    for cell in cells[start:]:
        run_cash_cell(magics, cell, cells=cells)


def test_a_refit_returning_the_same_factors_serves_what_was_built_on_them(cash_magics, mock_shell, tmp_path):
    log = tmp_path / "evaluate.log"
    cells = [_setup(log), "U, V, hist = als_fit(5)", "m = evaluate(U, V)"]
    _run(cash_magics, cells)
    assert _runs(log) == 1
    lineage = dict(cash_magics.tracking_state.variable_lineage)

    cells[1] = "U, V, hist = als_fit(5, tol=1e-3)"
    _run(cash_magics, cells, 1)

    assert mock_shell.user_ns["m"] == 10.0
    assert _runs(log) == 1, "evaluate ran again on the same factors"
    for name in ("U", "V", "hist", "m"):
        assert cash_magics.tracking_state.variable_lineage[name] == lineage[name]


@pytest.mark.parametrize(
    "refit",
    ["U, V, hist = als_fit(5, tol=1e-3, bump=1.0)", "U, V, hist = als_fit(6, tol=1e-3)"],
    ids=["one-output-changed", "all-changed"],
)
def test_a_refit_with_any_output_changed_runs_what_is_built_on_them(cash_magics, mock_shell, tmp_path, refit):
    """Negative control: ``U`` comes out the same while ``V`` does not."""
    log = tmp_path / "evaluate.log"
    cells = [_setup(log), "U, V, hist = als_fit(5)", "m = evaluate(U, V)"]
    _run(cash_magics, cells)
    lineage = dict(cash_magics.tracking_state.variable_lineage)

    cells[1] = refit
    _run(cash_magics, cells, 1)

    assert _runs(log) == 2
    assert mock_shell.user_ns["m"] == float(mock_shell.user_ns["U"] @ mock_shell.user_ns["V"])
    assert cash_magics.tracking_state.variable_lineage["U"] != lineage["U"]


def test_other_names_for_the_same_values_keep_their_own_lineage(cash_magics, tmp_path):
    """``V, U, hist = ...`` binds the names to other values: no cutoff."""
    log = tmp_path / "evaluate.log"
    cells = [_setup(log), "U, V, hist = als_fit(5)", "m = evaluate(U, V)"]
    _run(cash_magics, cells)

    cells[1] = "V, U, hist = als_fit(5, tol=1e-3)"
    _run(cash_magics, cells, 1)

    assert _runs(log) == 2


class TestTheCutoff:
    def _values(self, bump=0.0):
        return {"U": np.arange(3.0), "V": np.ones(3) + bump, "hist": [0.5, 0.25]}

    def test_all_outputs_equal_take_the_earlier_shared_lineage(self):
        cutoff = ValueCutoff()
        first = cutoff.lineage_for_outputs(self._values(), dict.fromkeys(("U", "V", "hist"), "L1"), compute_hash)
        assert set(first.values()) == {"L1"}
        again = cutoff.lineage_for_outputs(self._values(), dict.fromkeys(("U", "V", "hist"), "L2"), compute_hash)
        assert again == dict.fromkeys(("U", "V", "hist"), "L1")

    def test_one_output_different_keeps_the_new_lineage(self):
        cutoff = ValueCutoff()
        cutoff.lineage_for_outputs(self._values(), dict.fromkeys(("U", "V", "hist"), "L1"), compute_hash)
        again = cutoff.lineage_for_outputs(self._values(1.0), dict.fromkeys(("U", "V", "hist"), "L2"), compute_hash)
        assert set(again.values()) == {"L2"}

    def test_an_output_whose_content_is_not_all_in_its_digest_keeps_the_new_lineage(self):
        cutoff = ValueCutoff()
        values = {"U": np.arange(3.0), "model": object()}
        cutoff.lineage_for_outputs(values, dict.fromkeys(values, "L1"), compute_hash)
        assert cutoff.lineage_for_outputs(values, dict.fromkeys(values, "L2"), compute_hash) == dict.fromkeys(
            values, "L2"
        )

    def test_outputs_with_lineages_of_their_own_are_left_alone(self):
        cutoff = ValueCutoff()
        values = self._values()
        lineages = {"U": "L1", "V": "L1", "hist": "other"}
        cutoff.lineage_for_outputs(values, lineages, compute_hash)
        assert cutoff.lineage_for_outputs(values, {"U": "L2", "V": "L2", "hist": "other2"}, compute_hash) == {
            "U": "L2",
            "V": "L2",
            "hist": "other2",
        }

    def test_a_single_value_equal_to_an_output_does_not_take_its_lineage(self):
        """What several outputs share is never lent to one value."""
        cutoff = ValueCutoff()
        cutoff.lineage_for_outputs(self._values(), dict.fromkeys(("U", "V", "hist"), "L1"), compute_hash)
        assert cutoff.lineage_for(np.arange(3.0), "L9", compute_hash) == "L9"


def test_resetting_forgets_the_outputs_seen():
    cutoff = ValueCutoff()
    values = {"a": np.arange(2.0), "b": np.ones(2)}
    cutoff.lineage_for_outputs(values, dict.fromkeys(values, "L1"), compute_hash)
    cutoff.clear()
    assert cutoff == ValueCutoff()
