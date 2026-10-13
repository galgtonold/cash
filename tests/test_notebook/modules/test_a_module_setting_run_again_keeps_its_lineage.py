"""A module setting run again leaves the module's lineage where one run left it.

``pd.set_option('display.width', 120)`` or ``plt.style.use('ggplot')`` is a
change to its module, so the module's lineage moves. Each run built on the
lineage the run before left, so a session that ran the setup cell three times
keyed every statement reading the module apart from a Run All after a
restart, which ran it once; headless, all of them ran again.

A setting writes the values it is handed over what was there, so the same
setting run again, or the same run of settings again, changes nothing, and
the lineage stays. Another value, or the settings in another order, is still
another lineage.
"""

from __future__ import annotations

import ast
import warnings

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IGNORE = "warnings.filterwarnings('ignore', category=DeprecationWarning)"
ERROR = "warnings.filterwarnings('error', category=DeprecationWarning)"
ONCE = "warnings.simplefilter('once', UserWarning)"


@pytest.fixture(autouse=True)
def _keep_warning_filters():
    saved = warnings.filters[:]
    yield
    warnings.filters[:] = saved
    warnings._filters_mutated()


def _lineage(magics, name="warnings"):
    return magics.tracking_state.variable_lineage.get(name)


def _run(magics, *statements):
    for statement in statements:
        run_cash_cell(magics, statement)
    return _lineage(magics)


class TestTheModuleLineage:
    def test_the_same_setting_run_again_keeps_it(self, cash_magics):
        run_cash_cell(cash_magics, "import warnings")
        imported = _lineage(cash_magics)
        once = _run(cash_magics, IGNORE)
        assert once != imported
        assert _run(cash_magics, IGNORE, IGNORE) == once

    def test_a_setup_cell_run_again_keeps_it(self, cash_magics):
        run_cash_cell(cash_magics, "import warnings")
        once = _run(cash_magics, f"{IGNORE}\n{ONCE}")
        assert _run(cash_magics, f"{IGNORE}\n{ONCE}", f"{IGNORE}\n{ONCE}") == once

    def test_another_value_moves_it(self, cash_magics):
        run_cash_cell(cash_magics, "import warnings")
        ignore = _run(cash_magics, IGNORE)
        assert _run(cash_magics, ERROR) != ignore

    def test_settings_in_another_order_are_another_lineage(self, cash_magics):
        run_cash_cell(cash_magics, "import warnings")
        ignore_then_error = _run(cash_magics, IGNORE, ERROR)
        run_cash_cell(cash_magics, "import warnings")
        error_then_ignore = _run(cash_magics, ERROR, IGNORE)
        assert ignore_then_error != error_then_ignore
        # A setting run again after another one is not taken back to where
        # it alone left the module: the other one may have changed the same
        # option in between.
        run_cash_cell(cash_magics, "import warnings")
        ignore = _run(cash_magics, IGNORE)
        assert _run(cash_magics, ERROR, IGNORE) != ignore

    def test_a_setting_that_reads_the_module_moves_it_every_run(self, cash_magics):
        run_cash_cell(cash_magics, "import warnings")
        first = _run(cash_magics, "warnings.simplefilter(warnings.defaultaction)")
        assert _run(cash_magics, "warnings.simplefilter(warnings.defaultaction)") != first

    def test_a_reader_after_the_setting_ran_three_times_is_cached(self, cash_magics):
        cells = [
            "import time\nimport warnings",
            IGNORE,
            f"n = len(warnings.filters) + int(time.sleep({ABOVE_PERSISTENCE_FLOOR_S}) or 0)",
        ]
        run_cash_cell(cash_magics, cells[0], cells=cells)
        for _ in range(3):
            run_cash_cell(cash_magics, cells[1], cells=cells)
        run_cash_cell(cash_magics, cells[2], cells=cells)
        run_cash_cell(cash_magics, cells[2], cells=cells)
        statuses = [
            (m.get("code"), str(m.get("status")))
            for m in cash_magics.cash_status("dict")["last_cell"]["statements"]
            if m.get("code")
        ]
        assert [status for code, status in statuses if code.startswith("n =")] == ["RESTORED"], statuses

    def test_the_simulation_gives_the_module_the_lineage_the_runtime_did(self, cash_magics):
        """With the notebook found, the upstream check rebuilds the module's
        lineage from the cells, each run once: it must reach the lineage the
        runtime recorded however often the cell ran."""
        run_cash_cell(cash_magics, "import warnings")
        runtime = _run(cash_magics, IGNORE, IGNORE, IGNORE)

        simulator = cash_magics._upstream_checker.simulator
        imported = simulator.simulate_cell("import warnings")
        simulated = simulator.simulate_cell(IGNORE, imported.virtual_lineage, imported.virtual_modules)
        assert simulated.virtual_lineage["warnings"] == runtime


class TestTheRun:
    """`setting_lineage`, the run of settings a module's lineage stands for."""

    def test_a_repeat_at_the_end_is_kept_once(self):
        from cash.notebook.lineage_formula import setting_lineage

        one = setting_lineage("base", "A")
        assert setting_lineage(one, "A") == one
        two = setting_lineage(one, "B")
        assert setting_lineage(setting_lineage(two, "A"), "B") == two

    def test_order_and_values_count(self):
        from cash.notebook.lineage_formula import setting_lineage

        ab = setting_lineage(setting_lineage("base", "A"), "B")
        ba = setting_lineage(setting_lineage("base", "B"), "A")
        assert ab != ba
        assert setting_lineage(ab, "A") != setting_lineage("base", "A")
        assert setting_lineage("base", "A") != setting_lineage("other", "A")


@pytest.mark.parametrize(
    ("code", "module"),
    [
        ("pd.set_option('display.width', 120)", "pd"),
        ("plt.style.use('ggplot')", "plt"),
        ("plt.rcParams.update({'font.size': 9})", "plt"),
        ("np.seterr(all='raise')", "np"),
        ("logging.basicConfig(level=logging_level)", "logging"),
        ("pd.set_option('display.width', pd.get_option('display.width') + 1)", None),
        ("mylib.items.append(1)", None),
        ("mylib.counts.update(words)", None),
        ("get_lib().set_option('a', 1)", None),
        ("old = pd.set_option('display.width', 120)", None),
        ("pd.DataFrame({'a': [1]})", None),
    ],
)
def test_which_statements_are_a_setting_run_again_alike(code, module):
    from cash.analysis.mutations import repeatable_setting_receiver

    assert repeatable_setting_receiver(ast.parse(code)) == module
