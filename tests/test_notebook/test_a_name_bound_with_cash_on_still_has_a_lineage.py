"""A control structure records a lineage for every name it read.

Its record of "what I read was worth this" came from
``TrackingState.variable_lineage`` alone, so a name with no entry there was
recorded as nothing at all. A name bound in the same cell as ``%cash_on`` is
exactly that name, forever: cash was not listening when that cell started.
The simulation reads that cell out of the .ipynb and has a lineage for it, so
the recorded entry was permanently short of a key the simulation carried,
``recorded[0] == input_hashes`` was false every time, and the loop's outcome
was never adopted.

The integration twin is
``tests/test_notebook_integration/test_a_loop_reading_a_cash_on_cell_name_still_restores.py``
and carries the measurements. Here the three pieces are pinned separately:
the fallback itself, the simulator writing the field it reads, and the
control processor consuming it.
"""

import ast

from tests._cell_driver import run_cash_cell


class TestTheFallback:
    """``_entry_lineages`` itself.

    Imported inside each test rather than at module scope on purpose: the
    helper did not exist before this change, and a module-level import would
    make the whole file fail to COLLECT on the parent commit, hiding whether
    the wiring tests below actually discriminate.
    """

    def test_a_name_the_runtime_never_tracked_takes_the_simulation_s_lineage(self):
        from cash.notebook.control_structures.processor import _entry_lineages

        entry = _entry_lineages(
            {"DATA", "ALIAS"}, {"ALIAS": "runtime-alias"}, {"DATA": "simulated-data", "ALIAS": "sim-alias"}
        )
        assert entry == {"ALIAS": "runtime-alias", "DATA": "simulated-data"}, (
            "DATA has no runtime lineage and the simulation has one; recording "
            "nothing for it is what made the comparison fail forever"
        )

    def test_the_runtime_s_own_lineage_wins_where_it_has_one(self):
        from cash.notebook.control_structures.processor import _entry_lineages

        entry = _entry_lineages({"ALIAS"}, {"ALIAS": "runtime"}, {"ALIAS": "simulated"})
        assert entry == {"ALIAS": "runtime"}

    def test_a_name_neither_side_knows_stays_out(self):
        from cash.notebook.control_structures.processor import _entry_lineages

        assert _entry_lineages({"MYSTERY"}, {}, {"OTHER": "x"}) == {}

    def test_no_simulation_at_all_behaves_as_before(self):
        """No notebook to read, or no upstream check ran: unchanged."""
        from cash.notebook.control_structures.processor import _entry_lineages

        assert _entry_lineages({"A", "B"}, {"A": "x"}, None) == {"A": "x"}
        assert _entry_lineages({"A", "B"}, {"A": "x"}, {}) == {"A": "x"}


class TestTheWiring:
    """The field is written by one component and read by another."""

    def test_the_simulator_leaves_its_view_on_the_tracking_state(self, cash_magics):
        """Without this the fallback above has nothing to fall back to."""
        run_cash_cell(cash_magics, "SECOND = FIRST * 2", cells=["FIRST = 21", "SECOND = FIRST * 2"])
        simulated = cash_magics.tracking_state.simulated_lineage
        assert simulated.get("FIRST"), "the simulation knows what FIRST is worth and did not pass it on: %r" % (
            simulated,
        )

    def test_a_loop_records_a_lineage_for_an_untracked_name_it_read(self, cash_magics, mock_shell, tmp_path):
        """The end the bug was at: what lands in ``control_outcomes``."""
        data = tmp_path / "rows.txt"
        data.write_text("aa\nbbb\n", encoding="utf-8")

        # DATA as the `%cash_on` cell leaves it: in the namespace and in the
        # notebook the simulation reads, but never executed through cash, so
        # the runtime has no lineage for it.
        bind = "DATA = " + repr(str(data))
        mock_shell.user_ns["DATA"] = str(data)
        code = "OUT = {}\nfor line in open(DATA).read().splitlines():\n    OUT[line] = len(line)"
        run_cash_cell(cash_magics, code, cells=[bind, code])

        import hashlib

        key = hashlib.sha256(ast.unparse(ast.parse(code).body[1]).encode("utf-8")).hexdigest()
        outcome = cash_magics.tracking_state.control_outcomes.get(key)
        assert outcome is not None, sorted(cash_magics.tracking_state.control_outcomes)
        simulated = cash_magics.tracking_state.simulated_lineage.get("DATA")
        assert simulated, "the simulation should know what DATA is worth"
        assert outcome[0].get("DATA") == simulated, (
            "the loop read DATA and recorded nothing about it, so its outcome "
            "can never match the simulation: %r" % (outcome[0],)
        )
