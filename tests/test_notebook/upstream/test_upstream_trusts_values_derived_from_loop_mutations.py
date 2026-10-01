"""The upstream check keeps a value built from a loop-mutated variable in memory
instead of restoring a stale copy from the cache.

A list filled by a loop and a DataFrame built from it are trusted as they are,
transitively; the safety guard still refuses a restore that would hand back an
empty value where memory holds a filled one.
"""

from cash.notebook.upstream._types import TraceEntry
from cash.notebook.upstream.virtual_lineage import loop_derived_vars
from tests._cell_driver import run_cash_cell


class TestTransitiveLoopMutation:
    """Test that variables derived from loop-mutated inputs are trusted in memory."""

    def test_transitive_propagation_simple(self, cash_magics, mock_shell):
        """Variable derived from loop-mutated var should be trusted, not restored from cache.

        Scenario:
        - Cell 1: events = []; for item in data: events.append(item)
        - Cell 2: df = pd.DataFrame(events)  -> depends on loop-mutated 'events'
        - Cell 3: top = df.head()  -> depends on 'df' which depends on 'events'

        When running cell 3, upstream should NOT restore 'events' as [] from cache.
        """
        # Simulate loop-mutated variable
        mock_shell.user_ns["events"] = [1, 2, 3, 4, 5]
        run_cash_cell(cash_magics, "events = []")

        # Simulate the loop populating events
        mock_shell.user_ns["events"] = [1, 2, 3, 4, 5]  # As if loop populated it

        # Now create derived variable
        run_cash_cell(cash_magics, "total = len(events)")
        assert mock_shell.user_ns["total"] == 5

    def test_safety_guard_blocks_empty_restore(self):
        """A restore should refuse to overwrite non-empty with empty cached value."""
        from unittest.mock import MagicMock

        from cash.notebook.upstream import UpstreamChecker

        shell = MagicMock()
        # In-memory: non-empty list with 1000 items
        shell.user_ns = {"my_list": list(range(1000))}

        cash_instance = MagicMock()
        # Cache returns empty list
        cash_instance.backend.get.return_value = (
            {"output_lineages": {"my_list": "hash123"}, "execution_time": 1.0},
            {"variables": {"my_list": []}},  # EMPTY cached value
        )

        checker = UpstreamChecker(shell, cash_instance)

        restored = checker.simulator.restore_statement(
            "my_list = compute_data()",
            {"my_list"},
            {"compute_data"},
            {},
        )

        # The empty cached value should NOT overwrite the non-empty in-memory value
        assert "my_list" not in restored or len(shell.user_ns["my_list"]) == 1000, (
            "Safety guard should block restoring empty value over non-empty in-memory value"
        )

    def test_safety_guard_allows_valid_restore(self):
        """A restore should allow restoring a non-empty cached value."""
        from unittest.mock import MagicMock

        from cash.notebook.upstream import UpstreamChecker

        shell = MagicMock()
        shell.user_ns = {"x": 42}  # scalar — no len()

        cash_instance = MagicMock()
        cash_instance.backend.get.return_value = (
            {"output_lineages": {"x": "hash456"}, "execution_time": 0.5},
            {"variables": {"x": 99}},
        )

        checker = UpstreamChecker(shell, cash_instance)

        restored = checker.simulator.restore_statement(
            "x = compute()",
            {"x"},
            {"compute"},
            {},
        )

        # Scalar values (no len) should be restored normally
        assert "x" in restored
        assert shell.user_ns["x"] == 99

    def test_transitive_loop_vars_computed_from_simulation_trace(self):
        """The simulation trace should propagate loop-mutation flag transitively."""
        # Build a mock simulation trace:
        # Statement 1: events = []  (outputs: {events})
        # Statement 2: events.append(x)  → events is loop-mutated
        # Statement 3: df = DataFrame(events)  (inputs: {events}, outputs: {df})
        # Statement 4: top = df.head()  (inputs: {df}, outputs: {top})

        vars_mutated_by_loops = {"events"}
        simulation_trace = [
            TraceEntry("events = []", {"events"}, set(), {}, {}, False),
            TraceEntry("df = pd.DataFrame(events, columns=['a'])", {"df"}, {"events", "pd"}, {}, {}, False),
            TraceEntry("top = df.head()", {"top"}, {"df"}, {}, {}, False),
        ]

        vars_derived = loop_derived_vars(vars_mutated_by_loops, simulation_trace)

        assert "events" in vars_derived, "Directly mutated var should be in derived set"
        assert "df" in vars_derived, "df depends on events → should be derived"
        assert "top" in vars_derived, "top depends on df → should be transitively derived"
