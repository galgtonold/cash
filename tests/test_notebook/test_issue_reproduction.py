import unittest
from unittest.mock import MagicMock, patch

from cash.notebook._protocols import TrackingState
from cash.notebook.cache_status import CacheStatus
from cash.notebook.upstream import UpstreamChecker


# Stands in for CodeAnalyzer inside the _update_virtual_lineage side effect.
class MockCodeAnalyzer:
    @staticmethod
    def strip_magics(code):
        return code

    @staticmethod
    def analyze_code_block(code):
        # Return inputs, outputs based on simple heuristics or hardcoded
        if "stats =" in code:
            return set(), {"stats", "ticker_stats"}
        return set(), set()


class TestIssueReproduction(unittest.TestCase):
    def setUp(self):
        self.shell = MagicMock()
        # Configure backend
        self.shell.cash_instance.backend.get.return_value = ({"output_lineages": {}}, {})
        self.checker = UpstreamChecker(self.shell)
        self.checker.set_tracking_state(TrackingState())

    @patch("cash.notebook.upstream.checker.get_notebook_cells")
    def test_unused_broken_var_triggers_restore(self, mock_get_cells):
        print("\n=== TEST: Unused Broken Variable Triggering Restore ===")

        # Scenario:
        # Cell 1: stat_producer
        #   Produces 'stats' and 'ticker_stats'
        # Memory has Correct 'ticker_stats' but Broken 'stats'
        # Current execution requires ONLY 'ticker_stats'

        # 1. Notebook Content
        cell1_code = "stats, ticker_stats = 1, 1"
        cell2_code = "ticker_stats.keys()"
        mock_get_cells.return_value = [cell1_code, cell2_code]

        # 2. Setup Memory Lineage (Actual)
        with patch.object(self.checker.simulator.virtual_lineage, "_update_virtual_lineage") as mock_update:
            # Side effect for _update_virtual_lineage(stmt, lineage, modules)
            # Returns (outputs, lookup_time, files_stale, file_deps)
            def side_effect(stmt, lineage, modules, occurrence_index=0):
                input_set, output_set = MockCodeAnalyzer.analyze_code_block(stmt)
                # Update lineage dict in place
                for out in output_set:
                    lineage[out] = f"hash_virtual_{out}"
                return output_set, 0.0, False, {}  # 4 values: outputs, lookup_time, files_stale, file_deps

            mock_update.side_effect = side_effect

            # Set ACTUAL lineage in memory. Written into the shared tracking
            # state (not rebound on the checker) so the simulator sees it too.
            self.checker.variable_lineage.update(
                {
                    "stats": "hash_BROKEN_stats",  # Mismatch
                    "ticker_stats": "hash_virtual_ticker_stats",  # Match
                }
            )
            self.shell.user_ns = {"stats": 1, "ticker_stats": 1}

            # Set executed codes to match
            self.checker.executed_cell_codes.update({"stats": cell1_code, "ticker_stats": cell1_code})

            # 3. Checker call
            # Current cell is cell 2.
            # cell1 is upstream.

            # Required inputs for current cell (Cell 2) is only 'ticker_stats'
            required_inputs = {"ticker_stats"}

            # We also need try_virtual_restore to work so it reports success if attempted
            with patch.object(self.checker.simulator.virtual_lineage, "try_virtual_restore") as mock_restore:
                mock_restore.return_value = ({"stats", "ticker_stats"}, 0.1, 0.1)

                # Only the producer's simulation is stubbed; the cells parse
                # as they are, one statement binding both names.
                with patch(
                    "cash.notebook.upstream.virtual_lineage.is_control_structure", return_value=False
                ) as mock_is_cs:
                    # Execute Check for cell2
                    # Note: cell_code=cell2_code. REQUIRED INPUTS match what we setup.
                    all_metrics, _, _ = self.checker._check_notebook_based(cell2_code, required_inputs, None, None)

                    # The patches must reach the code under test: without the
                    # notebook cells the check returns before simulating
                    # anything, and the assertion below passes vacuously.
                    mock_get_cells.assert_called()
                    mock_update.assert_called()
                    mock_is_cs.assert_called()

                    restored_codes = [r["code"] for r in all_metrics if r.get("status") == CacheStatus.RESTORED]
                    print(f"Restored Codes: {restored_codes}")

                    # The fix ensures that unused broken variables (like 'stats')
                    # do NOT trigger restoration. Only 'ticker_stats' is a required input,
                    # and it matches the virtual lineage, so no restoration should occur.
                    if not restored_codes:
                        print("✓ FIX VERIFIED: Unused 'stats' did NOT trigger unnecessary restoration")
                    else:
                        print(f"X BUG STILL PRESENT: Unnecessary restoration occurred: {restored_codes}")

                    self.assertEqual(
                        restored_codes,
                        [],
                        "Unused broken variable 'stats' should NOT trigger restoration when only 'ticker_stats' is required",
                    )


if __name__ == "__main__":
    unittest.main()
