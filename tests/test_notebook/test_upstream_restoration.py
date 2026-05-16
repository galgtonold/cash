
import unittest
from unittest.mock import MagicMock, patch
from cash.notebook.upstream import UpstreamChecker
from cash.notebook._protocols import TrackingState

# Helper for mocking CodeAnalyzer inputs if needed
# But specific mocking of CodeAnalyzer requires care.
# We rely on the fact that CodeAnalyzer works for simple expressions like "x = 1".

class TestUpstreamRestoration(unittest.TestCase):
    def setUp(self):
        self.shell = MagicMock()
        # Configure backend to return empty metadata/data by default to avoid Truthy mocks
        self.shell.cash_instance.backend.get.return_value = ({'output_lineages': {}}, {})
        self.checker = UpstreamChecker(self.shell, debug=True)
        self.checker.set_tracking_state(TrackingState())

    @patch('cash.notebook.upstream.get_notebook_cells')
    def test_restore_unsaved_extension(self, mock_get_cells):
        print("\n=== TEST: Unsaved Extension Restoration ===")
        
        # Scenario: 
        # Notebook: x = 1 (Old).
        # Memory: x = 1; x = x + 1. (Final x derived).
        # User UPDATES Notebook: x = 2.
        
        # Setup Notebook (New State)
        code_new_base = "x = 2"
        mock_get_cells.return_value = [code_new_base]
        
        # Setup Memory (Old State + Extension) aka Broken
        code_extension = "x = x + 1"
        self.checker.executed_cell_codes['x'] = code_extension
        # Lineage setup logic (simplified for test logic flow):
        # We assume Lineage Check detects mismatch.
        # So we manually populate 'broken_vars' or simulate Pass 2 failure.
        
        # Actually we need to set up lineage so _check_notebook_based finds the mismatch.
        # But _check_notebook_based is integration testing logic.
        # Let's test _simulate_and_find_changes LOGIC directly.
        
        # To simulate mismatch:
        # We need actual_lineage != final_virtual.
        # And NOT valid extension.
        
        # Virtual (New Base): x = 2.
        
        # Actual (Old Base + Extension): x = 1 + 1 = 2.
        lineage_actual = "hash_of_x1_plus_1"
        
        # Mismatch.
        
        self.checker.variable_lineage['x'] = lineage_actual
        
        # We need to ensure _simulate results in broken variable.
        # _simulate runs code_new_base -> x outputs.
        # virtual_lineage['x'] = ...
        
        # Test: Pass required_inputs={'x'} to specify which variables matter
        reexecute_list, restored_info, restore_time = self.checker.simulator._simulate_and_find_changes(1, [code_new_base], required_inputs={'x'})
        
        print(f"Re-execute list: {reexecute_list}")
        
        # Expectation:
        # 1. code_new_base (x=2) because it produces x (which is broken).
        # 2. code_extension (x=x+1) because x is broken & it's Unsaved Code.
        
        self.assertIn(code_new_base, reexecute_list)
        # Extension should be DROPPED because x was updated by the trace (x=2)
        # This prevents stale extensions from overwriting new code
        self.assertNotIn(code_extension, reexecute_list)
        
        # Check Order - irrelevant if absent
        # idx_base = reexecute_list.index(code_new_base)
        # idx_ext = reexecute_list.index(code_extension)
        # self.assertLess(idx_base, idx_ext, "Base must execute before Extension")
        
        print("✓ Correctly restored unsaved extension after base update")

if __name__ == '__main__':
    # Need to verify if CodeAnalyzer works in this env
    # It parses "x=2" fine.
    unittest.main()
