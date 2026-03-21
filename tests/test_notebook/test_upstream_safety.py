
import unittest
from unittest.mock import MagicMock, patch
import hashlib

from cash.notebook.upstream import UpstreamChecker
from cash.notebook._protocols import TrackingState

class TestUpstreamSafety(unittest.TestCase):
    def setUp(self):
        self.shell = MagicMock()
        self.shell.user_ns = {'_ih': []} # Mock history
        self.checker = UpstreamChecker(self.shell, debug=True)
        
        state = TrackingState()
        self.checker.set_tracking_state(state)
        self.codes = state.executed_cell_codes
        self.hashes = state.executed_cell_hashes
        self.lineage = state.variable_lineage

    @patch('cash.notebook.upstream.get_notebook_cells')
    def test_lineage_projection_safety(self, mock_get_cells):
        print("\n=== TEST: Lineage Projection Safety ===")
        
        # Scenario 1: Valid Unsaved Extension
        # Notebook: x = 1 (Hash A)
        # Memory: y = x + 1 (Hash B). y is NOT in notebook.
        # Check y.
        # Virtual y missing.
        # Wait, if y missing from notebook, Pass 2 usually ignores it unless it existed in virtual trace.
        # If y is NEW variable, it's ignored by Pass 2 default logic "if var_name in virtual_lineage".
        # So we test mismatch on existing variable 'x'.
        
        # Scenario 1 Revised: Unsaved Modification of x
        # Notebook: x = 1.
        # Memory: x = 1; x = x + 1. (Final x derived from x).
        # Virtual x: from 'x=1'.
        # Actual x: from 'x=x+1' using input 'x=1'.
        
        code_start = "x = 1"
        code_mod = "x = x + 1"
        
        hash_start = hashlib.sha256(code_start.encode()).hexdigest()
        lineage_start = hashlib.sha256((hash_start + ":").encode()).hexdigest()
        
        hash_mod = hashlib.sha256(code_mod.encode()).hexdigest()
        # Mod lineage depends on lineage_start
        lineage_mod_str = hash_mod + ":" + lineage_start
        lineage_mod = hashlib.sha256(lineage_mod_str.encode()).hexdigest()
        
        # Setup Notebook (Virtual)
        mock_get_cells.return_value = [code_start]
        
        # Setup Memory (Actual)
        self.lineage['x'] = lineage_mod
        self.codes['x'] = code_mod
        self.hashes['x'] = hash_mod # Not strictly needed for logic but good practice
        
        # Run Check - pass required_inputs to specify which variables matter
        # Current index after Cell 0
        reexecute, restored_info, restore_time = self.checker._simulate_and_find_changes(1, [code_start], required_inputs={'x'})
        print(f"Re-execute List (Valid Case): {reexecute}")
        
        # Expectation: Empty list (Valid Extension)
        self.assertEqual(reexecute, [], "Should accept valid unsaved extension")
        print("✓ Valid Extension Accepted")
        
        # Scenario 2: Stale Dependency
        # Notebook: x = 2 (Start Changed!)
        # Memory: x derived from x=1.
        
        code_new_start = "x = 2"
        hash_new_start = hashlib.sha256(code_new_start.encode()).hexdigest()
        hashlib.sha256((hash_new_start + ":").encode()).hexdigest()
        
        # Notebook has New Start
        mock_get_cells.return_value = [code_new_start]
        
        # Memory still has Old Mod (derived from Old Start)
        # Verify Mismatch
        
        reexecute_stale, restored_info, restore_time = self.checker._simulate_and_find_changes(1, [code_new_start], required_inputs={'x'})
        print(f"Re-execute List (Stale Case): {reexecute_stale}")
        
        # Expectation: Should detect mismatch and NOT be valid extension
        # Since 'x=x+1' projected on 'x=2' != 'x=x+1' projected on 'x=1'.
        # It should trigger re-execution of 'x=2' (the notebook cell).
        
        self.assertIn(code_new_start, reexecute_stale, "Should re-execute stale upstream")
        print("✓ Stale Dependency Rejected")
        
 

if __name__ == '__main__':
    unittest.main()
