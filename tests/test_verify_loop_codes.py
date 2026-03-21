"""
Test that loop body statements are passed to the statement processor per-iteration.

For loops are decomposed per-iteration: each body statement is processed
separately with an iteration context marker.  This test verifies that
body statements are individually passed through and that the iteration
context marker is present.
"""

import unittest
from cash.notebook.control_structures import ControlStructureProcessor
import ast
from unittest.mock import MagicMock
from cash.notebook.cache_status import CacheStatus


class TestLoopCodeCapture(unittest.TestCase):
    def test_loop_passes_body_statements_per_iteration(self):
        """Loop body statements should be passed individually per iteration."""
        shell = MagicMock()
        shell.user_ns = {'range': range}

        sp = MagicMock()
        sp.process = MagicMock(return_value={
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.01,
            'stdout': '',
            'stderr': '',
            'outputs': []
        })
        sp.variable_lineage = {}
        sp.vars_with_mutation_lineage = set()
        sp.compute_hash = MagicMock(return_value='fakehash')

        csp = ControlStructureProcessor(shell, sp)

        loop_code = """
for i in range(3):
    x = i * 10
    y = {'a': x}
"""
        node = ast.parse(loop_code).body[0]
        csp.process(node)

        # 2 body statements × 3 iterations = 6 calls
        self.assertEqual(sp.process.call_count, 6)

        # Each call should have iteration context and body statement code
        for call in sp.process.call_args_list:
            passed_code = call[0][0]
            self.assertIn('# __iteration_context__:', passed_code)

        # Check that both body statements are covered
        all_codes = [call[0][0] for call in sp.process.call_args_list]
        has_x = any('x = i * 10' in c for c in all_codes)
        has_y = any("y = {'a': x}" in c for c in all_codes)
        self.assertTrue(has_x, "Body statement 'x = i * 10' not found")
        self.assertTrue(has_y, "Body statement \"y = {'a': x}\" not found")


if __name__ == '__main__':
    unittest.main()
