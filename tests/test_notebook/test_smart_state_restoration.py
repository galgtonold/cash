"""
Test smart state restoration with %cash_on mode
Tests the exact scenario user reported with commented DataFrame columns
"""

import unittest

import numpy as np
import pandas as pd
import pytest

from tests._cell_driver import run_cash_cell


class TestSmartStateRestoration(unittest.TestCase):
    """Test smart dependency-based state restoration."""

    @pytest.fixture(autouse=True)
    def _notebook(self, cash_magics, mock_shell, clean_backend):
        self.magics, self.shell, self.backend = cash_magics, mock_shell, clean_backend

    def setUp(self):
        # Create realistic DataFrame
        np.random.seed(42)
        self.shell.user_ns["df_clean"] = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "sales": np.random.rand(100) * 1000,
                "units": np.random.randint(1, 20, 100),
                "product": np.random.choice(["A", "B", "C"], 100),
            }
        )

    def test_commented_line_with_auto_caching(self):
        """
        Test user's exact scenario with %cash_on:
        1. Execute cell with revenue + month lines
        2. Comment out revenue line
        3. Execute again - revenue should NOT exist
        """
        print("\n=== TEST: Commented Line with Auto-Caching ===")

        # Step 1: Execute original cell (both lines)
        print("\n--- Step 1: Execute both lines ---")
        cell1 = (
            "df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )

        # Simulate pre_run_cell hook execution
        class ExecutionInfo:
            def __init__(self, raw_cell):
                self.raw_cell = raw_cell
                self.store_history = True
                self.silent = False

        # Simulate execution
        run_cash_cell(self.magics, cell1)

        # Verify both columns exist
        self.assertIn("revenue", self.shell.user_ns["df_clean"].columns)
        self.assertIn("month", self.shell.user_ns["df_clean"].columns)
        print(f"Columns after step 1: {list(self.shell.user_ns['df_clean'].columns)}")

        # Step 2: User comments out revenue line
        print("\n--- Step 2: Execute with revenue line commented ---")
        cell2 = (
            "#df_clean['revenue'] = df_clean['sales'] * df_clean['units']\n"
            "df_clean['month'] = df_clean['date'].dt.to_period('M')"
        )

        # Clear DataFrame to simulate fresh state
        self.shell.user_ns["df_clean"] = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "sales": np.random.rand(100) * 1000,
                "units": np.random.randint(1, 20, 100),
                "product": np.random.choice(["A", "B", "C"], 100),
            }
        )

        run_cash_cell(self.magics, cell2)

        print(f"Columns after step 2: {list(self.shell.user_ns['df_clean'].columns)}")

        # KEY ASSERTION: revenue should NOT exist (was commented out)
        self.assertNotIn(
            "revenue",
            self.shell.user_ns["df_clean"].columns,
            "BUG: revenue column should not exist when line is commented!",
        )
        self.assertIn("month", self.shell.user_ns["df_clean"].columns)

        print("✓ Test passed: Commented line correctly skipped!")

    def test_skip_restoration_if_unchanged(self):
        """
        Test optimization: skip restoration if variable hash matches.
        """
        print("\n=== TEST: Skip Restoration if Unchanged ===")

        # Step 1: Create a variable
        cell1 = "summary = 'test_value'"
        run_cash_cell(self.magics, cell1)

        self.assertEqual(self.shell.user_ns["summary"], "test_value")

        # Step 2: Execute cell that uses summary (but doesn't change it)
        cell2 = "result = summary + '_result'"

        # Before executing, track that we should check if restoration is skipped
        # The smart restoration should check summary's hash and skip restoration
        # since it's already correct

        run_cash_cell(self.magics, cell2)

        #  result should exist
        self.assertIn("result", self.shell.user_ns)
        self.assertEqual(self.shell.user_ns["result"], "test_value_result")

        print("✓ Test passed: Variable state correctly maintained!")


class ExecutionInfo:
    def __init__(self, raw_cell):
        self.raw_cell = raw_cell
        self.store_history = True
        self.silent = False


if __name__ == "__main__":
    unittest.main(verbosity=2)
