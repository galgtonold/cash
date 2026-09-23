"""
Test for uncommenting lines scenario - variables should reflect current session state
"""

import unittest

import numpy as np
import pandas as pd
import pytest

from tests._cell_driver import run_cash_cell


class TestUncommentLine(unittest.TestCase):
    """Test uncommenting a line and using the modified variable in next cell."""

    @pytest.fixture(autouse=True)
    def _notebook(self, cash_magics, mock_shell, clean_backend):
        self.magics, self.shell, self.backend = cash_magics, mock_shell, clean_backend

    def setUp(self):
        # Create initial DataFrame
        np.random.seed(42)
        self.shell.user_ns["df_clean"] = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "sales": np.random.rand(100) * 1000,
                "units": np.random.randint(1, 20, 100),
                "product": np.random.choice(["A", "B", "C"], 100),
                "region": np.random.choice(["North", "South", "East", "West"], 100),
            }
        )

        self.shell.user_ns["selected_region"] = "South"

    def test_uncomment_line_and_use_result(self):
        """
        Test uncommenting revenue line and then using revenue column.
        This simulates:
        1. Initially revenue line is commented
        2. User uncomments it and runs the cell
        3. Next cell tries to use revenue column - should work!
        """
        print("\n=== TEST: Uncomment Line and Use Result ===")

        # Step 1: Run cell WITH revenue line (uncommented)
        print("\n--- Step 1: Execute revenue calculation ---")
        cell1 = "df_clean['revenue'] = df_clean['sales'] * df_clean['units']"

        run_cash_cell(self.magics, cell1)

        # Verify revenue column exists
        self.assertIn("revenue", self.shell.user_ns["df_clean"].columns)
        print(f"Columns after step 1: {list(self.shell.user_ns['df_clean'].columns)}")

        # Step 2: Try to use revenue column in next cell
        print("\n--- Step 2: Use revenue column ---")
        cell2 = "summary = df_clean[df_clean['region'] == selected_region].groupby('product')['revenue'].sum()"

        run_cash_cell(self.magics, cell2)

        # Verify summary was created successfully
        self.assertIn("summary", self.shell.user_ns)
        self.assertIsNotNone(self.shell.user_ns["summary"])
        print(f"Summary created: {type(self.shell.user_ns['summary'])}")
        print("✓ Test passed: Revenue column accessible in next cell!")

    def test_comment_then_uncomment(self):
        """
        Test commenting out a line, then uncommenting it again.
        """
        print("\n=== TEST: Comment Then Uncomment ===")

        # Step 1: Run with revenue line
        print("\n--- Step 1: Execute with revenue ---")
        cell1 = "df_clean['revenue'] = df_clean['sales'] * df_clean['units']"
        run_cash_cell(self.magics, cell1)
        self.assertIn("revenue", self.shell.user_ns["df_clean"].columns)

        # Step 2: Comment out revenue line (reset df)
        print("\n--- Step 2: Comment out revenue (reset df) ---")
        # Simulate fresh df without revenue
        np.random.seed(42)
        self.shell.user_ns["df_clean"] = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=100),
                "sales": np.random.rand(100) * 1000,
                "units": np.random.randint(1, 20, 100),
                "product": np.random.choice(["A", "B", "C"], 100),
                "region": np.random.choice(["North", "South", "East", "West"], 100),
            }
        )
        self.assertNotIn("revenue", self.shell.user_ns["df_clean"].columns)

        # Step 3: Uncomment revenue line again
        print("\n--- Step 3: Uncomment revenue line ---")
        cell3 = "df_clean['revenue'] = df_clean['sales'] * df_clean['units']"
        run_cash_cell(self.magics, cell3)
        self.assertIn("revenue", self.shell.user_ns["df_clean"].columns)

        # Step 4: Use revenue in next cell
        print("\n--- Step 4: Use revenue column ---")
        cell4 = "summary = df_clean['revenue'].sum()"
        run_cash_cell(self.magics, cell4)

        self.assertIn("summary", self.shell.user_ns)
        self.assertIsInstance(self.shell.user_ns["summary"], (int, float, np.number))
        print(f"Summary value: {self.shell.user_ns['summary']}")
        print("✓ Test passed: Revenue accessible after uncommenting!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
