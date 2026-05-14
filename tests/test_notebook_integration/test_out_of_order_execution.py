"""
Tests for out-of-order cell execution in notebooks.

When users execute a downstream cell first (e.g., display cell), cash restores
variables from disk cache. Then when the user executes an upstream cell, the
upstream checker should recognise that the in-memory variable is correct and
NOT force expensive re-execution.

With _sync_simulation_cache_lineages(), the simulation cache is kept in sync
with actual variable lineages after each execution. This means that when
cell 2 runs after cell 3 was already executed, the virtual lineages already
match the actual lineages — no mismatch is detected, so no downstream
advancement check is needed and no broken vars are found.
"""
import pytest

pytestmark = pytest.mark.upstream


class TestOutOfOrderDownstreamFirst:
    """Executing a downstream cell before its upstream should use cached data."""

    def test_downstream_display_then_upstream_computation(self, nb_runner):
        """
        Scenario: 3 cells
          Cell 1: create DataFrame df
          Cell 2: heavy transformation of df (df['new'] = ...; df = df.sort_values(...))
          Cell 3: display df

        Execute order: 1 → 3 → 2.
        On the second execution of cell 2, the upstream checker should see that
        df is "ahead" (downstream advancement), reset its lineage to the virtual
        state, and the cell's own statements should get cache hits.
        """
        import textwrap

        cell_1 = textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            np.random.seed(42)
            n = 200
            df = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=n, freq='D'),
                'price': np.cumsum(np.random.randn(n)) + 100,
                'volume': np.random.randint(100, 10000, n),
            })
            print(f"Created df: {df.shape}")""")

        cell_2 = textwrap.dedent("""\
            df['rolling_mean'] = df['price'].rolling(window=10).mean()
            df = df.sort_values('date')
            print(f"Computed, shape={df.shape}")""")

        cell_3 = textwrap.dedent("""\
            print(f"Display: {df.shape}, cols={list(df.columns)}")""")

        nb_runner.create_notebook([cell_1, cell_2, cell_3])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # --- First pass: Run all cells in order to populate cache ---
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Created df" in out1

        out2_first = nb_runner.get_output(2)
        assert "Computed" in out2_first
        assert "shape=" in out2_first

        out3_first = nb_runner.get_output(3)
        assert "rolling_mean" in out3_first

        # --- Reset cash state to simulate a fresh kernel session ---
        nb_runner.reset_cash_state()

        # --- Out-of-order execution: run cell 3 first, then cell 2 ---
        # Cell 3 should trigger upstream restoration of df from cache
        nb_runner.run_cell(3)
        out3_rerun = nb_runner.get_output(3)
        assert "rolling_mean" in out3_rerun, (
            f"Cell 3 should have restored df with rolling_mean column, got: {out3_rerun}")

        # Cell 2 should NOT re-compute; df lineage should be reset to virtual state
        # and the cell's statements should get cache hits.
        nb_runner.run_cell(2)
        out2_rerun = nb_runner.get_raw_output(2)

        # Verify no upstream re-execution — this is the key correctness check
        assert "Auto-executing upstream" not in out2_rerun, (
            f"Cell 2 should NOT auto-execute upstream when df is already correct. "
            f"Got: {out2_rerun}")

    def test_downstream_first_with_file_dependency(self, nb_runner, tmp_path):
        """
        Like the above but cell 1 reads from a CSV file, adding a file dependency.
        This mirrors the financial_analysis_demo.ipynb scenario exactly.
        """
        import pandas as pd

        # Create test CSV
        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace('\\', '/')
        pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'price': range(100),
            'volume': range(100, 200),
        }).to_csv(csv_path, index=False)

        cell_1 = f"""\
import pandas as pd
df = pd.read_csv('{csv_str}')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded df: {{df.shape}}")"""

        cell_2 = """\
df = df.sort_values('date')
df['rolling'] = df['price'].rolling(5).mean()
print(f"Processed df: {df.shape}")"""

        cell_3 = """\
print(f"Display: {df.shape}, cols={list(df.columns)}")"""

        nb_runner.create_notebook([cell_1, cell_2, cell_3])
        nb_runner.start_kernel()

        # First pass: populate cache
        nb_runner.run_all()
        out3 = nb_runner.get_output(3)
        assert "rolling" in out3

        # Simulate fresh session
        nb_runner.reset_cash_state()

        # Out-of-order: cell 3 first, then cell 2
        nb_runner.run_cell(3)
        out3_rerun = nb_runner.get_output(3)
        assert "rolling" in out3_rerun

        nb_runner.run_cell(2)
        out2_rerun = nb_runner.get_raw_output(2)
        assert "Auto-executing upstream" not in out2_rerun, (
            f"Cell 2 should NOT auto-execute upstream. Got: {out2_rerun}")

    def test_downstream_first_repeated_assignment(self, nb_runner):
        """
        Test scenario where df is assigned in multiple cells (common pattern):
          Cell 1: df = pd.DataFrame(...)  (creation)
          Cell 2: df = df.sort_values(...)  (self-assignment)
          Cell 3: df['new_col'] = ...  (mutation then df as last expr)
          Cell 4: df  (display)

        Execute: 1 → 2 → 3 → 4 (populate cache), then reset, then 4 → 3.
        Cell 3 should not trigger upstream re-execution.
        """
        import textwrap

        cell_1 = textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            np.random.seed(42)
            df = pd.DataFrame({
                'a': np.random.randn(50),
                'b': np.random.randint(1, 10, 50),
            })
            print(f"Created: {df.shape}")""")

        cell_2 = textwrap.dedent("""\
            df = df.sort_values('b')
            print(f"Sorted: {df.shape}")""")

        cell_3 = textwrap.dedent("""\
            df['c'] = df['a'] * df['b']
            print(f"Added column: {df.shape}")
            df""")

        cell_4 = textwrap.dedent("""\
            print(f"Display: {df.shape}, cols={list(df.columns)}")""")

        nb_runner.create_notebook([cell_1, cell_2, cell_3, cell_4])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()
        out4 = nb_runner.get_output(4)
        assert "'c'" in out4

        # Simulate fresh session
        nb_runner.reset_cash_state()

        # Out-of-order: cell 4 first, then cell 3
        nb_runner.run_cell(4)
        out4_rerun = nb_runner.get_output(4)
        assert "'c'" in out4_rerun, f"df should be restored with 'c' column: {out4_rerun}"

        nb_runner.run_cell(3)
        out3_rerun = nb_runner.get_raw_output(3)
        assert "Auto-executing upstream" not in out3_rerun, (
            f"Cell 3 should NOT auto-execute upstream. Got: {out3_rerun}")


class TestOutOfOrderEdgeCases:
    """Edge cases for out-of-order execution."""

    def test_middle_cell_first(self, nb_runner):
        """
        Execute the middle cell first. It should restore its inputs from
        cache and not mark anything as broken.
        """
        nb_runner.create_notebook([
            "x = 42\nprint(f'x={x}')",
            "y = x * 2\nprint(f'y={y}')",
            "z = y + 1\nprint(f'z={z}')",
        ])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()

        # Reset and run cell 2 first
        nb_runner.reset_cash_state()
        nb_runner.run_cell(2)
        out2 = nb_runner.get_output(2)
        assert "y=84" in out2

        # Now run cell 3 — should use cached y
        nb_runner.run_cell(3)
        out3 = nb_runner.get_output(3)
        assert "z=85" in out3

    def test_last_cell_first_then_second_to_last(self, nb_runner):
        """
        Run the last cell, then the second-to-last. Second-to-last should
        not re-execute upstream when the variable is ahead.
        """
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 5\nprint(f'b={b}')",
            "c = b * 2\nprint(f'c={c}')",
        ])
        nb_runner.start_kernel()

        # Populate cache
        nb_runner.run_all()

        # Reset and run cell 3 first
        nb_runner.reset_cash_state()
        nb_runner.run_cell(3)
        out3 = nb_runner.get_output(3)
        assert "c=30" in out3

        # Now run cell 2 — b was already restored, should use cache
        nb_runner.run_cell(2)
        out2 = nb_runner.get_raw_output(2)
        assert "Auto-executing upstream" not in out2
        assert "b=15" in nb_runner.get_output(2)
