"""Running cells out of order, then top to bottom."""

import time

import pytest

pytestmark = [pytest.mark.upstream]


# Out-of-order execution & re-execution patterns.
#
# Tests that exercise:
# - Running cells in non-sequential order
# - Re-running earlier cells after later ones
# - Skipping cells then coming back to them
# - Running cell 3 before cell 2 (non-linear workflows)
# - Re-running cells without changes (idempotency)
class TestOutOfOrderExecution:
    """Run cells in non-sequential order and check consistency."""

    @pytest.mark.stress
    def test_run_last_before_middle(self, nb_runner):
        """Run cell 3 before cell 2 — cell 3 should trigger upstream."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        # Run cell 1, skip cell 2, run cell 3
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "z = 30" in nb_runner.get_output(3)

    @pytest.mark.stress
    def test_run_cell_then_rerun_earlier(self, nb_runner):
        """Run all, then re-run cell 2 again — should skip or use cache."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5\nprint(f'y = {y}')",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)
        assert "z = 30" in nb_runner.get_output(3)

        # Re-run cell 2 — should produce same result
        nb_runner.run_cell(2)
        assert "y = 15" in nb_runner.get_output(2)

    @pytest.mark.stress
    def test_run_cells_reverse_order(self, nb_runner):
        """Run cells 3, 2, 1 — upstream checker should handle correctly."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 10\nprint(f'b = {b}')",
                "c = b + 100\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        # Run in reverse
        nb_runner.run_cell(3)
        assert "c = 111" in nb_runner.get_output(3)

        nb_runner.run_cell(2)
        assert "b = 11" in nb_runner.get_output(2)

        nb_runner.run_cell(1)

    @pytest.mark.stress
    def test_skip_middle_then_run_it(self, nb_runner):
        """Skip cell 2 initially, run cell 3, then run cell 2."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 3\nprint(f'y = {y}')",
                "z = x + 10\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "z = 15" in nb_runner.get_output(3)

        # Now run the skipped cell 2
        nb_runner.run_cell(2)
        assert "y = 15" in nb_runner.get_output(2)

    # Out-of-order execution interaction tests.
    #
    # Tests executing cells in non-sequential order and verifying
    # that cash correctly handles the dependency tracking.
    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_skip_middle_cell(self, nb_runner):
        """Run cells 1 and 3, skipping cell 2."""
        nb_runner.create_notebook(
            [
                "x = 5  # step 1",
                "y = x * 2  # step 2 (will be skipped initially)",
                "z = x + 10\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "z = 15" in nb_runner.get_output(3)

        # Now run all to fill in middle
        nb_runner.run_all()
        assert "z = 15" in nb_runner.get_output(3)

    @pytest.mark.stress
    @pytest.mark.timeout(90)
    def test_run_all_then_single_cell(self, nb_runner):
        """Run all, then re-run a single middle cell."""
        nb_runner.create_notebook(
            [
                "a = 1  # value a",
                "b = a + 1",
                "c = b + 1\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        # Re-run just cell 2 — should be idempotent
        nb_runner.run_cell(2)
        nb_runner.run_cell(3)
        assert "c = 3" in nb_runner.get_output(3)


@pytest.mark.stress
class TestRerunWithoutChanges:
    """Re-running cells without changes should be idempotent."""

    def test_idempotent_rerun_single_cell(self, nb_runner):
        """Run cell 3 multiple times — each should give same result."""
        nb_runner.create_notebook(
            [
                "val = 7",
                "doubled = val * 2",
                "print(f'doubled = {doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled = 14" in nb_runner.get_output(3)

        for _ in range(3):
            nb_runner.run_cell(3)
            assert "doubled = 14" in nb_runner.get_output(3)

    def test_rerun_all_three_times(self, nb_runner):
        """Run all three times — regression check for state accumulation bugs."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5",
                "c = b * 2\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        for _ in range(3):
            nb_runner.run_all()
            assert "c = 30" in nb_runner.get_output(3)


@pytest.mark.stress
class TestEditBetweenOutOfOrderRuns:
    """Combine edits with out-of-order execution."""

    def test_edit_cell1_run_cell3_then_cell2(self, nb_runner):
        """Edit cell 1, run cell 3 (should auto-propagate), then run cell 2."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x + 1\nprint(f'y = {y}')",
                "z = x * 10\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)
        assert "z = 50" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 20")
        nb_runner.run_cell(3)
        assert "z = 200" in nb_runner.get_output(3)

        nb_runner.run_cell(2)
        assert "y = 21" in nb_runner.get_output(2)

    def test_edit_after_partial_run(self, nb_runner):
        """Run cells 1-2, edit cell 1, then run cell 3."""
        nb_runner.create_notebook(
            [
                "x = 3",
                "y = x ** 2",
                "z = y + x\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2])

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_cell(3)
        assert "z = 110" in nb_runner.get_output(3)

    def test_interleaved_edits_and_runs(self, nb_runner):
        """Edit cell 1, run cell 2, edit cell 1 again, run cell 3."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 10\nprint(f'y = {y}')",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)
        assert "z = 22" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(2)
        assert "y = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cell(3)
        assert "z = 120" in nb_runner.get_output(3)


@pytest.mark.stress
class TestSelfAssignmentInteractions:
    """Test cache coherence with self-assignment patterns (df = df.something())."""

    def test_self_assignment_chain(self, nb_runner):
        """Self-assignment across cells with upstream edit."""
        nb_runner.create_notebook(
            [
                "data = [3, 1, 2]",
                "data = sorted(data)",
                "result = data[0]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(3)

        # Change initial data
        nb_runner.set_cell_source(1, "data = [30, 10, 20]")
        nb_runner.run_cell(3)
        assert "result = 10" in nb_runner.get_output(3)

    def test_self_assignment_rerun(self, nb_runner):
        """Self-assignment re-run should not accumulate."""
        nb_runner.create_notebook(
            [
                "items = [1, 2, 3]",
                "items = [x * 2 for x in items]",
                "print(f'items = {items}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [2, 4, 6]" in nb_runner.get_output(3)

        # Re-run all — should NOT produce [4, 8, 12]
        nb_runner.run_all()
        assert "items = [2, 4, 6]" in nb_runner.get_output(3)


@pytest.mark.stress
class TestDiamondDependency:
    """Multiple cells depend on the same upstream cell."""

    def test_diamond_edit_root(self, nb_runner):
        """
        Cell 1: x = 10
        Cell 2: y = x + 1
        Cell 3: z = x * 2
        Cell 4: w = y + z  (diamond dependency on x through y and z)
        """
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1",
                "z = x * 2",
                "w = y + z\nprint(f'w = {w}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 31" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(4)
        assert "w = 16" in nb_runner.get_output(4)

    def test_diamond_edit_one_branch(self, nb_runner):
        """Edit only one branch of the diamond."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1",
                "z = x * 2",
                "w = y + z\nprint(f'w = {w}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 31" in nb_runner.get_output(4)

        # Change only cell 2 formula (one branch)
        # y = 10 + 100 = 110, z = 10 * 2 = 20, w = 110 + 20 = 130
        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cell(4)
        assert "w = 130" in nb_runner.get_output(4)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSelectiveCellExecution:
    """Running specific subsets of cells."""

    def test_run_cells_subset(self, nb_runner):
        """Run only specific cells."""
        nb_runner.create_notebook(
            [
                "x = 10  # cell 1",
                "y = 20  # cell 2 (independent)",
                "z = x + y\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2, 3])
        assert "z = 30" in nb_runner.get_output(3)

    def test_edit_and_run_single_cell(self, nb_runner):
        """Edit one cell and run only that cell and its dependents."""
        nb_runner.create_notebook(
            [
                "n = 5  # parameter",
                "result = n ** 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

        # Edit and run just the edited cell + dependent
        nb_runner.set_cell_source(1, "n = 10  # parameter bigger")
        nb_runner.run_cells([1, 2])
        assert "result = 100" in nb_runner.get_output(2)

    def test_run_last_cell_only_after_full_run(self, nb_runner):
        """After full run, re-running last cell should use cached deps."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]  # data list",
                "total = sum(data)",
                "print(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(3)

        # Re-run only the last cell
        nb_runner.run_cell(3)
        assert "total = 15" in nb_runner.get_output(3)


# Tests for out-of-order cell execution in notebooks.
#
# When users execute a downstream cell first (e.g., display cell), cash restores
# variables from disk cache. Then when the user executes an upstream cell, the
# upstream checker should recognise that the in-memory variable is correct and
# NOT force expensive re-execution.
#
# With _sync_simulation_cache_lineages(), the simulation cache is kept in sync
# with actual variable lineages after each execution. This means that when
# cell 2 runs after cell 3 was already executed, the virtual lineages already
# match the actual lineages — no mismatch is detected, so no downstream
# advancement check is needed and no broken vars are found.
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
            f"Cell 3 should have restored df with rolling_mean column, got: {out3_rerun}"
        )

        # Cell 2 should NOT re-compute; df lineage should be reset to virtual state
        # and the cell's statements should get cache hits.
        nb_runner.run_cell(2)
        out2_rerun = nb_runner.get_raw_output(2)

        # Verify no upstream re-execution — this is the key correctness check
        assert "Auto-executing upstream" not in out2_rerun, (
            f"Cell 2 should NOT auto-execute upstream when df is already correct. Got: {out2_rerun}"
        )

    def test_downstream_first_with_file_dependency(self, nb_runner, tmp_path):
        """
        Like the above but cell 1 reads from a CSV file, adding a file dependency.
        This mirrors the financial_analysis_demo.ipynb scenario exactly.
        """
        import pandas as pd

        # Create test CSV
        csv_path = tmp_path / "data.csv"
        csv_str = str(csv_path).replace("\\", "/")
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "price": range(100),
                "volume": range(100, 200),
            }
        ).to_csv(csv_path, index=False)

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
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below

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
            f"Cell 2 should NOT auto-execute upstream. Got: {out2_rerun}"
        )

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
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below

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
            f"Cell 3 should NOT auto-execute upstream. Got: {out3_rerun}"
        )


class TestOutOfOrderEdgeCases:
    """Edge cases for out-of-order execution."""

    def test_middle_cell_first(self, nb_runner):
        """
        Execute the middle cell first. It should restore its inputs from
        cache and not mark anything as broken.
        """
        nb_runner.create_notebook(
            [
                "x = 42\nprint(f'x={x}')",
                "y = x * 2\nprint(f'y={y}')",
                "z = y + 1\nprint(f'z={z}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5\nprint(f'b={b}')",
                "c = b * 2\nprint(f'c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.enable_debug()  # needed for the "Auto-executing upstream" assertion below

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


def _fingerprint(output: str) -> str:
    """Return cell 4's ``CHK ...`` line, or fail loudly if it is absent.

    Comparing two absent fingerprints would compare equal and quietly assert
    nothing -- the exact way the wall-clock assertion this replaces went
    vacuous. An empty or missing line is a test failure, not a match.
    """
    for line in output.splitlines():
        if line.startswith("CHK "):
            return line.strip()
    raise AssertionError(f"cell 4 produced no CHK fingerprint line; got:\n{output[:1000]}")


# Tests for out-of-order execution: display cell → display cell → computation cell.
#
# Reproduces the bug where executing display cells out of order before a computation
# cell causes the computation cell to miss its cache, even though cached data exists
# on disk.
#
# Scenario (from financial_analysis_demo.ipynb):
#   Cell 1: import + load data (df = pd.read_csv(...))
#   Cell 2: preprocess (df = df.sort_values(...))
#   Cell 3: display df (just `df`)
#   Cell 4: heavy computation (df['col1'] = ...; df['col2'] = ...)
#   Cell 5: display df (just `df` again)
#
# First run: cells 1 → 2 → 3 → 4 → 5 (populates cache).
# After reset (simulating fresh session): execute cells 5 → 3 → 4.
#
# Originally written for a bug where `cell_code_changed` was computed as "this
# raw code has not run this session", which is true of EVERY cell in a fresh
# session, not just edited ones -- so cache lookups were disabled wholesale even
# though the on-disk data was valid. That mechanism is gone: neither
# `cell_code_changed` nor `force_recompute` exists in the source anymore.
#
# What still needs guarding is the behaviour: running the cells OUT OF ORDER
# after a reset must reconstruct ``df`` to exactly the value a top-to-bottom run
# produces. Cell 5 fires first and has to rebuild ``df`` from scratch, including
# re-establishing the seeded RNG stream, before cell 4's columns can be right.
#
# Deliberately NOT asserted: that cell 4 restores from cache. Both of its
# expensive statements are subscript stores on ``df``
# (``df['rolling_mean'] = ...``), which ``analyze_statement`` reports as
# ``top_level_mutated_vars={'df'}`` -- the in-place-mutation route sets
# ``skip_cache``, so those statements re-execute BY DESIGN and the sleep inside
# cell 4 always runs. The assertion this replaces was ``has_restore or
# t_elapsed < 0.5``; ``has_restore`` was False on passing runs too, so the test
# was decided purely by whether 0.3s of sleep plus ~0.15s of work fitted under
# 0.5s. It failed ~2 runs in 4 under parallel load and passed on an idle box,
# having never once verified a restore.
@pytest.mark.core
class TestOutOfOrderDisplayThenCompute:
    """Bug: Display cells executed out-of-order break computation cache restore."""

    def test_display_display_compute_order(self, nb_runner):
        """
        Scenario from financial_analysis_demo.ipynb:
        After cache is populated, reset and execute:
          Cell 5 (display after computation) →
          Cell 3 (display before computation) →
          Cell 4 (heavy computation)

        Cell 4 should restore from cache, NOT recompute.
        """
        import textwrap

        cell_1 = textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            import time
            np.random.seed(42)
            n = 500
            df = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=n, freq='D'),
                'ticker': ['AAPL'] * (n // 2) + ['GOOGL'] * (n // 2),
                'close': np.cumsum(np.random.randn(n)) + 100,
                'volume': np.random.randint(100, 10000, n),
            })
            print(f"Created df: {df.shape}")""")

        cell_2 = textwrap.dedent("""\
            df = df.sort_values(by=['ticker', 'date'])
            print(f"Sorted df: {df.shape}")""")

        cell_3 = textwrap.dedent("""\
            print("Display 1")
            df""")

        cell_4 = textwrap.dedent("""\
            t0 = time.time()
            df['rolling_mean'] = df.groupby('ticker')['close'].transform(
                lambda x: x.rolling(window=20).mean())
            elapsed1 = time.time() - t0
            print(f"Rolling mean: {elapsed1:.2f}s")

            t0 = time.time()
            time.sleep(0.3)
            df['vol_ratio'] = df.groupby('ticker')['volume'].transform(
                lambda x: x / x.rolling(window=10).mean())
            elapsed2 = time.time() - t0
            print(f"Vol ratio: {elapsed2:.2f}s")

            # Fingerprint of everything cell 1's seeded draw fed into: the
            # oracle for "was df reconstructed to the top-to-bottom value?".
            print(
                f"CHK close={df['close'].sum():.6f} "
                f"vol={df['volume'].sum()} "
                f"rm={df['rolling_mean'].sum():.6f} "
                f"vr={df['vol_ratio'].sum():.6f} "
                f"rows={len(df)}")

            df""")

        cell_5 = textwrap.dedent("""\
            print("Display 2")
            df""")

        nb_runner.create_notebook([cell_1, cell_2, cell_3, cell_4, cell_5])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # --- First pass: Run all cells in order to populate cache ---
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Created df" in out1

        out4_first = nb_runner.get_output(4)
        assert "Rolling mean" in out4_first
        assert "Vol ratio" in out4_first
        chk_first = _fingerprint(out4_first)

        # --- Reset cash state to simulate fresh session ---
        nb_runner.reset_cash_state()

        # Also clear user_ns variables to better simulate kernel restart
        # (only lineage/codes are cleared by reset_cash_state, not the actual vars)
        clear_code = """
for _v in ['df', 't0', 'elapsed1', 'elapsed2', 'n']:
    try:
        del globals()[_v]
    except KeyError:
        pass
"""
        nb_runner._run_async(nb_runner.client.kc._async_execute_interactive(clear_code, store_history=False))

        # --- Out-of-order execution: cell 5, then cell 3, then cell 4 ---
        # Cell 5: triggers upstream restore of fully computed df
        nb_runner.run_cell(5)
        out5_rerun = nb_runner.get_raw_output(5)
        print(f"\n=== Cell 5 rerun output (raw) ===\n{out5_rerun[:2000]}")

        # Cell 3: display df (before computation cell)
        nb_runner.run_cell(3)
        out3_rerun = nb_runner.get_raw_output(3)
        print(f"\n=== Cell 3 rerun output (raw) ===\n{out3_rerun[:2000]}")

        # Cell 4: heavy computation
        # This should restore from cache, NOT recompute
        t_start = time.time()
        nb_runner.run_cell(4)
        t_elapsed = time.time() - t_start
        out4_rerun = nb_runner.get_raw_output(4)

        out4_text = nb_runner.get_output(4)
        print(f"\n=== Cell 4 rerun output (filtered) ===\n{out4_text}")
        print(f"\n=== Cell 4 rerun output (raw) ===\n{out4_rerun[:2000]}")
        print(f"\n=== Cell 4 timing: {t_elapsed:.2f}s ===\n")

        assert "Rolling mean" in out4_text
        assert "Vol ratio" in out4_text

        # Cell 5 ran first with df deleted, so it had to rebuild df -- including
        # re-seeding the RNG stream -- before cells 3 and 4 could see it. If any
        # of that reconstruction is wrong (a draw replayed at the wrong stream
        # position, a stale df restored, the sort skipped), these numbers move.
        chk_rerun = _fingerprint(out4_text)
        assert chk_rerun == chk_first, (
            f"out-of-order execution reconstructed a DIFFERENT df.\n"
            f"  top-to-bottom: {chk_first}\n"
            f"  after 5->3->4: {chk_rerun}"
        )

        # Timing is reported for diagnosis only. It is NOT asserted on: cell 4
        # mutates df in place, so its statements re-execute by design and the
        # 0.3s sleep always runs. See the module docstring.
        print(f"(cell 4 recomputed in {t_elapsed:.2f}s -- expected, not asserted)")

    def test_display_before_compute_simple(self, nb_runner):
        """
        Simplified version: 4 cells.
        After cache populated, reset and run: cell 4, cell 3, cell 3_compute.

        Cell 1: x = 10
        Cell 2: y = x * 2 (expensive simulation with sleep)
        Cell 3: y  (display)
        Cell 4: z = y + 1; print(z)

        Reset, run: 3 → 2.
        Cell 2 should restore from cache, not recompute.
        """
        import textwrap

        cell_1 = textwrap.dedent("""\
            x = 10
            print(f"x={x}")""")

        cell_2 = textwrap.dedent("""\
            import time
            t0 = time.time()
            time.sleep(0.5)
            y = x * 2
            elapsed = time.time() - t0
            print(f"y={y}, took {elapsed:.2f}s")""")

        cell_3 = textwrap.dedent("""\
            y""")

        cell_4 = textwrap.dedent("""\
            z = y + 1
            print(f"z={z}")""")

        nb_runner.create_notebook([cell_1, cell_2, cell_3, cell_4])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # First pass: populate cache
        nb_runner.run_all()
        assert "y=20" in nb_runner.get_output(2)
        assert "z=21" in nb_runner.get_output(4)

        # Reset
        nb_runner.reset_cash_state()

        # Out-of-order: cell 4 first, then cell 3, then cell 2
        nb_runner.run_cell(4)
        assert "z=21" in nb_runner.get_output(4)

        nb_runner.run_cell(3)

        # Now run cell 2 - should restore from cache, not sleep 0.5s
        t_start = time.time()
        nb_runner.run_cell(2)
        t_elapsed = time.time() - t_start

        out2_rerun = nb_runner.get_output(2)
        assert "y=20" in out2_rerun

        # If it took > 0.4s, it recomputed (sleep was 0.5s)
        # If cache restored, it should be < 0.3s
        assert t_elapsed < 0.4, (
            f"Cell 2 took {t_elapsed:.2f}s, suggesting it recomputed instead of "
            f"restoring from cache. Debug output: {nb_runner.get_raw_output(2)[:500]}"
        )
