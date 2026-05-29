"""
Batch 102 — Out-of-order execution & re-execution patterns.

Tests that exercise:
- Running cells in non-sequential order
- Re-running earlier cells after later ones
- Skipping cells then coming back to them
- Running cell 3 before cell 2 (non-linear workflows)
- Re-running cells without changes (idempotency)
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream]


class TestOutOfOrderExecution:
    """Run cells in non-sequential order and check consistency."""

    def test_run_last_before_middle(self, nb_runner):
        """Run cell 3 before cell 2 — cell 3 should trigger upstream."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "z = y * 2\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        # Run cell 1, skip cell 2, run cell 3
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "z = 30" in nb_runner.get_output(3)

    def test_run_cell_then_rerun_earlier(self, nb_runner):
        """Run all, then re-run cell 2 again — should skip or use cache."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5\nprint(f'y = {y}')",
            "z = y * 2\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)
        assert "z = 30" in nb_runner.get_output(3)

        # Re-run cell 2 — should produce same result
        nb_runner.run_cell(2)
        assert "y = 15" in nb_runner.get_output(2)

    def test_run_cells_reverse_order(self, nb_runner):
        """Run cells 3, 2, 1 — upstream checker should handle correctly."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 10\nprint(f'b = {b}')",
            "c = b + 100\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        # Run in reverse
        nb_runner.run_cell(3)
        assert "c = 111" in nb_runner.get_output(3)

        nb_runner.run_cell(2)
        assert "b = 11" in nb_runner.get_output(2)

        nb_runner.run_cell(1)

    def test_skip_middle_then_run_it(self, nb_runner):
        """Skip cell 2 initially, run cell 3, then run cell 2."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 3\nprint(f'y = {y}')",
            "z = x + 10\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
        assert "z = 15" in nb_runner.get_output(3)

        # Now run the skipped cell 2
        nb_runner.run_cell(2)
        assert "y = 15" in nb_runner.get_output(2)


class TestRerunWithoutChanges:
    """Re-running cells without changes should be idempotent."""


    def test_idempotent_rerun_single_cell(self, nb_runner):
        """Run cell 3 multiple times — each should give same result."""
        nb_runner.create_notebook([
            "val = 7",
            "doubled = val * 2",
            "print(f'doubled = {doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled = 14" in nb_runner.get_output(3)

        for _ in range(3):
            nb_runner.run_cell(3)
            assert "doubled = 14" in nb_runner.get_output(3)

    def test_rerun_all_three_times(self, nb_runner):
        """Run all three times — regression check for state accumulation bugs."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 5",
            "c = b * 2\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        for _ in range(3):
            nb_runner.run_all()
            assert "c = 30" in nb_runner.get_output(3)


class TestEditBetweenOutOfOrderRuns:
    """Combine edits with out-of-order execution."""

    def test_edit_cell1_run_cell3_then_cell2(self, nb_runner):
        """Edit cell 1, run cell 3 (should auto-propagate), then run cell 2."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x + 1\nprint(f'y = {y}')",
            "z = x * 10\nprint(f'z = {z}')",
        ])
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
        nb_runner.create_notebook([
            "x = 3",
            "y = x ** 2",
            "z = y + x\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2])

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_cell(3)
        assert "z = 110" in nb_runner.get_output(3)

    def test_interleaved_edits_and_runs(self, nb_runner):
        """Edit cell 1, run cell 2, edit cell 1 again, run cell 3."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 10\nprint(f'y = {y}')",
            "z = y * 2\nprint(f'z = {z}')",
        ])
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


class TestSelfAssignmentInteractions:
    """Test cache coherence with self-assignment patterns (df = df.something())."""

    def test_self_assignment_chain(self, nb_runner):
        """Self-assignment across cells with upstream edit."""
        nb_runner.create_notebook([
            "data = [3, 1, 2]",
            "data = sorted(data)",
            "result = data[0]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(3)

        # Change initial data
        nb_runner.set_cell_source(1, "data = [30, 10, 20]")
        nb_runner.run_cell(3)
        assert "result = 10" in nb_runner.get_output(3)

    def test_self_assignment_rerun(self, nb_runner):
        """Self-assignment re-run should not accumulate."""
        nb_runner.create_notebook([
            "items = [1, 2, 3]",
            "items = [x * 2 for x in items]",
            "print(f'items = {items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [2, 4, 6]" in nb_runner.get_output(3)

        # Re-run all — should NOT produce [4, 8, 12]
        nb_runner.run_all()
        assert "items = [2, 4, 6]" in nb_runner.get_output(3)


class TestDiamondDependency:
    """Multiple cells depend on the same upstream cell."""

    def test_diamond_edit_root(self, nb_runner):
        """
        Cell 1: x = 10
        Cell 2: y = x + 1
        Cell 3: z = x * 2
        Cell 4: w = y + z  (diamond dependency on x through y and z)
        """
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 1",
            "z = x * 2",
            "w = y + z\nprint(f'w = {w}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 31" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(4)
        assert "w = 16" in nb_runner.get_output(4)

    def test_diamond_edit_one_branch(self, nb_runner):
        """Edit only one branch of the diamond."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 1",
            "z = x * 2",
            "w = y + z\nprint(f'w = {w}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 31" in nb_runner.get_output(4)

        # Change only cell 2 formula (one branch)
        # y = 10 + 100 = 110, z = 10 * 2 = 20, w = 110 + 20 = 130
        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cell(4)
        assert "w = 130" in nb_runner.get_output(4)
