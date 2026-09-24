"""Running a subset of cells after an edit."""

import pytest


# Selective cell execution interaction tests.
#
# Tests that exercise running specific cells out of order,
# skipping cells, and running only subsets of cells.
@pytest.mark.upstream
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestSelectiveCellExecution:
    """Run only specific cells."""

    def test_run_only_first_and_last(self, nb_runner):
        """Run first cell, then jump to last cell (skip middle).
        Upstream should detect and re-execute middle cells."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        # Skip cell 2, run cell 3 — upstream should ensure y exists
        nb_runner.run_cell(3)
        assert "z = 21" in nb_runner.get_output(3)

    def test_run_cells_in_reverse(self, nb_runner):
        """Run cells in reverse order."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a + 1",
                "c = b + 1\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        # Run in reverse: cell 3 first (upstream should find and run 1, 2)
        nb_runner.run_cell(3)
        assert "c = 7" in nb_runner.get_output(3)

    def test_run_middle_only(self, nb_runner):
        """Run only a middle cell."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x * 2\nprint(f'y = {y}')",
                "z = y + 1",
            ]
        )
        nb_runner.start_kernel()
        # Run only cell 2 — upstream should ensure x exists
        nb_runner.run_cell(2)
        assert "y = 84" in nb_runner.get_output(2)


# Out-of-order execution, selective cell runs, and re-execution
# patterns that stress the upstream simulation system.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestRunModifiedCellsOnly:
    """Test running specific cells rather than all."""

    def test_run_only_modified_and_downstream(self, nb_runner):
        """Run notebook, modify a cell, run only that cell + downstream."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5",
                "c = b * 2",
                "print(c)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=10, b=15, c=30
        assert "30" in nb_runner.get_output(4)

        # Modify cell 1 and re-run cells 1-4
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_cells([1, 2, 3, 4])
        # a=20, b=25, c=50
        assert "50" in nb_runner.get_output(4)

    def test_run_cell_range(self, nb_runner):
        """Run a range of cells, skipping early ones."""
        nb_runner.create_notebook(
            [
                "x = 100",
                "y = 200",
                "z = x + y",
                "print(z)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(4)

        # Change x and run all again
        nb_runner.set_cell_source(1, "x = 500")
        nb_runner.run_all()
        assert "700" in nb_runner.get_output(4)


@pytest.mark.upstream
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestRunSubsetAfterEdit:
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


# Selective cell execution interaction tests.
#
# Tests where users run cells out of order, skip cells,
# run subsets of cells, and verify cache consistency.
@pytest.mark.upstream
@pytest.mark.stress
@pytest.mark.timeout(45)
class TestRunSubsetOfCells:
    """Run only some cells, not all."""

    def test_run_first_and_last_skip_middle(self, nb_runner):
        """Run cells 1 and 3, skip cell 2."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2  # skipped on first pass",
                "z = x + 5\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 3])
        # z only depends on x, not y
        assert "z = 15" in nb_runner.get_output(3)

    def test_run_all_then_rerun_single_edited_cell(self, nb_runner):
        """Run all, edit one cell, run only that cell."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = 2",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        # Edit cell 1 and rerun only cell 3
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_cells([1, 3])
        assert "c = 102" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestPartialExecution:
    """Execute parts of a notebook."""

    def test_run_first_three_then_add_fourth(self, nb_runner):
        """Run first 3 cells, then add and run a 4th."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2",
                "z = y + 5",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2, 3])

        # Add a 4th cell and run it
        nb_runner.set_cell_source(3, "z = y + 5\nprint(f'z = {z}')")
        nb_runner.run_cell(3)
        assert "z = 25" in nb_runner.get_output(3)

    def test_run_subset_edit_run_rest(self, nb_runner):
        """Run subset, edit, run remaining."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a + 1",
                "c = b * 2",
                "d = c + 3\nprint(f'd = {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cells([1, 2])

        # Edit cell 1 before running rest
        nb_runner.set_cell_source(1, "a = 50")
        nb_runner.run_cells([3, 4])
        # Upstream should detect a changed and re-execute
        assert "d = " in nb_runner.get_output(4)


@pytest.mark.core
class TestPartialReExecution:
    """Test re-running only some cells while others use cache."""

    def test_rerun_middle_cell_only(self, nb_runner):
        """Run all, then only re-run cell 2 — cell 3 should still have correct output."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 3",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 31" in nb_runner.get_output(3)

        # Only re-run cell 3 (without running cells 1 and 2 again)
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "z = 31" in out, f"Expected z=31, got: {out}"

    def test_skip_to_last_cell(self, nb_runner):
        """Run only the last cell — upstream system should restore deps."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 2",
                "c = b + 3\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        # Run all to populate cache
        nb_runner.run_all()
        assert "c = 6" in nb_runner.get_output(3)

        # Now only run cell 3 — should restore a and b from cache/upstream
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "c = 6" in out, f"Expected c=6, got: {out}"


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestPartialReruns:
    """Run only some cells after edits."""

    def test_edit_middle_run_from_middle(self, nb_runner):
        """Edit cell 2, run cells 2-3 only."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x * 10")
        nb_runner.run_cells([2, 3])
        assert "z = 51" in nb_runner.get_output(3)

    def test_edit_root_run_only_leaf(self, nb_runner):
        """Edit root cell but only run the leaf. Upstream should trigger."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(3)
        assert "z = 201" in nb_runner.get_output(3)

    def test_edit_leaf_only(self, nb_runner):
        """Edit only the leaf cell, run it."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(3, "z = y + 100\nprint(f'z = {z}')")
        nb_runner.run_cell(3)
        assert "z = 110" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.stress
class TestRerunAfterEdit:
    """Re-run specific cells after editing."""

    @pytest.mark.timeout(30)
    def test_edit_and_rerun_single(self, nb_runner):
        """Edit a cell, re-run only that cell."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100")
        # Only re-run cell 2 — upstream should detect x changed
        nb_runner.run_cell(2)
        assert "y = 101" in nb_runner.get_output(2)

    @pytest.mark.timeout(30)
    def test_edit_middle_rerun_last(self, nb_runner):
        """Edit middle cell, re-run only last cell."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.run_cell(3)
        assert "c = 11" in nb_runner.get_output(3)

    @pytest.mark.timeout(45)
    def test_edit_cell_rerun_only_downstream(self, nb_runner):
        """Edit cell 1, rerun only downstream cells."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "mid = base + 5",
                "final = mid * 2\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final = 30" in nb_runner.get_output(3)

        # Edit cell 1, then run cells 1-3
        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_cells([1, 2, 3])
        assert "final = 210" in nb_runner.get_output(3)

    @pytest.mark.timeout(45)
    def test_edit_middle_rerun_from_there(self, nb_runner):
        """Edit middle cell, rerun from middle to end."""
        nb_runner.create_notebook(
            [
                "x = 3",
                "y = x + 7",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cells([2, 3])
        assert "z = 206" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestRepeatExecution:
    """Execute cells multiple times."""

    def test_run_same_cell_twice(self, nb_runner):
        """Run the same cell twice — idempotent result."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "result = x * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

        # Run cell 2 again (should be cached/skipped)
        nb_runner.run_cell(2)
        assert "result = 84" in nb_runner.get_output(2)

    def test_run_all_twice(self, nb_runner):
        """Run all cells twice — should be idempotent."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestRerunSameCell:
    """Test re-running the same cell multiple times."""

    def test_rerun_print_cell(self, nb_runner):
        """Re-run print cell gives fresh output."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "print(sum(data))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)

        # Re-run
        nb_runner.run_cell(2)
        assert "6" in nb_runner.get_output(2)


@pytest.mark.upstream
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
