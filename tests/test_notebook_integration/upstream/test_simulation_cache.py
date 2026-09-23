"""The upstream simulation restoring values instead of re-running them."""

import pytest

pytestmark = [pytest.mark.stress]


# Simulation cache coherence under edits.
#
# The simulation cache stores (cell_hash, virtual_lineage, ...) for each cell.
# These tests probe whether the simulation cache is correctly invalidated when:
# - A cell is edited (hash changes → must re-simulate from that point)
# - The same cell is edited back (hash reverts → should match old cache)
# - Cells are inserted/removed (index shifts)
# - Multiple cells change simultaneously
@pytest.mark.upstream
class TestSimulationCacheInvalidation:
    """Verify simulation cache invalidation on cell edits."""

    def test_edit_first_cell_invalidates_subsequent(self, nb_runner):
        """Editing cell 1 must invalidate simulation cache for cells 2+."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 10",
                "z = y + 100",
                "print(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 111" in nb_runner.get_output(4)

        # Run again — builds simulation cache
        nb_runner.run_cell(4)
        assert "z = 111" in nb_runner.get_output(4)

        # Edit cell 1 — must invalidate cached simulation
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cell(4)
        assert "z = 160" in nb_runner.get_output(4)

    def test_edit_middle_preserves_earlier_cache(self, nb_runner):
        """Editing cell 2 should preserve cache for cell 1 but invalidate cell 3+."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5",
                "c = b * 2",
                "d = c + 1\nprint(f'd = {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 31" in nb_runner.get_output(4)

        # Build cache
        nb_runner.run_cell(4)

        # Edit only cell 2
        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.run_cell(4)
        assert "d = 201" in nb_runner.get_output(4)

    def test_revert_cell_uses_cached_values(self, nb_runner):
        """Edit cell 1, then revert — second revert should use cached computations."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 3",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(3)
        assert "y = 300" in nb_runner.get_output(3)

        # Revert
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(3)
        assert "y = 15" in nb_runner.get_output(3)

    def test_multiple_edits_same_cell_cache_churn(self, nb_runner):
        """Multiple edits to the same cell in sequence — cache should adapt."""
        nb_runner.create_notebook(
            [
                "val = 0",
                "result = val + 1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(2)

        for i in range(1, 6):
            nb_runner.set_cell_source(1, f"val = {i * 10}")
            nb_runner.run_cell(2)
            expected = i * 10 + 1
            assert f"result = {expected}" in nb_runner.get_output(2), (
                f"Failed at iteration {i}: expected result = {expected}"
            )


@pytest.mark.upstream
class TestSimulationCacheWithFunctions:
    """Functions defined in cells and the simulation cache."""

    def test_function_redefinition_invalidates_cache(self, nb_runner):
        """Redefining a function should cause downstream re-computation."""
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "result = transform(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3")
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_function_redefinition_then_revert(self, nb_runner):
        """Redefine function, then revert — should hit cache."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "val = double(7)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def double(x):\n    return x * 3")
        nb_runner.run_cell(2)
        assert "val = 21" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def double(x):\n    return x * 2")
        nb_runner.run_cell(2)
        assert "val = 14" in nb_runner.get_output(2)

    def test_function_used_in_multiple_downstream_cells(self, nb_runner):
        """Function changed → all downstream cells should update."""
        nb_runner.create_notebook(
            [
                "def f(x):\n    return x + 1",
                "a = f(10)\nprint(f'a = {a}')",
                "b = f(20)\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11" in nb_runner.get_output(2)
        assert "b = 21" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def f(x):\n    return x + 100")
        nb_runner.run_cell(2)
        assert "a = 110" in nb_runner.get_output(2)
        nb_runner.run_cell(3)
        assert "b = 120" in nb_runner.get_output(3)


@pytest.mark.upstream
class TestSimulationCacheWithMultiStatement:
    """Cells with multiple statements and simulation cache."""

    def test_multi_statement_cell_edit(self, nb_runner):
        """Edit a cell with multiple statements."""
        nb_runner.create_notebook(
            [
                "x = 10\ny = 20",
                "z = x + y\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100\ny = 200")
        nb_runner.run_cell(2)
        assert "z = 300" in nb_runner.get_output(2)

    def test_add_statement_to_cell(self, nb_runner):
        """Add a new statement to an existing cell."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)

        # Add a second statement to cell 1
        nb_runner.set_cell_source(1, "x = 5\nbonus = 100")
        nb_runner.set_cell_source(2, "y = x + 1 + bonus\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 106" in nb_runner.get_output(2)

    def test_remove_statement_from_cell(self, nb_runner):
        """Remove a statement from a multi-statement cell."""
        nb_runner.create_notebook(
            [
                "x = 5\nbonus = 100",
                "y = x + bonus\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 105" in nb_runner.get_output(2)

        # Remove bonus — cell 2 needs to change too
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 6" in nb_runner.get_output(2)


# Virtual restore + cell edit interaction tests.
#
# Tests that exercise virtual restore from disk cache after kernel restart,
# combined with cell edits and dependency changes.
@pytest.mark.restore
@pytest.mark.timeout(30)
class TestVirtualRestoreBasic:
    """Virtual restore from disk after restart."""

    def test_restore_simple_chain(self, nb_runner):
        """After restart, run last cell — upstream should be restored."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 85" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(3)
        assert "z = 85" in nb_runner.get_output(3)

    def test_restore_then_edit_upstream(self, nb_runner):
        """Restore from cache, then edit upstream cell and re-run."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        # First restore
        nb_runner.run_cell(2)
        assert "y = 15" in nb_runner.get_output(2)

        # Now edit upstream and re-run all
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 105" in nb_runner.get_output(2)

    def test_restore_multi_variable(self, nb_runner):
        """Restore multiple variables from a single cell."""
        nb_runner.create_notebook(
            [
                "a = 10\nb = 20",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 30" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "c = 30" in nb_runner.get_output(2)


@pytest.mark.restore
@pytest.mark.timeout(30)
class TestVirtualRestoreWithEdits:
    """Edit cells after restart, verify correct behavior."""

    def test_edit_before_restore(self, nb_runner):
        """Edit a cell before restarting, then run all."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 3\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)

        # Edit then restart
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 150" in nb_runner.get_output(2)

    def test_edit_leaf_after_restore(self, nb_runner):
        """Restore, then edit only the leaf cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        # Restore
        nb_runner.run_cell(2)
        assert "y = 11" in nb_runner.get_output(2)

        # Edit leaf
        nb_runner.set_cell_source(2, "y = x * 10\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 100" in nb_runner.get_output(2)

    def test_two_restarts_with_edits(self, nb_runner):
        """Restart twice with edits in between."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # First restart + edit
        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Second restart + edit
        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)


@pytest.mark.restore
@pytest.mark.timeout(30)
class TestVirtualRestoreWithFunctions:
    """Virtual restore of function definitions."""

    def test_restore_function_def(self, nb_runner):
        """Function defined in cell 1, used in cell 2 — restore cell 2."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "result = double(7)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "result = 14" in nb_runner.get_output(2)

    def test_edit_function_after_restore(self, nb_runner):
        """Restore, then edit the function and re-run."""
        nb_runner.create_notebook(
            [
                "def square(x):\n    return x ** 2",
                "val = square(5)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 25" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        # Edit function before running
        nb_runner.set_cell_source(1, "def square(x):\n    return x ** 3")
        nb_runner.run_all()
        assert "val = 125" in nb_runner.get_output(2)


@pytest.mark.restore
@pytest.mark.timeout(30)
class TestVirtualRestoreWithFiles:
    """Virtual restore with file dependencies."""

    def test_restore_with_unchanged_file(self, nb_runner, tmp_path):
        """File unchanged after restart — restore should work."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("42")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{path_str}') as f:\n    val = int(f.read().strip())",
                "result = val * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

    def test_restore_with_changed_file(self, nb_runner, tmp_path):
        """File changed after restart — should recompute."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("10")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{path_str}') as f:\n    val = int(f.read().strip())",
                "result = val * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Change file
        import time

        time.sleep(0.1)
        data_file.write_text("99")

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 198" in nb_runner.get_output(2)


# Simulation cache edge case interaction tests (advanced).
#
# Tests that specifically stress the simulation cache and upstream
# detection logic with tricky patterns that could cause divergence.
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestSimCacheAfterEdits:
    """Verify simulation cache remains coherent after edits."""

    def test_edit_root_long_chain_no_restart(self, nb_runner):
        """Edit root of long chain without restart.
        Upstream simulation should propagate correctly."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1",
                "z = y + 1",
                "w = z + 1",
                "result = w\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4" in nb_runner.get_output(5)

        # Edit root, run only last cell (upstream must detect and re-exec)
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(5)
        assert "result = 103" in nb_runner.get_output(5)

    def test_edit_two_independent_roots(self, nb_runner):
        """Edit two independent roots, verify both paths update."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = 2",
                "c = a * 10",
                "d = b * 10",
                "result = c + d\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(5)

        # Edit both roots
        nb_runner.set_cell_source(1, "a = 5")
        nb_runner.set_cell_source(2, "b = 7")
        nb_runner.run_cell(5)
        assert "result = 120" in nb_runner.get_output(5)

    def test_repeated_edits_same_cell_five_times(self, nb_runner):
        """Edit the same cell 5 times, run last each time."""
        nb_runner.create_notebook(
            [
                "x = 0",
                "result = x * 10\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0" in nb_runner.get_output(2)

        for val in [1, 2, 3, 4, 5]:
            nb_runner.set_cell_source(1, f"x = {val}")
            nb_runner.run_cell(2)
            assert f"result = {val * 10}" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestSimCacheWithRestart:
    """Simulation cache coherence after kernel restart."""

    def test_edit_restart_edit_again(self, nb_runner):
        """Edit → restart → edit again → verify coherence."""
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

        # First edit + restart
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 12" in nb_runner.get_output(3)

        # Second edit + restart
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 102" in nb_runner.get_output(3)

    def test_restart_without_edit_restores(self, nb_runner):
        """Restart without any edits — should restore from cache."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestUpstreamPropagationEdges:
    """Edge cases in upstream propagation detection."""

    def test_edit_does_not_change_output_value(self, nb_runner):
        """Edit code but the output value doesn't change."""
        nb_runner.create_notebook(
            [
                "x = 5 + 5",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Change code but same result
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_edit_comment_only_change(self, nb_runner):
        """Edit only a comment — code is different but effect is same."""
        nb_runner.create_notebook(
            [
                "x = 10  # initial value",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Only change comment
        nb_runner.set_cell_source(1, "x = 10  # updated comment")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_whitespace_only_change_still_correct(self, nb_runner):
        """Change only whitespace — should still produce correct results."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Add trailing whitespace
        nb_runner.set_cell_source(1, "x = 10  ")
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

    def test_reorder_independent_cells_swap(self, nb_runner):
        """Reorder two independent cells (swap order in notebook)."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 20",
                "result = a + b\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Swap cells 1 and 2
        nb_runner.set_cell_source(1, "b = 20")
        nb_runner.set_cell_source(2, "a = 10")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)
