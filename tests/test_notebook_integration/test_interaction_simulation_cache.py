"""
Batch 103 — Simulation cache coherence under edits.

The simulation cache stores (cell_hash, virtual_lineage, ...) for each cell.
These tests probe whether the simulation cache is correctly invalidated when:
- A cell is edited (hash changes → must re-simulate from that point)
- The same cell is edited back (hash reverts → should match old cache)
- Cells are inserted/removed (index shifts)
- Multiple cells change simultaneously
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream]


class TestSimulationCacheInvalidation:
    """Verify simulation cache invalidation on cell edits."""

    def test_edit_first_cell_invalidates_subsequent(self, nb_runner):
        """Editing cell 1 must invalidate simulation cache for cells 2+."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 10",
            "z = y + 100",
            "print(f'z = {z}')",
        ])
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
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 5",
            "c = b * 2",
            "d = c + 1\nprint(f'd = {d}')",
        ])
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
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 3",
            "print(f'y = {y}')",
        ])
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
        nb_runner.create_notebook([
            "val = 0",
            "result = val + 1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(2)

        for i in range(1, 6):
            nb_runner.set_cell_source(1, f"val = {i * 10}")
            nb_runner.run_cell(2)
            expected = i * 10 + 1
            assert f"result = {expected}" in nb_runner.get_output(2), \
                f"Failed at iteration {i}: expected result = {expected}"


class TestSimulationCacheWithFunctions:
    """Functions defined in cells and the simulation cache."""

    def test_function_redefinition_invalidates_cache(self, nb_runner):
        """Redefining a function should cause downstream re-computation."""
        nb_runner.create_notebook([
            "def transform(x):\n    return x * 2",
            "result = transform(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3")
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_function_redefinition_then_revert(self, nb_runner):
        """Redefine function, then revert — should hit cache."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2",
            "val = double(7)\nprint(f'val = {val}')",
        ])
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
        nb_runner.create_notebook([
            "def f(x):\n    return x + 1",
            "a = f(10)\nprint(f'a = {a}')",
            "b = f(20)\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11" in nb_runner.get_output(2)
        assert "b = 21" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def f(x):\n    return x + 100")
        nb_runner.run_cell(2)
        assert "a = 110" in nb_runner.get_output(2)
        nb_runner.run_cell(3)
        assert "b = 120" in nb_runner.get_output(3)


class TestSimulationCacheWithMultiStatement:
    """Cells with multiple statements and simulation cache."""

    def test_multi_statement_cell_edit(self, nb_runner):
        """Edit a cell with multiple statements."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "z = x + y\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100\ny = 200")
        nb_runner.run_cell(2)
        assert "z = 300" in nb_runner.get_output(2)

    def test_add_statement_to_cell(self, nb_runner):
        """Add a new statement to an existing cell."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x + 1\nprint(f'y = {y}')",
        ])
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
        nb_runner.create_notebook([
            "x = 5\nbonus = 100",
            "y = x + bonus\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 105" in nb_runner.get_output(2)

        # Remove bonus — cell 2 needs to change too
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 6" in nb_runner.get_output(2)
