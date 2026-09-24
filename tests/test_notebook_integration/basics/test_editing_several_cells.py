"""Several cells edited between two runs, and an edit followed by running only some cells."""

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.upstream
class TestMultiCellEdits:
    """Edit multiple cells between a single run."""

    def test_edit_two_upstream_cells(self, nb_runner):
        """Edit both cell 1 and cell 2 before running cell 3."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = 20",
                "z = x + y\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(3)

        # Edit BOTH upstream cells
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.set_cell_source(2, "y = 200")
        nb_runner.run_cell(3)
        assert "z = 300" in nb_runner.get_output(3)

    def test_edit_upstream_and_downstream(self, nb_runner):
        """Edit cell 1 (upstream) and cell 3 (downstream formula)."""
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

        # Change upstream value AND downstream formula
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.set_cell_source(3, "z = y + 100\nprint(f'z = {z}')")
        nb_runner.run_cell(3)
        assert "z = 120" in nb_runner.get_output(3)

    def test_edit_all_cells_simultaneously(self, nb_runner):
        """Edit all three cells at once before running the last one."""
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

        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.set_cell_source(2, "b = a * 2")
        nb_runner.set_cell_source(3, "c = b - 50\nprint(f'c = {c}')")
        nb_runner.run_cell(3)
        assert "c = 150" in nb_runner.get_output(3)


@pytest.mark.upstream
class TestEditThenRunPartial:
    """Edit cells and run only some of them — tests cache consistency."""

    def test_edit_upstream_run_only_downstream(self, nb_runner):
        """Edit cell 1 but only re-run cell 3. Upstream should auto-propagate."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 5",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 20")
        # Only run cell 3 — upstream checker should detect change and re-execute
        nb_runner.run_cell(3)
        assert "z = 50" in nb_runner.get_output(3)

    def test_edit_middle_run_only_last(self, nb_runner):
        """Edit cell 2 (middle) and only run cell 3."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x + 1",
                "print(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x * 100")
        nb_runner.run_cell(3)
        assert "y = 500" in nb_runner.get_output(3)

    def test_edit_first_run_middle_then_last(self, nb_runner):
        """Edit cell 1, run cell 2, then run cell 3 separately."""
        nb_runner.create_notebook(
            [
                "x = 3",
                "y = x ** 2\nprint(f'y = {y}')",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 9" in nb_runner.get_output(2)
        assert "z = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 4")
        nb_runner.run_cell(2)
        assert "y = 16" in nb_runner.get_output(2)

        nb_runner.run_cell(3)
        assert "z = 17" in nb_runner.get_output(3)


@pytest.mark.upstream
class TestLongChainEdits:
    """Test propagation through long dependency chains with edits."""

    def test_five_cell_chain_edit_root(self, nb_runner):
        """5-cell chain: edit root and verify propagation to the end."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 5" in nb_runner.get_output(5)

        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_cell(5)
        assert "e = 104" in nb_runner.get_output(5)

    def test_five_cell_chain_edit_middle(self, nb_runner):
        """5-cell chain: edit middle cell and verify downstream propagation."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b * 10",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 22" in nb_runner.get_output(5)

        # Edit middle (cell 3)
        nb_runner.set_cell_source(3, "c = b * 100")
        nb_runner.run_cell(5)
        assert "e = 202" in nb_runner.get_output(5)

    def test_chain_edit_root_twice(self, nb_runner):
        """Edit root of a chain twice — second edit should still propagate."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x * 2",
                "z = y * 3",
                "w = z * 4\nprint(f'w = {w}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "w = 24" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_cell(4)
        assert "w = 240" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(4)
        assert "w = 120" in nb_runner.get_output(4)
