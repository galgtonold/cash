"""Several rounds of edits and re-runs on one notebook."""

import pytest


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestMultiRoundWorkflows:
    """Multiple rounds of edits and reruns."""

    def test_three_rounds_of_edits(self, nb_runner):
        """Three successive rounds of editing the same cell."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit 1
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Edit 2
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)

        # Edit 3
        nb_runner.set_cell_source(1, "x = 1000")
        nb_runner.run_all()
        assert "y = 1001" in nb_runner.get_output(2)

    def test_alternating_cell_edits(self, nb_runner):
        """Alternate editing two different cells."""
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

        # Edit a
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "c = 12" in nb_runner.get_output(3)

        # Edit b
        nb_runner.set_cell_source(2, "b = 20")
        nb_runner.run_all()
        assert "c = 30" in nb_runner.get_output(3)

        # Edit a again
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()
        assert "c = 120" in nb_runner.get_output(3)

    def test_edit_with_intermediate_restart(self, nb_runner):
        """Edit, restart, edit again."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit and run
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Restart
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Edit again
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)

    def test_progressive_notebook_building(self, nb_runner):
        """Build a notebook progressively: run cells as they're added.
        This simulates typical notebook usage patterns."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "total = sum(data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Now edit cell 1 to add more data
        nb_runner.set_cell_source(1, "data = [1, 2, 3, 4, 5]")
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

    def test_revert_all_changes(self, nb_runner):
        """Make edits, then revert everything back to original."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 3" in nb_runner.get_output(3)

        # Edit all cells
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.set_cell_source(2, "y = x * 2")
        nb_runner.set_cell_source(3, "z = y * 3\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 600" in nb_runner.get_output(3)

        # Revert all
        nb_runner.set_cell_source(1, "x = 1")
        nb_runner.set_cell_source(2, "y = x + 1")
        nb_runner.set_cell_source(3, "z = y + 1\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 3" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestMultipleEditsInSequence:
    """Multiple sequential edits to the same chain."""

    def test_three_edits_to_root(self, nb_runner):
        """Edit root three times in sequence."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x * 10",
                "z = y + 5\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 2")
        nb_runner.run_all()
        assert "z = 25" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "z = 55" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "z = 105" in nb_runner.get_output(3)

    def test_edit_different_cells_alternating(self, nb_runner):
        """Alternate between editing cell 1 and cell 2."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5",
                "c = b * 2\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # c = (10+5)*2 = 30
        assert "c = 30" in nb_runner.get_output(3)

        # Edit cell 1
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_all()
        # c = (20+5)*2 = 50
        assert "c = 50" in nb_runner.get_output(3)

        # Edit cell 2
        nb_runner.set_cell_source(2, "b = a + 100")
        nb_runner.run_all()
        # c = (20+100)*2 = 240
        assert "c = 240" in nb_runner.get_output(3)

        # Edit cell 1 again
        nb_runner.set_cell_source(1, "a = 0")
        nb_runner.run_all()
        # c = (0+100)*2 = 200
        assert "c = 200" in nb_runner.get_output(3)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.upstream
class TestRapidModification:
    """Test rapid modifications and re-runs."""

    def test_multiple_rapid_changes(self, nb_runner):
        """Make several changes and re-run each time."""
        nb_runner.create_notebook(
            [
                "n = 1",
                "result = n * 100",
                "print(result)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

        for val in [2, 5, 10, 50]:
            nb_runner.set_cell_source(1, f"n = {val}")
            nb_runner.run_all()
            assert str(val * 100) in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.stress
@pytest.mark.timeout(45)
class TestRepeatedSingleCellRuns:
    """Run same cell repeatedly with edits."""

    def test_edit_and_rerun_same_cell_many_times(self, nb_runner):
        """Edit and rerun the same output cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "result = x\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        for multiplier in [2, 3, 5]:
            nb_runner.set_cell_source(2, f"result = x * {multiplier}\nprint(f'result = {{result}}')")
            nb_runner.run_cell(2)
            assert f"result = {10 * multiplier}" in nb_runner.get_output(2)
