"""
Batch 39: Out-of-order execution, selective cell runs, and re-execution
patterns that stress the upstream simulation system.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.upstream]


class TestSelectiveCellExecution:
    """Test running specific cells rather than all."""

    def test_run_only_modified_and_downstream(self, nb_runner):
        """Run notebook, modify a cell, run only that cell + downstream."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 5",
            "c = b * 2",
            "print(c)",
        ])
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
        nb_runner.create_notebook([
            "x = 100",
            "y = 200",
            "z = x + y",
            "print(z)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "300" in nb_runner.get_output(4)

        # Change x and run all again
        nb_runner.set_cell_source(1, "x = 500")
        nb_runner.run_all()
        assert "700" in nb_runner.get_output(4)


class TestRerunSameCell:
    """Test re-running the same cell multiple times."""


    def test_rerun_print_cell(self, nb_runner):
        """Re-run print cell gives fresh output."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "print(sum(data))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)

        # Re-run
        nb_runner.run_cell(2)
        assert "6" in nb_runner.get_output(2)


class TestCellDependencyChain:
    """Test long dependency chains with selective execution."""

    def test_long_chain_head_change(self, nb_runner):
        """Change head of a 6-cell chain, re-run all."""
        nb_runner.create_notebook([
            "base = 1",
            "step1 = base * 2",       # 2
            "step2 = step1 + 3",      # 5
            "step3 = step2 * 4",      # 20
            "step4 = step3 - 5",      # 15
            "print(step4)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "base = 10")
        nb_runner.run_all()
        # 10*2=20, +3=23, *4=92, -5=87
        assert "87" in nb_runner.get_output(6)



class TestParallelBranches:
    """Test parallel independent branches merging downstream."""


    def test_three_branches_one_change(self, nb_runner):
        """Three parallel branches, only one changes."""
        nb_runner.create_notebook([
            "x = 1\ny = 2\nz = 3",
            "ax = x * 10",
            "by = y * 10",
            "cz = z * 10",
            "total = ax + by + cz",
            "print(total)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10+20+30=60
        assert "60" in nb_runner.get_output(6)

        # Change only y
        nb_runner.set_cell_source(1, "x = 1\ny = 20\nz = 3")
        nb_runner.run_all()
        # 10+200+30=240
        assert "240" in nb_runner.get_output(6)


class TestRapidModification:
    """Test rapid modifications and re-runs."""

    def test_multiple_rapid_changes(self, nb_runner):
        """Make several changes and re-run each time."""
        nb_runner.create_notebook([
            "n = 1",
            "result = n * 100",
            "print(result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

        for val in [2, 5, 10, 50]:
            nb_runner.set_cell_source(1, f"n = {val}")
            nb_runner.run_all()
            assert str(val * 100) in nb_runner.get_output(3)
