"""Batch 136 – Complex realistic interaction scenarios.

End-to-end scenarios mimicking real data science workflows
where users iterate on their analysis, changing parameters,
adding cells, removing cells, and re-running.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestDataExplorationWorkflow:
    """Simulate a user exploring data iteratively."""

    def test_explore_then_refine(self, nb_runner):
        """User explores, then refines analysis."""
        nb_runner.create_notebook([
            "data = list(range(1, 21))",
            "mean_val = sum(data) / len(data)\nprint(f'mean = {mean_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 10.5" in nb_runner.get_output(2)

        # User decides to filter first
        nb_runner.set_cell_source(
            2,
            "filtered = [x for x in data if x > 10]\nmean_val = sum(filtered) / len(filtered)\nprint(f'mean = {mean_val}')",
        )
        nb_runner.run_all()
        assert "mean = 15.5" in nb_runner.get_output(2)

        # User changes the data source
        nb_runner.set_cell_source(1, "data = list(range(1, 101))")
        nb_runner.run_all()
        assert "mean = 55.5" in nb_runner.get_output(2)

    def test_iterative_parameter_tuning(self, nb_runner):
        """User tunes parameters across multiple iterations."""
        nb_runner.create_notebook([
            "threshold = 50\nscale = 2",
            "data = list(range(100))",
            "filtered = [x for x in data if x > threshold]",
            "result = sum(x * scale for x in filtered)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(4)
        assert "result = " in output1

        # Tune threshold
        nb_runner.set_cell_source(1, "threshold = 75\nscale = 2")
        nb_runner.run_all()
        output2 = nb_runner.get_output(4)
        assert "result = " in output2

        # Tune scale
        nb_runner.set_cell_source(1, "threshold = 75\nscale = 10")
        nb_runner.run_all()
        output3 = nb_runner.get_output(4)
        assert "result = " in output3

        # Values should be different
        assert output1 != output2 or output2 != output3


class TestModelRetraining:
    """Simulate a model training workflow with re-iterations."""

    def test_change_hyperparameters(self, nb_runner):
        """Change hyperparameters and retrain."""
        nb_runner.create_notebook([
            "# Hyperparams\nlr = 0.01\nepochs = 10",
            "# Training sim\nimport random\nrandom.seed(42)\nloss = 1.0\nfor e in range(epochs):\n    loss *= (1 - lr)\nfinal_loss = round(loss, 4)",
            "print(f'loss = {final_loss}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(3)
        assert "loss = " in output1

        # Increase learning rate
        nb_runner.set_cell_source(1, "# Hyperparams\nlr = 0.1\nepochs = 10")
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert "loss = " in output2

        # Higher lr → lower loss
        loss1 = float(output1.split("loss = ")[1].strip())
        loss2 = float(output2.split("loss = ")[1].strip())
        assert loss2 < loss1


class TestReportGeneration:
    """Simulate report generation with formatting changes."""

    def test_change_format_then_data(self, nb_runner):
        """Change formatting, then change data."""
        nb_runner.create_notebook([
            "sales = [100, 200, 300, 400, 500]",
            "total = sum(sales)\navg = total / len(sales)",
            "report = f'Total: {total}, Avg: {avg}'\nprint(report)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Total: 1500, Avg: 300.0" in nb_runner.get_output(3)

        # Change format
        nb_runner.set_cell_source(
            3,
            "report = f'Sales Report: total=${total}, average=${avg}'\nprint(report)",
        )
        nb_runner.run_all()
        assert "Sales Report: total=$1500, average=$300.0" in nb_runner.get_output(3)

        # Change data
        nb_runner.set_cell_source(1, "sales = [1000, 2000, 3000]")
        nb_runner.run_all()
        assert "total=$6000, average=$2000.0" in nb_runner.get_output(3)


class TestDebuggingWorkflow:
    """Simulate debugging workflow — add prints, fix, remove prints."""

    def test_add_debug_fix_remove(self, nb_runner):
        """Add debug output, fix bug, remove debug."""
        nb_runner.create_notebook([
            "numbers = [1, 2, 3, 4, 5]",
            "# Bug: using wrong formula\nresult = sum(numbers) / (len(numbers) + 1)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # Bug: divides by 6 instead of 5
        output = nb_runner.get_output(2)
        assert "result = 2.5" in output

        # Fix the bug
        nb_runner.set_cell_source(
            2,
            "# Fixed formula\nresult = sum(numbers) / len(numbers)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 3.0" in nb_runner.get_output(2)


class TestBackAndForthEditing:
    """User goes back and forth between cells."""

    def test_edit_cell1_then_cell3_then_cell1_again(self, nb_runner):
        """Edit cell 1, then cell 3, then cell 1 again."""
        nb_runner.create_notebook([
            "base = 10",
            "mid = base * 2",
            "final = mid + 5\nprint(f'final = {final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final = 25" in nb_runner.get_output(3)

        # Edit cell 1
        nb_runner.set_cell_source(1, "base = 20")
        nb_runner.run_all()
        assert "final = 45" in nb_runner.get_output(3)

        # Edit cell 3
        nb_runner.set_cell_source(3, "final = mid + 100\nprint(f'final = {final}')")
        nb_runner.run_all()
        assert "final = 140" in nb_runner.get_output(3)

        # Edit cell 1 again
        nb_runner.set_cell_source(1, "base = 1")
        nb_runner.run_all()
        assert "final = 102" in nb_runner.get_output(3)

    def test_oscillate_between_two_values(self, nb_runner):
        """Toggle a value back and forth — each cell gets unique code."""
        nb_runner.create_notebook([
            "mode = 'A'",
            "result = 100 if mode == 'A' else 200\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Switch to B
        nb_runner.set_cell_source(1, "mode = 'B'")
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)

        # Switch back to A with different code
        nb_runner.set_cell_source(1, "mode = 'A'  # restored")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    def test_sequential_distinct_changes(self, nb_runner):
        """Make many distinct changes to same cell — each unique."""
        nb_runner.create_notebook([
            "val = 1",
            "out = val * 10\nprint(f'out = {out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 10" in nb_runner.get_output(2)

        for v in [2, 3, 5, 7]:
            nb_runner.set_cell_source(1, f"val = {v}")
            nb_runner.run_all()
            assert f"out = {v * 10}" in nb_runner.get_output(2)
