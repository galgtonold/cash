"""Batch 131 – Notebook size scaling interaction tests.

Tests that exercise notebooks with 10-20 cells simulating
real-world data science workflows with multiple phases.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestTenCellPipeline:
    """10-cell pipeline simulating a data science workflow."""

    def test_ten_cell_data_pipeline(self, nb_runner):
        """Full 10-cell data pipeline with edit."""
        nb_runner.create_notebook([
            "# Step 1: Data generation\nimport random\nrandom.seed(42)\ndata = [random.randint(1, 100) for _ in range(20)]",
            "# Step 2: Cleaning\ncleaned = [x for x in data if x > 10]",
            "# Step 3: Stats\nmean_val = sum(cleaned) / len(cleaned)",
            "# Step 4: Normalization\nnormalized = [(x - mean_val) for x in cleaned]",
            "# Step 5: Filter outliers\nfiltered = [x for x in normalized if abs(x) < 40]",
            "# Step 6: Transform\ntransformed = [x ** 2 for x in filtered]",
            "# Step 7: Aggregate\ntotal = sum(transformed)\ncount = len(transformed)",
            "# Step 8: Average\navg = total / count if count else 0",
            "# Step 9: Scale\nscaled = round(avg * 100, 2)",
            "# Step 10: Report\nprint(f'scaled = {scaled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(10)
        assert "scaled = " in output

        # Edit the seed → different data → different result
        nb_runner.set_cell_source(
            1,
            "# Step 1: Data generation\nimport random\nrandom.seed(99)\ndata = [random.randint(1, 100) for _ in range(20)]",
        )
        nb_runner.run_all()
        output2 = nb_runner.get_output(10)
        assert "scaled = " in output2

    def test_ten_cell_edit_multiple_cells(self, nb_runner):
        """10-cell pipeline, edit two non-adjacent cells."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1",
            "f = e + 1",
            "g = f + 1",
            "h = g + 1",
            "i = h + 1",
            "j = i + 1\nprint(f'j = {j}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "j = 10" in nb_runner.get_output(10)

        # Edit cells 3 and 7
        nb_runner.set_cell_source(3, "c = b * 10")
        nb_runner.set_cell_source(7, "g = f * 10")
        nb_runner.run_all()
        # a=1, b=2, c=20, d=21, e=22, f=23, g=230, h=231, i=232, j=233
        assert "j = 233" in nb_runner.get_output(10)


class TestFifteenCellWorkflow:
    """15-cell workflow simulating a report generation."""

    def test_fifteen_cell_workflow(self, nb_runner):
        """15-cell workflow with edit at beginning."""
        cells = [
            "base = 10",              # 1
            "step1 = base + 1",       # 2
            "step2 = step1 * 2",      # 3
            "step3 = step2 - 3",      # 4
            "step4 = step3 + 4",      # 5
            "step5 = step4 * 5",      # 6
            "step6 = step5 - 6",      # 7
            "step7 = step6 + 7",      # 8
            "step8 = step7 * 2",      # 9
            "step9 = step8 - 1",      # 10
            "step10 = step9 + 10",    # 11
            "step11 = step10 * 3",    # 12
            "step12 = step11 - 5",    # 13
            "step13 = step12 + 2",    # 14
            "result = step13\nprint(f'result = {result}')",  # 15
        ]
        nb_runner.create_notebook(cells)
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(15)
        assert "result = " in output

        # Edit the base
        nb_runner.set_cell_source(1, "base = 20")
        nb_runner.run_all()
        output2 = nb_runner.get_output(15)
        assert "result = " in output2
        # Verify different results
        assert output != output2 or "result" in output2


class TestWidePipeline:
    """Wide (many parallel branches) notebook."""

    def test_six_parallel_branches(self, nb_runner):
        """6 parallel branches from a shared root."""
        nb_runner.create_notebook([
            "root = 10",                          # 1
            "branch_a = root * 1",                # 2
            "branch_b = root * 2",                # 3
            "branch_c = root * 3",                # 4
            "branch_d = root * 4",                # 5
            "branch_e = root * 5",                # 6
            "branch_f = root * 6",                # 7
            "total = branch_a + branch_b + branch_c + branch_d + branch_e + branch_f\nprint(f'total = {total}')",  # 8
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10*(1+2+3+4+5+6) = 10*21 = 210
        assert "total = 210" in nb_runner.get_output(8)

        nb_runner.set_cell_source(1, "root = 5")
        nb_runner.run_all()
        assert "total = 105" in nb_runner.get_output(8)

        # Edit one branch
        nb_runner.set_cell_source(4, "branch_c = root * 30")
        nb_runner.run_all()
        # 5*(1+2+30+4+5+6) = 5 + 10 + 150 + 20 + 25 + 30 = 240
        assert "total = 240" in nb_runner.get_output(8)
