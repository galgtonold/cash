"""Batch 120 – Large notebook workflow interaction tests.

Tests that exercise 8+ cell notebooks with complex dependency
graphs, simulating real-world data analysis workflows.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestLargeLinearWorkflow:
    """8+ cell linear pipeline."""

    def test_eight_cell_pipeline(self, nb_runner):
        """8-cell linear pipeline, edit various cells."""
        nb_runner.create_notebook([
            "raw = list(range(1, 11))",
            "cleaned = [x for x in raw if x > 0]",
            "normalized = [x / max(cleaned) for x in cleaned]",
            "filtered = [x for x in normalized if x > 0.5]",
            "transformed = [x ** 2 for x in filtered]",
            "aggregated = sum(transformed)",
            "scaled = aggregated * 100",
            "result = round(scaled, 2)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(8)
        assert "result = " in output

        # Edit raw data
        nb_runner.set_cell_source(1, "raw = list(range(1, 6))")
        nb_runner.run_all()
        output2 = nb_runner.get_output(8)
        assert "result = " in output2

    def test_eight_cell_edit_middle(self, nb_runner):
        """8-cell pipeline, edit a cell in the middle."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1",
            "f = e + 1",
            "g = f + 1",
            "h = g + 1\nprint(f'h = {h}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "h = 8" in nb_runner.get_output(8)

        # Edit cell 4
        nb_runner.set_cell_source(4, "d = c * 10")
        nb_runner.run_all()
        # d = 3*10 = 30, e = 31, f = 32, g = 33, h = 34
        assert "h = 34" in nb_runner.get_output(8)


class TestLargeDAGWorkflow:
    """Complex dependency graph (DAG) with multiple paths."""

    def test_complex_dag(self, nb_runner):
        """
        DAG:
            raw_a -> proc_a -
                              \
            raw_b -> proc_b ---> combined -> result
                              /
            raw_c -> proc_c -
        """
        nb_runner.create_notebook([
            "raw_a = [1, 2, 3]",
            "raw_b = [10, 20]",
            "raw_c = [100]",
            "proc_a = sum(raw_a)",
            "proc_b = sum(raw_b)",
            "proc_c = sum(raw_c)",
            "combined = proc_a + proc_b + proc_c",
            "result = combined * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 6 + 30 + 100 = 136, * 2 = 272
        assert "result = 272" in nb_runner.get_output(8)

        # Edit one branch
        nb_runner.set_cell_source(1, "raw_a = [10, 20, 30]")
        nb_runner.run_all()
        # 60 + 30 + 100 = 190, * 2 = 380
        assert "result = 380" in nb_runner.get_output(8)

    def test_dag_edit_two_branches(self, nb_runner):
        """Edit two branches simultaneously."""
        nb_runner.create_notebook([
            "x = 1",
            "y = 2",
            "a = x * 10",
            "b = y * 10",
            "result = a + b\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(5)

        # Edit both roots
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.set_cell_source(2, "y = 10")
        nb_runner.run_all()
        assert "result = 150" in nb_runner.get_output(5)


class TestLargeWorkflowWithFunctions:
    """Large workflow using helper functions."""

    def test_helper_functions_workflow(self, nb_runner):
        """Multiple helper functions feeding into a pipeline."""
        nb_runner.create_notebook([
            "def clean(data):\n    return [x for x in data if x > 0]",
            "def scale(data, factor):\n    return [x * factor for x in data]",
            "def aggregate(data):\n    return sum(data)",
            "raw = [-1, 2, -3, 4, 5]",
            "cleaned = clean(raw)",
            "scaled = scale(cleaned, 10)",
            "result = aggregate(scaled)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # clean: [2, 4, 5], scale: [20, 40, 50], aggregate: 110
        assert "result = 110" in nb_runner.get_output(7)

        # Edit clean function
        nb_runner.set_cell_source(
            1, "def clean(data):\n    return [x for x in data if x >= 0]"
        )
        nb_runner.run_all()
        # clean: [0, 2, 0, 4, 5] — wait, 0 and negatives...
        # Actually clean([-1, 2, -3, 4, 5]) with x >= 0: [2, 4, 5]
        # Same result since -1 and -3 are < 0. Let me change to >=-3
        nb_runner.set_cell_source(
            1, "def clean(data):\n    return [abs(x) for x in data]"
        )
        nb_runner.run_all()
        # abs: [1, 2, 3, 4, 5], scale: [10, 20, 30, 40, 50], agg: 150
        assert "result = 150" in nb_runner.get_output(7)

    def test_edit_data_in_large_workflow(self, nb_runner):
        """Large workflow, only edit the data source."""
        nb_runner.create_notebook([
            "def process(x):\n    return x * 2 + 1",
            "data = [1, 2, 3]",
            "processed = [process(x) for x in data]",
            "total = sum(processed)",
            "avg = total / len(processed)",
            "print(f'avg = {avg:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # process: [3, 5, 7], total: 15, avg: 5.0
        assert "avg = 5.00" in nb_runner.get_output(6)

        # Edit data
        nb_runner.set_cell_source(2, "data = [10, 20, 30]")
        nb_runner.run_all()
        # process: [21, 41, 61], total: 123, avg: 41.0
        assert "avg = 41.00" in nb_runner.get_output(6)


class TestLargeWorkflowWithRestart:
    """Large workflow + kernel restart."""

    def test_restart_large_workflow(self, nb_runner):
        """Run large workflow, restart, edit, run again."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b * 2",
            "d = c + 3",
            "e = d ** 2",
            "print(f'e = {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=1, b=2, c=4, d=7, e=49
        assert "e = 49" in nb_runner.get_output(6)

        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "a = 2")
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=2, b=3, c=6, d=9, e=81
        assert "e = 81" in nb_runner.get_output(6)

    @pytest.mark.timeout(90)
    def test_restore_large_workflow_leaf(self, nb_runner):
        """Large workflow — after restart, run only the leaf cell."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 2",
            "z = y + 3",
            "w = z * 4",
            "result = w - 1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x=5, y=10, z=13, w=52, result=51
        assert "result = 51" in nb_runner.get_output(5)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(5)
        assert "result = 51" in nb_runner.get_output(5)
