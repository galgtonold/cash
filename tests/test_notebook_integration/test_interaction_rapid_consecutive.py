"""Batch 175 – Multiple rapid consecutive edits interaction tests.

Tests making multiple rapid edits to the same cell and verifying
that each edit is properly picked up.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestRapidSameCellEdits:
    """Multiple rapid edits to the same cell."""

    def test_three_consecutive_edits(self, nb_runner):
        """Edit a cell three times consecutively."""
        nb_runner.create_notebook([
            "x = 1  # version 1",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 1" in nb_runner.get_output(2)

        # Edit 1
        nb_runner.set_cell_source(1, "x = 10  # version 2")
        nb_runner.run_all()
        assert "x = 10" in nb_runner.get_output(2)

        # Edit 2
        nb_runner.set_cell_source(1, "x = 100  # version 3")
        nb_runner.run_all()
        assert "x = 100" in nb_runner.get_output(2)

        # Edit 3
        nb_runner.set_cell_source(1, "x = 1000  # version 4")
        nb_runner.run_all()
        assert "x = 1000" in nb_runner.get_output(2)

    def test_oscillating_values(self, nb_runner):
        """Alternate between two values rapidly."""
        nb_runner.create_notebook([
            "val = 'A'  # oscillate val A",
            "print(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = A" in nb_runner.get_output(2)

        # Switch to B
        nb_runner.set_cell_source(1, "val = 'B'  # oscillate val B")
        nb_runner.run_all()
        assert "val = B" in nb_runner.get_output(2)

        # Back to A (new comment to differentiate)
        nb_runner.set_cell_source(1, "val = 'A'  # oscillate val A again")
        nb_runner.run_all()
        assert "val = A" in nb_runner.get_output(2)

    def test_edit_type_change(self, nb_runner):
        """Edit a cell to change variable type each time."""
        nb_runner.create_notebook([
            "data = 42  # type change start",
            "print(f'type = {type(data).__name__}, data = {data}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type = int" in nb_runner.get_output(2)

        # Change to string
        nb_runner.set_cell_source(1, "data = 'hello'  # type change str")
        nb_runner.run_all()
        assert "type = str" in nb_runner.get_output(2)

        # Change to list
        nb_runner.set_cell_source(1, "data = [1, 2, 3]  # type change list")
        nb_runner.run_all()
        assert "type = list" in nb_runner.get_output(2)

        # Change to dict
        nb_runner.set_cell_source(1, "data = {'a': 1}  # type change dict")
        nb_runner.run_all()
        assert "type = dict" in nb_runner.get_output(2)


class TestRapidDependentEdits:
    """Rapid edits to dependent cells."""

    def test_edit_producer_then_consumer_rapidly(self, nb_runner):
        """Edit both producer and consumer cells rapidly."""
        nb_runner.create_notebook([
            "x = 5  # producer",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Edit producer
        nb_runner.set_cell_source(1, "x = 50  # producer big")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)

        # Edit consumer
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 150" in nb_runner.get_output(2)

        # Edit both at once
        nb_runner.set_cell_source(1, "x = 7  # producer small")
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 8" in nb_runner.get_output(2)

    def test_rapid_formula_changes(self, nb_runner):
        """Rapidly change the formula applied to same input."""
        nb_runner.create_notebook([
            "n = 10  # input number",
            "result = n + 1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        formulas = [
            ("result = n * 2\nprint(f'result = {result}')", "20"),
            ("result = n ** 2\nprint(f'result = {result}')", "100"),
            ("result = n // 3\nprint(f'result = {result}')", "3"),
            ("result = n - 7\nprint(f'result = {result}')", "3"),
        ]
        for code, expected in formulas:
            nb_runner.set_cell_source(2, code)
            nb_runner.run_all()
            assert f"result = {expected}" in nb_runner.get_output(2)
