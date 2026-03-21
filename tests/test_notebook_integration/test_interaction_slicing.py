"""Batch 206 – List slicing and indexing interaction tests.

Tests editing list slice operations, negative indexing,
step slicing, and their propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestSlicingEdits:
    """Editing list slicing patterns."""

    def test_edit_slice_range(self, nb_runner):
        """Edit slice start/stop."""
        nb_runner.create_notebook([
            "data = list(range(10))  # slice source",
            "result = data[2:5]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "result = data[5:8]\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = [5, 6, 7]" in nb_runner.get_output(2)

    def test_edit_slice_step(self, nb_runner):
        """Edit slice step."""
        nb_runner.create_notebook([
            "nums = list(range(20))  # slice step source",
            "result = nums[::2]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "result = nums[::5]\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = [0, 5, 10, 15]" in nb_runner.get_output(2)

    def test_edit_negative_index(self, nb_runner):
        """Edit negative indexing."""
        nb_runner.create_notebook([
            "items = ['a', 'b', 'c', 'd', 'e']  # neg index source",
            "result = items[-1]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = e" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "result = items[-3:]\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = ['c', 'd', 'e']" in nb_runner.get_output(2)

    def test_edit_source_then_slice(self, nb_runner):
        """Edit the source list, verify slice updates."""
        nb_runner.create_notebook([
            "seq = [10, 20, 30, 40, 50]  # slice propagation source",
            "first_half = seq[:3]\nprint(f'first_half = {first_half}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first_half = [10, 20, 30]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "seq = [100, 200, 300, 400]  # slice propagation source v2")
        nb_runner.run_all()
        assert "first_half = [100, 200, 300]" in nb_runner.get_output(2)
