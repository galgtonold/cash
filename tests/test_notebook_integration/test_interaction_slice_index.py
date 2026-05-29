"""Batch 266 – Slice and index pattern edits.

Tests list slicing, indexing operations with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSliceIndexEdits:
    """Slice and index edit patterns."""

    def test_slice_params_edit(self, nb_runner):
        """Edit slice parameters, result updates."""
        nb_runner.create_notebook([
            "data = list(range(10, 21))",
            "start = 2\nstop = 7",
            "sliced = data[start:stop]\nprint(f'sliced = {sliced}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sliced = [12, 13, 14, 15, 16]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "start = 0\nstop = 3")
        nb_runner.run_all()
        assert "sliced = [10, 11, 12]" in nb_runner.get_output(3)

    def test_step_slice_edit(self, nb_runner):
        """Edit step in slice."""
        nb_runner.create_notebook([
            "nums = list(range(20))",
            "step = 2",
            "selected = nums[::step]\nprint(f'selected = {selected}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "selected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "step = 5")
        nb_runner.run_all()
        assert "selected = [0, 5, 10, 15]" in nb_runner.get_output(3)

