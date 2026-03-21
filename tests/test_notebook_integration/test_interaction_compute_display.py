"""Batch 180 – Mixed computation and display pattern tests.

Tests combining computation cells with display/print cells,
editing either the computation or the display logic.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestComputeDisplaySplit:
    """Computation and display in separate cells."""

    def test_edit_computation_only(self, nb_runner):
        """Edit computation cell, display cell stays same."""
        nb_runner.create_notebook([
            "values = [1, 2, 3, 4, 5]  # compute data",
            "total = sum(values)\navg = total / len(values)",
            "print(f'total={total} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15 avg=3.0" in nb_runner.get_output(3)

        # Edit source data
        nb_runner.set_cell_source(1, "values = [10, 20, 30]  # compute data changed")
        nb_runner.run_all()
        assert "total=60 avg=20.0" in nb_runner.get_output(3)

    def test_edit_display_only(self, nb_runner):
        """Edit display cell, computation stays same."""
        nb_runner.create_notebook([
            "nums = [2, 4, 6, 8]  # display nums",
            "s = sum(nums)\nm = max(nums)",
            "print(f'sum={s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=20" in nb_runner.get_output(3)

        # Change display to show max
        nb_runner.set_cell_source(3, "print(f'max={m}')")
        nb_runner.run_all()
        assert "max=8" in nb_runner.get_output(3)

    def test_edit_both_compute_and_display(self, nb_runner):
        """Edit both computation and display cells."""
        nb_runner.create_notebook([
            "x = 7  # base value for compute+display",
            "squared = x ** 2",
            "print(f'squared = {squared}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squared = 49" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10  # base value updated")
        nb_runner.set_cell_source(2, "cubed = x ** 3")
        nb_runner.set_cell_source(3, "print(f'cubed = {cubed}')")
        nb_runner.run_all()
        assert "cubed = 1000" in nb_runner.get_output(3)


class TestFormattedOutput:
    """Formatted output patterns with edits."""

    def test_table_format_edit(self, nb_runner):
        """Edit table-like output formatting."""
        nb_runner.create_notebook([
            "items = [('apple', 3), ('banana', 5)]  # items for table",
            "for name, count in items:\n    print(f'{name}: {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple: 3" in out

        # Change to aligned format
        nb_runner.set_cell_source(
            2, "for name, count in items:\n    print(f'{name:>10s} | {count:>3d}')"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apple" in out
        assert "3" in out

    def test_multi_line_output_edit(self, nb_runner):
        """Edit multi-line output generation."""
        nb_runner.create_notebook([
            "header = 'Results'  # output header",
            "lines = [header, '-' * len(header), 'Item 1: OK', 'Item 2: OK']\nprint('\\n'.join(lines))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Results" in out
        assert "Item 1: OK" in out

        nb_runner.set_cell_source(1, "header = 'Summary'  # output header changed")
        nb_runner.run_all()
        assert "Summary" in nb_runner.get_output(2)
