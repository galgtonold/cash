"""Batch 240 – Unpacking and star expression edit tests.

Tests editing cells with tuple unpacking, star expressions,
and chained assignments.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestUnpackingStarEdits:
    """Editing cells with unpacking and star expressions."""

    def test_edit_triple_unpack_values(self, nb_runner):
        """Edit a 3-variable tuple unpacking."""
        nb_runner.create_notebook([
            "a, b, c = 1, 2, 3",
            "total = a + b + c\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a, b, c = 10, 20, 30")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_edit_star_first_to_last(self, nb_runner):
        """Switch from first/*rest to *init/last pattern."""
        nb_runner.create_notebook([
            "first, *rest = [1, 2, 3, 4, 5]",
            "print(f'first={first} rest={rest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "*init, last = [1, 2, 3, 4, 5]")
        nb_runner.set_cell_source(2, "print(f'init={init} last={last}')")
        nb_runner.run_all()
        assert "last=5" in nb_runner.get_output(2)

    def test_edit_chained_value(self, nb_runner):
        """Edit chained assignment x = y = z = value."""
        nb_runner.create_notebook([
            "x = y = z = 5",
            "total = x + y + z\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = y = z = 10")
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)
