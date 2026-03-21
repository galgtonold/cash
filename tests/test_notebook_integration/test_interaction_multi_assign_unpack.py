"""
Batch 334: multiple assignment / unpacking patterns with caching.
Tests tuple unpacking, star expressions, swap, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultipleAssignUnpack:
    """Test multiple assignment and unpacking caching."""

    def test_tuple_unpack(self, nb_runner):
        """Basic tuple unpacking with caching."""
        nb_runner.create_notebook([
            "data = (10, 20, 30)",
            "a, b, c = data",
            "total = a + b + c\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total=60" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "total=60" in out2

    def test_star_unpack_edit(self, nb_runner):
        """Star unpacking with edit propagation."""
        nb_runner.create_notebook([
            "values = [1, 2, 3, 4, 5]",
            "first, *middle, last = values",
            "print(f'first={first} middle={middle} last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "first=1" in out
        assert "middle=[2, 3, 4]" in out
        assert "last=5" in out

        nb_runner.set_cell_source(1, "values = [10, 20, 30]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "first=10" in out2
        assert "middle=[20]" in out2
        assert "last=30" in out2

    def test_swap_pattern(self, nb_runner):
        """Variable swap pattern with caching."""
        nb_runner.create_notebook([
            "x = 'hello'\ny = 'world'",
            "x, y = y, x",
            "print(f'x={x} y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "x=world" in out
        assert "y=hello" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "x=world" in out2
