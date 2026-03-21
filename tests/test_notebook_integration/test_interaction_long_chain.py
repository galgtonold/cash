"""Batch 268 – Long chain dependency propagation (5+ cells).

Tests editing early cell in long chain, verifying final cell updates.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestLongChainPropagation:
    """Long dependency chain edit patterns."""

    def test_six_cell_chain(self, nb_runner):
        """Edit cell 1 in 6-cell chain, cell 6 reflects."""
        nb_runner.create_notebook([
            "base = 10",
            "step1 = base + 5",
            "step2 = step1 * 2",
            "step3 = step2 - 3",
            "step4 = step3 // 4",
            "result = step4\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10+5=15, 15*2=30, 30-3=27, 27//4=6
        assert "result = 6" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_all()
        # 100+5=105, 105*2=210, 210-3=207, 207//4=51
        assert "result = 51" in nb_runner.get_output(6)

    def test_edit_middle_of_long_chain(self, nb_runner):
        """Edit middle cell (3 of 5), tail updates."""
        nb_runner.create_notebook([
            "x = 2",
            "y = x * 3",
            "z = y + 10",
            "w = z ** 2",
            "final = w - 1\nprint(f'final = {final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x=2, y=6, z=16, w=256, final=255
        assert "final = 255" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "z = y + 100")
        nb_runner.run_all()
        # x=2, y=6, z=106, w=11236, final=11235
        assert "final = 11235" in nb_runner.get_output(5)

    def test_branching_chain(self, nb_runner):
        """Two branches merge in final cell."""
        nb_runner.create_notebook([
            "a = 10",
            "b = 20",
            "left = a * 2",
            "right = b * 3",
            "combined = left + right\nprint(f'combined = {combined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10*2 + 20*3 = 20 + 60 = 80
        assert "combined = 80" in nb_runner.get_output(5)

        # Edit one branch source
        nb_runner.set_cell_source(1, "a = 50")
        nb_runner.run_all()
        # 50*2 + 20*3 = 100 + 60 = 160
        assert "combined = 160" in nb_runner.get_output(5)
