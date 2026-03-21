"""Batch 163 – Deep dependency chain interaction tests.

Tests with long chains of cells (5+ cells) where a change at any
point in the chain must properly propagate through all downstream cells.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDeepChainPropagation:
    """Deep dependency chains with edits at various points."""

    def test_edit_head_of_chain(self, nb_runner):
        """5-cell chain, edit the first cell."""
        nb_runner.create_notebook([
            "a = 1  # head",
            "b = a + 1  # step 2",
            "c = b * 2  # step 3",
            "d = c + 10  # step 4",
            "result = d * 3\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=1, b=2, c=4, d=14, result=42
        assert "result = 42" in nb_runner.get_output(5)

        # Edit head
        nb_runner.set_cell_source(1, "a = 10  # head changed")
        nb_runner.run_all()
        # a=10, b=11, c=22, d=32, result=96
        assert "result = 96" in nb_runner.get_output(5)

    def test_edit_middle_of_chain(self, nb_runner):
        """5-cell chain, edit the middle cell."""
        nb_runner.create_notebook([
            "x = 2  # start",
            "y = x * 3  # middle1",
            "z = y + 1  # middle2",
            "w = z ** 2  # step4",
            "out = w - 1\nprint(f'out = {out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x=2, y=6, z=7, w=49, out=48
        assert "out = 48" in nb_runner.get_output(5)

        # Edit middle
        nb_runner.set_cell_source(3, "z = y + 100  # middle2 boosted")
        nb_runner.run_all()
        # x=2, y=6, z=106, w=11236, out=11235
        assert "out = 11235" in nb_runner.get_output(5)

    def test_edit_tail_of_chain(self, nb_runner):
        """5-cell chain, edit the last cell."""
        nb_runner.create_notebook([
            "p = 5  # start",
            "q = p * 2  # step2",
            "r = q + 3  # step3",
            "s = r - 1  # step4",
            "final = s\nprint(f'final = {final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # p=5, q=10, r=13, s=12, final=12
        assert "final = 12" in nb_runner.get_output(5)

        # Edit output cell
        nb_runner.set_cell_source(5, "final = s * 100\nprint(f'final = {final}')")
        nb_runner.run_all()
        assert "final = 1200" in nb_runner.get_output(5)

    def test_edit_two_points_in_chain(self, nb_runner):
        """Edit two non-adjacent cells in a chain simultaneously."""
        nb_runner.create_notebook([
            "a = 1  # chain start",
            "b = a + 10  # chain step2",
            "c = b * 2  # chain step3",
            "d = c + 5  # chain step4",
            "e = d * 3\nprint(f'e = {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=1, b=11, c=22, d=27, e=81
        assert "e = 81" in nb_runner.get_output(5)

        # Edit cells 1 and 4
        nb_runner.set_cell_source(1, "a = 100  # chain start big")
        nb_runner.set_cell_source(4, "d = c + 1000  # chain step4 big")
        nb_runner.run_all()
        # a=100, b=110, c=220, d=1220, e=3660
        assert "e = 3660" in nb_runner.get_output(5)


class TestChainWithFunctions:
    """Deep chains involving function definitions."""

    def test_function_chain_edit(self, nb_runner):
        """Chain where each cell defines a function using the previous."""
        nb_runner.create_notebook([
            "def step1(x):\n    return x + 1",
            "def step2(x):\n    return step1(x) * 2",
            "def step3(x):\n    return step2(x) + 10",
            "result = step3(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # step1(5)=6, step2(5)=12, step3(5)=22
        assert "result = 22" in nb_runner.get_output(4)

        # Edit step1
        nb_runner.set_cell_source(1, "def step1(x):\n    return x + 100")
        nb_runner.run_all()
        # step1(5)=105, step2(5)=210, step3(5)=220
        assert "result = 220" in nb_runner.get_output(4)

    def test_lambda_chain_edit(self, nb_runner):
        """Chain of lambda functions with edits."""
        nb_runner.create_notebook([
            "fn1 = lambda x: x * 2",
            "fn2 = lambda x: fn1(x) + 3",
            "fn3 = lambda x: fn2(x) ** 2",
            "out = fn3(4)\nprint(f'out = {out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # fn1(4)=8, fn2(4)=11, fn3(4)=121
        assert "out = 121" in nb_runner.get_output(4)

        # Edit fn1
        nb_runner.set_cell_source(1, "fn1 = lambda x: x * 10")
        nb_runner.run_all()
        # fn1(4)=40, fn2(4)=43, fn3(4)=1849
        assert "out = 1849" in nb_runner.get_output(4)
