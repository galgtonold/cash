"""Batch 124 – Mixed assignment types + cell edit interaction tests.

Tests that exercise augmented assignments, tuple/list unpacking,
walrus operator, global/nonlocal, and compound assignments.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestAugmentedAssignment:
    """Augmented assignment operators + edits."""

    def test_augmented_add_edit(self, nb_runner):
        """Edit augmented addition."""
        nb_runner.create_notebook([
            "x = 10",
            "x += 5",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "x += 20")
        nb_runner.run_all()
        assert "x = 30" in nb_runner.get_output(3)

    def test_augmented_mul_edit_base(self, nb_runner):
        """Edit the base value before augmented multiply."""
        nb_runner.create_notebook([
            "x = 3",
            "x *= 4",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 12" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "x = 40" in nb_runner.get_output(3)

    def test_chain_of_augmented_ops(self, nb_runner):
        """Chain of augmented assignments, edit one in the middle."""
        nb_runner.create_notebook([
            "v = 100",
            "v -= 10  # subtract",
            "v *= 2  # double",
            "v //= 3  # floor divide",
            "print(f'v = {v}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 100 - 10 = 90, * 2 = 180, // 3 = 60
        assert "v = 60" in nb_runner.get_output(5)

        nb_runner.set_cell_source(2, "v -= 50  # subtract more")
        nb_runner.run_all()
        # 100 - 50 = 50, * 2 = 100, // 3 = 33
        assert "v = 33" in nb_runner.get_output(5)


class TestUnpackingEdits:
    """Tuple/list unpacking + cell edits."""

    def test_tuple_unpack_edit_source(self, nb_runner):
        """Edit the source of a tuple unpack."""
        nb_runner.create_notebook([
            "data = (1, 2, 3)",
            "a, b, c = data",
            "result = a + b + c\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data = (10, 20, 30)")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(3)

    def test_star_unpack_edit(self, nb_runner):
        """Edit data with star unpacking."""
        nb_runner.create_notebook([
            "values = [1, 2, 3, 4, 5]",
            "first, *rest = values",
            "result = first + sum(rest)\nprint(f'first = {first}, rest_sum = {sum(rest)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first = 1, rest_sum = 14" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "values = [100, 1, 1, 1]")
        nb_runner.run_all()
        assert "first = 100, rest_sum = 3" in nb_runner.get_output(3)

    def test_nested_unpack_edit(self, nb_runner):
        """Nested unpacking + edit."""
        nb_runner.create_notebook([
            "pair = ((1, 2), (3, 4))",
            "(a, b), (c, d) = pair",
            "result = a * d - b * c\nprint(f'det = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1*4 - 2*3 = -2
        assert "det = -2" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "pair = ((3, 1), (2, 4))")
        nb_runner.run_all()
        # 3*4 - 1*2 = 10
        assert "det = 10" in nb_runner.get_output(3)


class TestComprehensionEdits:
    """List/dict/set comprehensions + edits."""

    def test_list_comp_edit_filter(self, nb_runner):
        """Edit the filter in a list comprehension."""
        nb_runner.create_notebook([
            "data = list(range(20))",
            "evens = [x for x in data if x % 2 == 0]",
            "result = sum(evens)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0+2+4+6+8+10+12+14+16+18 = 90
        assert "result = 90" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "evens = [x for x in data if x % 3 == 0]"
        )
        nb_runner.run_all()
        # 0+3+6+9+12+15+18 = 63
        assert "result = 63" in nb_runner.get_output(3)

    def test_dict_comp_edit(self, nb_runner):
        """Edit a dict comprehension."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']",
            "d = {k: i for i, k in enumerate(keys)}",
            "result = d['b']\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "d = {k: (i + 1) * 10 for i, k in enumerate(keys)}"
        )
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

    def test_nested_comprehension_edit(self, nb_runner):
        """Nested comprehension (matrix) + edit."""
        nb_runner.create_notebook([
            "n = 3",
            "matrix = [[i * n + j for j in range(n)] for i in range(n)]",
            "flat = [x for row in matrix for x in row]\nresult = sum(flat)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0+1+2+3+4+5+6+7+8 = 36
        assert "result = 36" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "n = 4")
        nb_runner.run_all()
        # 0+1+...+15 = 120
        assert "result = 120" in nb_runner.get_output(3)


class TestMultipleReturnEdits:
    """Functions returning multiple values + cell edits."""

    def test_multi_return_edit(self, nb_runner):
        """Edit function that returns multiple values."""
        nb_runner.create_notebook([
            "def stats(data):\n    return min(data), max(data), sum(data) / len(data)",
            "lo, hi, avg = stats([1, 2, 3, 4, 5])",
            "print(f'lo={lo} hi={hi} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=5 avg=3.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "lo, hi, avg = stats([10, 20, 30])"
        )
        nb_runner.run_all()
        assert "lo=10 hi=30 avg=20.0" in nb_runner.get_output(3)

    def test_edit_multi_return_function(self, nb_runner):
        """Edit the function body that returns multiple values."""
        nb_runner.create_notebook([
            "def analyze(x):\n    return x, x * 2, x * 3",
            "a, b, c = analyze(5)",
            "result = a + b + c\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 5 + 10 + 15 = 30
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1, "def analyze(x):\n    return x, x ** 2, x ** 3"
        )
        nb_runner.run_all()
        # 5 + 25 + 125 = 155
        assert "result = 155" in nb_runner.get_output(3)
