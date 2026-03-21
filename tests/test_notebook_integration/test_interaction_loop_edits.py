"""Batch 143 – Loop and iteration pattern interaction tests.

Tests where users write loops across cells, edit loop bounds,
body, and iteration patterns, verifying cache consistency.
"""

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.stress, pytest.mark.timeout(45)]


class TestForLoopEdits:
    """For loop editing patterns."""

    def test_edit_loop_range(self, nb_runner):
        """Edit the range of a for loop."""
        nb_runner.create_notebook([
            "total = 0\nfor i in range(5):\n    total += i",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "total = 0\nfor i in range(10):\n    total += i")
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(2)

    def test_edit_loop_body(self, nb_runner):
        """Edit the body of a for loop."""
        nb_runner.create_notebook([
            "results = []\nfor i in range(5):\n    results.append(i * 2)",
            "total = sum(results)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # [0,2,4,6,8] -> 20
        assert "total = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1, "results = []\nfor i in range(5):\n    results.append(i ** 2)"
        )
        nb_runner.run_all()
        # [0,1,4,9,16] -> 30
        assert "total = 30" in nb_runner.get_output(2)

    def test_edit_source_data_for_loop(self, nb_runner):
        """Edit the data that a loop iterates over."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "doubled = [x * 2 for x in data]",
            "total = sum(doubled)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 12" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data = [10, 20, 30, 40]")
        nb_runner.run_all()
        assert "total = 200" in nb_runner.get_output(3)


class TestComprehensionEdits:
    """List/dict/set comprehension edits."""

    def test_edit_list_comprehension_filter(self, nb_runner):
        """Edit filter in list comprehension."""
        nb_runner.create_notebook([
            "numbers = list(range(20))",
            "evens = [x for x in numbers if x % 2 == 0]\nprint(f'count = {len(evens)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(2)

        # Change filter to multiples of 3
        nb_runner.set_cell_source(
            2, "threes = [x for x in numbers if x % 3 == 0]\nprint(f'count = {len(threes)}')"
        )
        nb_runner.run_all()
        assert "count = 7" in nb_runner.get_output(2)

    def test_edit_dict_comprehension(self, nb_runner):
        """Edit dict comprehension."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']",
            "d = {k: i for i, k in enumerate(keys)}\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "d = {k: (i + 1) * 10 for i, k in enumerate(keys)}\nprint(f'd = {d}')"
        )
        nb_runner.run_all()
        assert "'a': 10" in nb_runner.get_output(2)


class TestWhileLoopEdits:
    """While loop editing patterns."""

    def test_edit_while_condition(self, nb_runner):
        """Edit while loop condition."""
        nb_runner.create_notebook([
            "count = 0\nval = 1\nwhile val < 100:\n    val *= 2\n    count += 1",
            "print(f'count = {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 7" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1, "count = 0\nval = 1\nwhile val < 1000:\n    val *= 2\n    count += 1"
        )
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(2)

    def test_edit_while_body(self, nb_runner):
        """Edit while loop body."""
        nb_runner.create_notebook([
            "n = 10\ntotal = 0\nwhile n > 0:\n    total += n\n    n -= 1",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1, "n = 10\ntotal = 0\nwhile n > 0:\n    total += n * n\n    n -= 1"
        )
        nb_runner.run_all()
        assert "total = 385" in nb_runner.get_output(2)
