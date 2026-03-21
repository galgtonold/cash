"""Batch 279 – Star unpacking and extended iterable unpacking.

Tests *args, **kwargs, and extended unpacking with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStarUnpacking:
    """Star unpacking edit patterns."""

    def test_star_rest_edit(self, nb_runner):
        """Edit list, star unpack head/*rest changes."""
        nb_runner.create_notebook([
            "data = [10, 20, 30, 40, 50]",
            "head, *rest = data\nprint(f'head = {head}, rest = {rest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head = 10" in out
        assert "rest = [20, 30, 40, 50]" in out

        nb_runner.set_cell_source(1, "data = [99, 88]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head = 99" in out
        assert "rest = [88]" in out

    def test_dict_merge_unpack_edit(self, nb_runner):
        """Edit dict, merge with ** changes."""
        nb_runner.create_notebook([
            "base = {'a': 1, 'b': 2}",
            "extra = {'c': 3}",
            "merged = {**base, **extra}\nprint(f'merged = {merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "base = {'a': 100, 'b': 200}")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'a': 100" in out
        assert "'c': 3" in out

    def test_function_args_kwargs_edit(self, nb_runner):
        """Edit function with *args/**kwargs."""
        nb_runner.create_notebook([
            "def combine(*args, **kwargs):\n    return list(args) + list(kwargs.values())",
            "result = combine(1, 2, x=10, y=20)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 10, 20]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def combine(*args, **kwargs):\n    return [a * 2 for a in args] + [v * 3 for v in kwargs.values()]",
        )
        nb_runner.run_all()
        assert "result = [2, 4, 30, 60]" in nb_runner.get_output(2)
