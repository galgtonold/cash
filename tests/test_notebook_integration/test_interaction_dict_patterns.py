"""Batch 185 – Dictionary manipulation pattern interaction tests.

Tests editing dict comprehensions, merges, nested dicts,
defaultdict patterns across cells.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDictComprehensionEdits:
    """Editing dict comprehension patterns."""

    def test_edit_dict_comprehension_expression(self, nb_runner):
        """Edit the value expression in a dict comprehension."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']",
            "d = {k: i for i, k in enumerate(keys)}\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = {'a': 0, 'b': 1, 'c': 2}" in nb_runner.get_output(2)

        # Change value expression
        nb_runner.set_cell_source(
            2, "d = {k: (i+1)*10 for i, k in enumerate(keys)}\nprint(f'd = {d}')"
        )
        nb_runner.run_all()
        assert "d = {'a': 10, 'b': 20, 'c': 30}" in nb_runner.get_output(2)

    def test_edit_dict_source_keys(self, nb_runner):
        """Edit the source keys list."""
        nb_runner.create_notebook([
            "names = ['alice', 'bob']  # dict source keys",
            "scores = {n: len(n) for n in names}\nprint(f'scores = {scores}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "scores = {'alice': 5, 'bob': 3}" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "names = ['charlie', 'dan', 'eve']  # dict source keys updated")
        nb_runner.run_all()
        assert "'charlie': 7" in nb_runner.get_output(2)
        assert "'eve': 3" in nb_runner.get_output(2)


class TestDictMergeEdits:
    """Editing dict merge patterns."""

    def test_edit_merge_operand(self, nb_runner):
        """Edit one dict in a merge operation."""
        nb_runner.create_notebook([
            "base = {'x': 1, 'y': 2}  # base dict",
            "override = {'y': 20, 'z': 30}  # override dict",
            "merged = {**base, **override}\nprint(f'merged = {merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'x': 1" in nb_runner.get_output(3)
        assert "'y': 20" in nb_runner.get_output(3)
        assert "'z': 30" in nb_runner.get_output(3)

        # Change override
        nb_runner.set_cell_source(2, "override = {'y': 200, 'w': 400}  # override dict changed")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'y': 200" in out
        assert "'w': 400" in out


    def test_edit_defaultdict_factory(self, nb_runner):
        """Edit defaultdict usage."""
        nb_runner.create_notebook([
            "from collections import defaultdict",
            "dd = defaultdict(int)\ndd['a'] += 1\ndd['b'] += 5\nprint(f'dd = {dict(dd)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        assert "'b': 5" in nb_runner.get_output(2)

        # Change factory and values
        nb_runner.set_cell_source(
            2,
            "dd = defaultdict(list)\ndd['a'].append(10)\ndd['b'].extend([20, 30])\nprint(f'dd = {dict(dd)}')",
        )
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': [10]" in out
        assert "'b': [20, 30]" in out
