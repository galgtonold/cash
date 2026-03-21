"""Batch 209 – Zip and enumerate interaction tests.

Tests editing cells with zip, enumerate, and itertools
patterns and verifying cache invalidation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestZipEnumerateEdits:
    """Editing zip and enumerate patterns."""

    def test_edit_zip_sources(self, nb_runner):
        """Edit one of two lists being zipped."""
        nb_runner.create_notebook([
            "names = ['Alice', 'Bob', 'Charlie']",
            "scores = [90, 85, 78]",
            "pairs = list(zip(names, scores))\nprint(pairs)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('Alice', 90)" in nb_runner.get_output(3)

        # Edit scores
        nb_runner.set_cell_source(2, "scores = [95, 88, 82]")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "('Alice', 95)" in out
        assert "('Bob', 88)" in out

    def test_edit_enumerate_start(self, nb_runner):
        """Edit list and re-enumerate."""
        nb_runner.create_notebook([
            "items = ['apple', 'banana', 'cherry']",
            "indexed = list(enumerate(items, start=1))\nfor i, item in indexed:\n    print(f'{i}: {item}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "1: apple" in nb_runner.get_output(2)

        # Edit items
        nb_runner.set_cell_source(1, "items = ['mango', 'kiwi', 'grape', 'plum']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "1: mango" in out
        assert "4: plum" in out

    def test_edit_zip_with_transform(self, nb_runner):
        """Edit transform applied to zipped pairs."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']\nvals = [1, 2, 3]",
            "result = dict(zip(keys, vals))\nprint(result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        # Change keys
        nb_runner.set_cell_source(1, "keys = ['x', 'y', 'z']\nvals = [10, 20, 30]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'x': 10" in out

    def test_edit_multi_zip(self, nb_runner):
        """Edit cells with multiple zip operations."""
        nb_runner.create_notebook([
            "first = [1, 2, 3]\nsecond = [4, 5, 6]",
            "sums = [a + b for a, b in zip(first, second)]\nprint(f'sums = {sums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sums = [5, 7, 9]" in nb_runner.get_output(2)

        # Double the first list values
        nb_runner.set_cell_source(1, "first = [10, 20, 30]\nsecond = [4, 5, 6]")
        nb_runner.run_all()
        assert "sums = [14, 25, 36]" in nb_runner.get_output(2)
